import vkdispatch as vd
from ..limits import check_buffer_bytes
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import Buff, f32, c64, v2, Const

import numpy as np

from ..plan import Comparator
from ..weights import RadialWeights, edge_mask   # edge_mask: one definition, in weights.py
from .fused_conv import FusedNormState, make_fused_norm_convolution_shader
from ..ctf.ctf import image_ctf_value

@vd.map
def crop_mapping(in_buff: Buff[f32], sum_buffer: Buff[vc.v3], micrograph_shape: Const[vc.iv2], template_width: Const[vc.i32]):
    """Pads each template, centred, into a (micrograph_shape) grid as (t - mean_edge) / std_box.

    Subtracting the mean of the box's edge pixels (as cisTEM and Leopard-EM do) makes the template ~0 at the box
    edge, so the zero padding adds no step. Dividing by std_box only keeps values O(1); the score is normalized by
    the template's exact (whitened) norm later, so this scale cancels. template_width = the real box width w: the
    rfft layout (h, w//2 + 1) holds w + 1 (odd w) or w + 2 (even w) floats per row, so w cannot be read off the
    buffer shape (before: (w//2) * 2 was assumed, which dropped the last column of odd-width boxes)."""
    read_op = vd.fft.read_op()

    ind = read_op.io_index.to_dtype(vc.i32)

    read_index_3d = vc.new_ivec3_register()
    read_index_3d.x = ind // (micrograph_shape.x * (micrograph_shape.y + 2))
    read_index_3d.y = (ind // (micrograph_shape.y + 2)) % micrograph_shape.x
    read_index_3d.z = ind % (micrograph_shape.y + 2)

    read_index_3d.y -= micrograph_shape.x // 2
    read_index_3d.z -= micrograph_shape.y // 2

    read_index_3d.y += in_buff.shape.y // 2
    read_index_3d.z += in_buff.shape.z - 1

    with vc.if_block(vc.any(
            read_index_3d.y < 0,
            read_index_3d.y >= in_buff.shape.y,
            read_index_3d.z < 0,
            read_index_3d.z >= template_width
        )):
        read_op.register.real = 0.0
        read_op.register.imag = 0.0
    with vc.else_block():
        span_x = in_buff.shape.y * in_buff.shape.z * 2
        span_y = in_buff.shape.z * 2

        read_index_flat = (read_index_3d.x * span_x + read_index_3d.y * span_y + read_index_3d.z).to_register()

        sums = sum_buffer[read_index_3d.x].to_register()
        n_box = in_buff.shape.y * template_width
        n_edge = 2 * in_buff.shape.y + 2 * template_width - 4
        mean_box = (sums.x / n_box).to_register()
        std_box = vc.sqrt(sums.y / n_box - mean_box * mean_box).to_register()

        read_op.register.real = (in_buff[read_index_flat] - sums.z / n_edge) / std_box
        read_op.register.imag = 0

@vd.shader(exec_size=lambda args: args.buf.size)
def fill_buffer(buf: Buff[c64], val: Const[c64] = 0):
    buf[vc.global_invocation_id().x] = val

_calc_sums_cache = {}

def calc_sums(wave):
    """Per template: (sum, sum of squares, sum over the box's edge pixels) of the real-space box.

    The real box (h, w) is stored in rfft layout (h, w//2 + 1) complex: complex column z holds the real pixels
    2z (.real) and 2z + 1 (.imag); pixels at or beyond w are padding (for odd w, the .imag of column w//2). One
    reduction per box width (w is not recoverable from the buffer shape)."""
    width = int(wave.real_shape[-1])
    if width not in _calc_sums_cache:
        _calc_sums_cache[width] = _make_calc_sums(width)
    return _calc_sums_cache[width](wave)

def _make_calc_sums(width: int):
    last = width - 1

    @vd.reduce.map_reduce(vd.reduce.SubgroupAdd, axes=[1, 2])
    def calc_sums_w(wave: Buff[c64]) -> vc.v3:
        ind = vd.reduce.mapped_io_index()
        c = vc.ravel_index(ind, wave.shape).to_register()
        result = vc.new_vec3_register(0, 0, 0)
        v = wave[ind].to_register()
        x0 = (c.z * 2).to_register()                       # real column of .real; .imag is x0 + 1
        re = vc.new_float_register(0)
        im = vc.new_float_register(0)
        with vc.if_block(x0 <= last):
            re[:] = v.real
        with vc.if_block(x0 + 1 <= last):
            im[:] = v.imag
        result.x = re + im
        result.y = re * re + im * im
        with vc.if_block(vc.any(c.y == 0, c.y == wave.shape.y - 1)):
            result.z = re + im                              # first / last row: every pixel is an edge pixel
        with vc.else_block():
            with vc.if_block(x0 == 0):
                result.z = result.z + re                    # first column
            with vc.if_block(x0 == last):
                result.z = result.z + re                    # last column (odd w: a .real)
            with vc.if_block(x0 + 1 == last):
                result.z = result.z + im                    # last column (even w: an .imag)
        return result

    return calc_sums_w

@vd.map
def whitened_norm_map(carrier: Buff[c64], template_ft: Buff[c64], weights: Buff[f32], template_count: Const[vc.i32],
                      last_column_single: Const[vc.i32]) -> f32:
    """One term of sum_k A_k |T_k|^2 over the lag grid for the (micrograph m, template k) plane of `carrier`
    (shape (M*K, G_h, G_w//2 + 1); only its shape is used). The rfft half-plane counts twice, except column 0
    and, for even G_w, the Nyquist column."""
    ind = vd.reduce.mapped_io_index()
    c = vc.ravel_index(ind, carrier.shape).to_register()
    mk = c.x.to_dtype(vc.i32).to_register()
    m = (mk // template_count).to_register()
    k = (mk - m * template_count).to_register()
    cz = c.z.to_dtype(vc.i32).to_register()
    in_plane = (c.y.to_dtype(vc.i32) * carrier.shape.z + cz).to_register()
    plane = carrier.shape.y * carrier.shape.z
    val = template_ft[k * plane + in_plane].to_register()
    term = (weights[m * plane + in_plane] * (val.real * val.real + val.imag * val.imag)).to_register()
    with vc.if_block(vc.all(cz > 0, cz < carrier.shape.z - last_column_single)):
        term[:] = term * 2.0
    return term

@vd.reduce.reduce(0, axes=[1, 2], mapping_function=whitened_norm_map)
def calc_whitened_norms(a: Const[f32], b: Const[f32]) -> f32:
    return a + b

def next_fft_size(n: int, max_prime: int = 13) -> int:
    """Smallest even integer >= n whose prime factors are all <= max_prime (even: tm2d's rfft layout assumes an
    even last dimension)."""
    m = max(int(n), 2)
    m += m % 2
    while True:
        k = m
        for p in (2, 3, 5, 7, 11, 13):
            if p > max_prime:
                break
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 2

def lag_window_weights(weights, shape, template_shape, grid):
    """The whitening weights moved onto the small lag grid `grid` (rfft2 half-plane), for exact template norms.

    With a = irfft2(w) (the noise-whitening kernel in real space), ||t||_w^2 = sum_s w(s) |T(s)|^2 / N
    = sum_{x, x'} t(x) t(x') a(x - x'). For t inside an (h, w) box only lags |d| < (h, w) matter, so a cropped to
    those lags and transformed on any grid G >= 2 * box - 1 gives ||t||_w^2 = sum_{k in G} A_k |T_k|^2 / |G| exactly.
    With grid == shape, A = w."""
    H, W = shape
    h, w = template_shape
    Gh, Gw = grid
    a = np.fft.irfft2(weights, s=(H, W))
    if (Gh, Gw) == (H, W):
        return np.fft.rfft2(a).real
    # a dimension where the grid is the image itself keeps every (circular) lag; otherwise G >= 2 * box - 1 and the
    # lags |d| < box are copied
    if Gh == H:
        iy = oy = np.arange(H)
    else:
        ly = min(h, (Gh + 1) // 2)
        iy, oy = np.r_[0:ly, H - ly + 1:H], np.r_[0:ly, Gh - ly + 1:Gh]
    if Gw == W:
        ix = ox = np.arange(W)
    else:
        lx = min(w, (Gw + 1) // 2)
        ix, ox = np.r_[0:lx, W - lx + 1:W], np.r_[0:lx, Gw - lx + 1:Gw]
    ac = np.zeros((Gh, Gw))
    ac[np.ix_(oy, ox)] = a[np.ix_(iy, ix)]
    return np.fft.rfft2(ac).real

def comparator_radial_table(weights: RadialWeights, shape) -> RadialWeights:
    """The RadialWeights the fused comparator actually uses for an (M, H, W) or (H, W) shape: `weights` resampled
    (linear in s, last value held) onto the comparator's fixed table of 2 samples per Fourier-pixel spacing out to the
    corner of Fourier space, in float32. tm2d.prefilter uses it so that a prefiltered image matches the full search's w."""
    H, W = (int(n) for n in tuple(shape)[-2:])
    M = int(shape[0]) if len(tuple(shape)) == 3 else 1
    table_size = int(2 * (max(H, W) // 2 + 1) * np.sqrt(2)) + 2
    table_ds = (np.sqrt(2) / 2) / (table_size - 1)
    tables = weights.tables(M) if weights.values.ndim == 2 else weights.tables(1)
    s = np.arange(table_size) * table_ds
    j = np.minimum(s / weights.ds, tables.shape[1] - 1)
    j0 = np.minimum(np.floor(j).astype(np.int64), tables.shape[1] - 2)
    f = j - j0
    resampled = (tables[:, j0] * (1 - f) + tables[:, j0 + 1] * f).astype(np.float32)
    return RadialWeights(resampled.astype(np.float64) if weights.values.ndim == 2 else resampled[0].astype(np.float64),
                         table_ds)


def make_norm_reduction():
    """Row sums of the (micrographs * templates, columns * threads) partials of the fused norm (= N ||t||_w^2).
    Built per comparator (a module-level reduction was not safe to share between plans; entry 8). The results
    kernels turn the sums into 1 / ||t||_w themselves, saving a dispatch per graph instance."""
    @vd.reduce.map_reduce(vd.reduce.SubgroupAdd, axes=[1])
    def sum_partials(partials: Buff[f32]) -> f32:
        return partials[vd.reduce.mapped_io_index()]

    return sum_partials

class ComparatorCrossCorrelation(Comparator):
    """Matched-filter score of every template against every micrograph, for noise of power spectrum P(s):

        X(u) = sum_s conj(T(s)) Y(s) w(s) e^{i 2 pi s.u} / N  /  ||t||_w,    ||t||_w^2 = sum_s w(s) |T(s)|^2 / N,

    with sums over the full N = H * W grid of numpy's unnormalized fft2, T the edge-mean-subtracted template padded
    to (H, W), and w = 1/P or w = 1 (no weights). w is 0 at DC and on the Nyquist lines. This equals cisTEM /
    Leopard-EM's correlation of the whitened image with the whitened, CTF-filtered template, normalized by the
    whitened template. Each template is normalized by its own exact norm, so under the noise model X has unit
    variance at every pixel, for every template.

    norm_method:
      "fused" (default): w is radial (tm2d.RadialWeights, e.g. tm2d_utils.whitening_radial_weights) and
          ||t||_w is summed inside the fused column pass on the full grid, at no extra FFT; the 1 / ||t||_w factor
          is applied by the results stage (self.score_norms, 1 / sqrt(score_norms * score_norm_inv_n)).
      "lag_grid": any 2-D w (array on the rfft2 half-plane); ||t||_w from an extra FFT on a small lag grid
          (lag_window_weights), applied before the inverse FFT."""

    def __init__(self, shape: tuple, template_shape: tuple, norm_method: str = None, prefiltered: bool = False):
        assert len(shape) == 3, "Shape must be a 3D tuple (N, H, W) but got: {}".format(shape)
        if norm_method is None:
            norm_method = "lag_grid" if prefiltered else "fused"
        if norm_method not in ("fused", "lag_grid"):
            raise ValueError(f"norm_method must be 'fused' or 'lag_grid', not {norm_method!r}")
        if prefiltered and norm_method != "lag_grid":
            raise ValueError("prefiltered data needs norm_method='lag_grid' (the norm weights come from the full image)")
        self.norm_method = norm_method
        self.prefiltered = bool(prefiltered)
        if int(shape[2]) % 2:
            # the rfft layouts here (and the results kernels, which recover W as 2 * (W//2 + 1) - 2) assume an even
            # width; odd widths used to give wrong scores silently
            raise ValueError(f"micrograph width {shape[2]} is odd: crop or pad it to an even width")

        check_buffer_bytes((shape[0], shape[1], shape[2] // 2 + 1), 8, "micrograph spectrum buffer",
                           "Use fewer micrographs per plan.")
        self.micrographs_buffer = vd.RFFTBuffer(shape)
        self.template_shape = (int(template_shape[0]), int(template_shape[1]))
        self.weights = None
        self.score_norms = None
        self.score_norm_inv_n = 1.0
        M, H, W = shape
        if norm_method == "lag_grid":
            # lag grid for the template norms: the smallest FFT size (primes <= 13, vkdispatch's limit) >= 2 * box - 1,
            # or the micrograph if that is smaller (exact for any grid >= 2 * box - 1, see lag_window_weights)
            # (prefiltered: always >= 2 * box - 1, since the noise kernel comes from the full image, not this grid)
            self.norm_grid = (next_fft_size(2 * self.template_shape[0] - 1),
                              next_fft_size(2 * self.template_shape[1] - 1))
            if not prefiltered:
                self.norm_grid = (min(self.norm_grid[0], shape[1]), min(self.norm_grid[1], shape[2]))
            self.lag_weights_buffer = vd.Buffer((shape[0], self.norm_grid[0], self.norm_grid[1] // 2 + 1), vd.float32)
            if not prefiltered:
                self._write_weights(None)
        else:
            # fixed radial table on the GPU (the graph is recorded once): 0 .. corner of Fourier space, 2 samples per
            # Fourier-pixel spacing; incoming RadialWeights are resampled onto it, and the CPU uses the same table
            self.table_size = int(2 * (max(H, W) // 2 + 1) * np.sqrt(2)) + 2
            self.table_ds = (np.sqrt(2) / 2) / (self.table_size - 1)
            self.weight_tables_buffer = vd.Buffer((M, self.table_size), vd.float32)
            self._write_radial(None)

    def _write_weights(self, weights):
        M, H, W = self.micrographs_buffer.real_shape
        if weights is None:
            weights = np.ones((H, W // 2 + 1))
        if isinstance(weights, RadialWeights):
            weights = weights.evaluate((H, W), M)
        weights = np.asarray(weights, dtype=np.float64)
        assert weights.shape in ((H, W // 2 + 1), (M, H, W // 2 + 1)), \
            f"weights must have shape ({H}, {W // 2 + 1}) or ({M}, {H}, {W // 2 + 1}), got {weights.shape}"
        assert np.all(np.isfinite(weights)) and np.all(weights >= 0), "weights must be finite and non-negative"
        w = np.broadcast_to(weights, (M, H, W // 2 + 1)).copy()
        w[:, edge_mask((H, W))] = 0.0
        self.weights = w
        A = np.stack([lag_window_weights(wm, (H, W), self.template_shape, self.norm_grid) for wm in w])
        self.lag_weights_buffer.write(A.astype(np.float32))

    def _write_radial(self, weights):
        M, H, W = self.micrographs_buffer.real_shape
        if weights is None:
            weights = RadialWeights.ones()
        if not isinstance(weights, RadialWeights):
            raise ValueError("norm_method='fused' takes tm2d.RadialWeights (e.g. tm2d_utils.whitening_radial_weights) "
                             "or None; use ComparatorCrossCorrelation(..., norm_method='lag_grid') for 2-D weights")
        tables = RadialWeights(weights.tables(M), weights.ds)                  # one table per micrograph
        self.radial = comparator_radial_table(tables, (M, H, W))
        self.weights = self.radial.evaluate((H, W), M)
        self.weight_tables_buffer.write(self.radial.tables(M).astype(np.float32))

    def norm_lag_weights(self, noise_weights: np.ndarray, full_shape: tuple) -> np.ndarray:
        """For prefiltered data: the lag-grid norm weights from the filtered image's noise weights on the grid it was
        filtered on. noise_weights: rfft2 half-plane array (Hf, Wf//2+1) or (n, Hf, Wf//2+1) on the (Hf, Wf) grid
        of the full image, = |f|^2 P for an image filtered by f; for f = w conj(C) and w = 1/P that is w |C|^2
        (tm2d.prefilter.prefilter_micrograph returns it). The template norm is then exact for any crop of that
        image: only lags smaller than the template box matter (lag_window_weights). Compute once per filter and pass
        to set_data (weights=); returns (n, Gh, Gw//2+1)."""
        nw = np.asarray(noise_weights, dtype=np.float64)
        nw = nw[None] if nw.ndim == 2 else nw
        Hf, Wf = (int(n) for n in full_shape)
        assert nw.shape[1:] == (Hf, Wf // 2 + 1), f"noise weights {nw.shape[1:]} do not match the full shape {(Hf, Wf)}"
        nw = nw.copy()
        nw[:, edge_mask((Hf, Wf))] = 0.0
        return np.stack([lag_window_weights(a, (Hf, Wf), self.template_shape, self.norm_grid) for a in nw])

    def set_data(self, data: np.ndarray, weights=None):
        """data: (N, H, W) or (H, W) micrographs.
        weights: None (w = 1), tm2d.RadialWeights (w = 1/P; one table or one per micrograph), or, with
        norm_method='lag_grid' only, a 2-D array w on the rfft2 half-plane (unshifted), (H, W//2+1) or (N, H, W//2+1)."""
        if data.ndim == 2:
            data = data.reshape((1, *data.shape))

        assert data.ndim == 3, "Micrographs must be a 3D array (N, H, W)."
        assert data.shape == tuple(self.micrographs_buffer.real_shape), \
            f"data shape {data.shape} != comparator shape {tuple(self.micrographs_buffer.real_shape)}"
        if self.prefiltered:
            # the data are already whitened and CTF-filtered (crops of a filtered image): no weighting and no DC
            # removal here (a crop's local mean is real signal of the filtered image); weights = the lag-grid norm
            # weights from norm_lag_weights, one per micrograph or one for all
            M_, H_, W_ = data.shape
            A = np.asarray(weights, dtype=np.float64) if weights is not None else None
            Gh, Gw = self.norm_grid
            if A is None or A.shape[-2:] != (Gh, Gw // 2 + 1):
                raise ValueError(f"prefiltered data need weights = comparator.norm_lag_weights(...) of shape "
                                 f"(M, {Gh}, {Gw // 2 + 1}) or ({Gh}, {Gw // 2 + 1})")
            self.weights = None
            self.lag_weights_buffer.write(np.broadcast_to(A, (M_, Gh, Gw // 2 + 1)).astype(np.float32))
            D = np.fft.rfft2(np.asarray(data, dtype=np.float64))
            self.micrographs_buffer.write_fourier(np.ascontiguousarray(D.transpose(0, 2, 1)).reshape(M_, H_, W_ // 2 + 1).astype(np.complex64))
            return
        if self.norm_method == "lag_grid":
            self._write_weights(weights)
        else:
            self._write_radial(weights)
        D = np.fft.rfft2(np.asarray(data, dtype=np.float64))
        D *= self.weights   # also zeroes DC and the Nyquist lines
        # stored transposed, (C, H) per micrograph (zipFFT's layout): the column pass then reads it contiguously.
        # Same buffer and size; only the order of elements changes.
        M_, H_, C_ = D.shape
        self.micrographs_buffer.write_fourier(np.ascontiguousarray(D.transpose(0, 2, 1)).reshape(M_, H_, C_).astype(np.complex64))

    def valid_trim_radius(self):
        """Smallest r such that rows y < r or y >= H - r and columns x < r or x >= W - r (raw correlation
        coordinates) hold every position where the template lies fully inside the micrograph; None if that is the
        whole map. Computing only those rows / columns (the output_radius pipeline) skips work the results drop."""
        from ..base_results import valid_center_range
        H, W = int(self.micrographs_buffer.real_shape[1]), int(self.micrographs_buffer.real_shape[2])
        need = 0
        for n, m in ((H, self.template_shape[0]), (W, self.template_shape[1])):
            lo, hi = valid_center_range(n, m)
            raw = (np.arange(lo, hi + 1) + (n + 1) // 2) % n     # results maps are fftshifted raw maps: raw = i - n // 2
            need = max(need, int(np.max(np.where(raw < n / 2, raw + 1, n - raw))))
        if 2 * need >= min(H, W):
            return None
        return need

    def compare_template(self, template_buffer: vd.RFFTBuffer, normalize: bool = True, output_radius: int = None,
                         trim_to_valid: bool = False, image_ctf=None) -> vd.RFFTBuffer:
        """trim_to_valid (Plan sets it when the results exclude edge positions and no output_radius is given):
        compute only the rows / columns that hold valid positions (valid_trim_radius); the others are left stale.

        image_ctf (Plan sets it with ctf_side="image"): an object with .ctf_params (CTFParams), .count (D, CTFs per
        batch), .pixel_size (float or graph variable) and .args (ctf_params.get_args(cmd_graph, D)). The templates are
        then CTF-free projections, and each of the D CTFs C_d is applied on the micrograph's own Fourier grid:

            X_d(u) = sum_s conj(T(s)) conj(C_d(s)) Y(s) w(s) e^{i 2 pi s.u} / N / ||c_d * t||_w,
            ||c_d * t||_w^2 = sum_s w(s) |C_d(s)|^2 |T(s)|^2 / N,

        i.e. the template-side score with the CTF evaluated on the (H, W) grid instead of the template box: no box
        truncation of the CTF's delocalized fringes and no CTF aliasing on the box grid. The template's forward FFT is
        shared by the D CTFs (they are extra kernels of the fused column pass: M * D kernels). Output maps are ordered
        (micrograph m, CTF d, template k), so per micrograph the D * K maps are slot d * K + k. Fused norms only."""
        assert normalize, "Normalization is mandatory for cross-correlation."
        if output_radius is None and trim_to_valid:
            output_radius = self.valid_trim_radius()
        D = 1 if image_ctf is None else int(image_ctf.count)
        if image_ctf is not None and self.norm_method != "fused":
            raise NotImplementedError("the image-side CTF needs norm_method='fused' (the per-CTF norm is summed in the "
                                      "fused column pass)")

        correlation_shape = (
            self.micrographs_buffer.real_shape[0] * D * template_buffer.shape[0],
            self.micrographs_buffer.real_shape[1],
            self.micrographs_buffer.real_shape[2]
        )

        check_buffer_bytes((correlation_shape[0], correlation_shape[1], correlation_shape[2] // 2 + 1), 8,
                           "correlation buffer (micrographs x templates per batch, H, W//2+1 complex)",
                           "Use a smaller template_batch_size or fewer micrographs per plan.")
        correlation_signal_out = vd.RFFTBuffer(correlation_shape)

        template_sum = calc_sums(template_buffer)

        # row r2c of the padded template over the template's own h rows only (zipFFT's first kernel), into a small
        # (K, h, W) band buffer: the other H - h rows are zero, and the column pass (conv_input_map) synthesizes
        # them. Band row b = template row b = micrograph row b + H//2 - h//2 (where crop_mapping centres the template).
        Hm, Wm = int(self.micrographs_buffer.real_shape[1]), int(self.micrographs_buffer.real_shape[2])
        Cm = Wm // 2 + 1
        hb = int(template_buffer.real_shape[1])
        row0 = Hm // 2 - hb // 2
        band_signal = vd.RFFTBuffer((template_buffer.shape[0], hb, Wm))
        vd.fft.fft(
            band_signal,
            template_buffer,
            template_sum,
            (hb, Wm),
            int(template_buffer.real_shape[2]),
            buffer_shape=(template_buffer.shape[0], hb, Wm),
            input_map=crop_mapping,
            r2c=True
        )

        kernel_offset = int(
            self.micrographs_buffer.shape[1] * self.micrographs_buffer.shape[2]
        )
        K = template_buffer.shape[0]
        template_count = int(K)
        M = int(self.micrographs_buffer.real_shape[0])
        H = int(self.micrographs_buffer.real_shape[1])
        W = int(self.micrographs_buffer.real_shape[2])
        C = W // 2 + 1

        @vd.map
        def conv_input_map(input_buffer: vc.Buffer[c64]):
            # input_buffer = the (K, h, C) band; rows outside the band are the zero padding
            op = vd.fft.read_op()
            row = op.fft_index.to_dtype(vc.i32).to_register()
            with vc.if_block(vc.all(row >= row0, row < row0 + hb)):
                io = op.io_index.to_dtype(vc.i32).to_register()
                kk = (io // (Hm * Cm)).to_register()
                col_ = (io % Cm).to_register()
                op.register[:] = input_buffer[(kk * hb + row - row0) * Cm + col_]
            with vc.else_block():
                op.register.real = 0.0
                op.register.imag = 0.0

        @vd.map
        def output_map_func(output_buffer: vc.Buffer[c64]):
            if output_radius is None:
                out_reg = vd.fft.write_op().register
                output_buffer[vd.fft.write_op().io_index + kernel_offset * template_buffer.shape[0] * vd.fft.mapped_kernel_index()] = out_reg
                return

            with vc.if_block(vc.any(
                vd.fft.write_op().fft_index < output_radius,
                vd.fft.write_op().fft_index >= self.micrographs_buffer.shape[1] - output_radius)):

                out_reg = vd.fft.write_op().register
                output_buffer[vd.fft.write_op().io_index + kernel_offset * template_buffer.shape[0] * vd.fft.mapped_kernel_index()] = out_reg

        if self.norm_method == "lag_grid":
            # exact (whitened) norm of each template against each micrograph's weights, on the lag grid
            Gh, Gw = self.norm_grid
            template_ft = vd.RFFTBuffer((K, Gh, Gw))
            vd.fft.fft(template_ft, template_buffer, template_sum, (Gh, Gw), int(template_buffer.real_shape[2]), buffer_shape=(K, Gh, Gw),
                       input_map=crop_mapping, r2c=True)
            vd.fft.fft(template_ft, buffer_shape=template_ft.shape, axis=1)
            norm_carrier = vd.Buffer((M * K, Gh, Gw // 2 + 1), vd.complex64)
            template_norms = calc_whitened_norms(norm_carrier, template_ft, self.lag_weights_buffer, K,
                                                 1 if Gw % 2 == 0 else 0)
            inv_grid = 1.0 / float(Gh * Gw)

            @vd.map
            def convolution_map(kernel_buffer: vc.Buffer[c64], norms_buffer: vc.Buffer[f32]):
                img_val = vd.fft.read_op().register
                read_register = vc.new_complex64_register()

                in_plane_l = (vd.fft.read_op().io_index % kernel_offset).to_dtype(vc.i32).to_register()
                read_register[:] = kernel_buffer[(in_plane_l % C) * H + in_plane_l // C + kernel_offset * vd.fft.mapped_kernel_index()]
                # 1 / ||t||_w for this (micrograph, template), folded into the product (no extra pass over the maps)
                k = (vd.fft.read_op().io_index // kernel_offset).to_dtype(vc.i32).to_register()
                mk = (k + vd.fft.mapped_kernel_index() * template_count).to_register()
                scale = (1.0 / vc.sqrt(norms_buffer[mk] * inv_grid)).to_register()
                img_val[:] = vc.mult_complex(read_register, img_val.conjugate()) * scale

            vd.fft.convolve(
                correlation_signal_out,
                band_signal,
                self.micrographs_buffer,
                template_norms,
                kernel_num=M,
                axis=1,
                buffer_shape=(K, H, C),
                input_map=conv_input_map,
                kernel_map=convolution_map,
                output_map=output_map_func
            )
            self.score_norms = None
        else:
            # fused: sum_s w |T|^2 accumulated in the column pass (fused_conv), w from the radial table
            state = FusedNormState()
            n_tab = int(self.table_size)
            inv_ds = float(1.0 / self.table_ds)
            nyq_col = W // 2 if W % 2 == 0 else -1
            nyq_row = H // 2 if H % 2 == 0 else -1

            def fused_body(kernel_buffer, tables, partials, *ctf_rest):
                state.partials = partials
                op = vd.fft.read_op()
                T = op.register
                kidx = vd.fft.mapped_kernel_index()      # python int: the kernel loop is unrolled
                m, d = kidx // D, kidx % D               # kernel = (micrograph m, image-side CTF d)
                in_plane = (op.io_index % kernel_offset).to_dtype(vc.i32).to_register()
                row = (in_plane // C).to_register()
                col = (in_plane % C).to_register()
                ky = vc.new_float_register(0.0)
                ky[:] = row.to_dtype(f32)
                with vc.if_block(row > H // 2):
                    ky[:] = (row - H).to_dtype(f32)
                sy = (ky / float(H)).to_register()
                sx = (col.to_dtype(f32) / float(W)).to_register()
                j = vc.min(vc.sqrt(sx * sx + sy * sy) * inv_ds, float(n_tab - 1)).to_register()
                j0 = vc.floor(j).to_dtype(vc.i32).to_register()
                with vc.if_block(j0 > n_tab - 2):
                    j0[:] = n_tab - 2
                frac = (j - j0.to_dtype(f32)).to_register()
                base = m * n_tab
                w = (tables[base + j0] * (1.0 - frac) + tables[base + j0 + 1] * frac).to_register()
                with vc.if_block(vc.any(vc.all(row == 0, col == 0), row == nyq_row, col == nyq_col)):
                    w[:] = 0.0
                with vc.if_block(vc.all(col > 0, col != nyq_col)):
                    w[:] = w * 2.0          # the rfft half-plane counts twice except column 0 and Nyquist

                img_val = kernel_buffer[col * H + row + kernel_offset * m].to_register()   # transposed image
                if ctf_rest:
                    # image-side CTF d on the micrograph grid: norm weight w |C_d|^2, image times conj(C_d). Within a
                    # pass, kernels of the same CTF (other micrographs) reuse the value computed by the first one.
                    if state.cache is None or kidx in state.compute_kernels:
                        pix = ctf_rest[0]
                        params_d = image_ctf.ctf_params.assemble_params_list_from_args(list(ctf_rest[1:]), D)[d]
                        ctf = image_ctf_value(H, W, ky, col, pix, params_d, getattr(image_ctf, "static_params", None))
                        if state.cache is not None:
                            state.cache[op.register_id][:] = ctf
                    else:
                        ctf = state.cache[op.register_id]
                    w[:] = w * (ctf.real * ctf.real + ctf.imag * ctf.imag)
                    img_val[:] = vc.mult_complex(img_val, ctf.conjugate())
                state.record(op.io_index, w * (T.real * T.real + T.imag * T.imag))
                T[:] = vc.mult_complex(img_val, T.conjugate())

            map_types = [vc.Buffer[c64], vc.Buffer[f32], vc.Buffer[f32]]
            ctf_call_args = []
            if image_ctf is not None:
                from ..templates.atomic import get_pixel_size_type
                map_types += [get_pixel_size_type(image_ctf.pixel_size)] + image_ctf.ctf_params.get_type_list(D)
                ctf_call_args = [image_ctf.pixel_size] + list(image_ctf.args)
            fused_map = vd.map(func=fused_body, input_types=map_types)

            # kernel passes: the M * D kernels (kernel m * D + d) in CTF-major order, at most kernels_per_pass per
            # shader. A long unrolled kernel loop (e.g. 8 micrographs x 4 CTFs = 32) made the 4096^2 column pass ~10%
            # slower than the template side's 8 (log entry 26); each pass redoes the forward column FFT instead.
            if image_ctf is None:
                passes = [list(range(M))]
            else:
                order = [m * D + d for d in range(D) for m in range(M)]
                P = max(1, int(getattr(image_ctf, "kernels_per_pass", None) or len(order)))
                passes = [order[i:i + P] for i in range(0, len(order), P)]
            partials = None
            for kernels in passes:
                state.compute_kernels = {k for i, k in enumerate(kernels) if i == 0 or k % D != kernels[i - 1] % D}
                reuse = image_ctf is not None and len(state.compute_kernels) < len(kernels)
                shader, partials_per_row = make_fused_norm_convolution_shader(
                    (K, H, C), fused_map, M * D, 1, conv_input_map, output_map_func, state, kernel_offset,
                    template_count, C, kernel_indices=kernels, cache_registers=reuse)
                if partials is None:
                    partials = vd.Buffer((M * D * K, partials_per_row), vd.float32)
                shader(correlation_signal_out, band_signal, self.micrographs_buffer, self.weight_tables_buffer,
                       partials, *ctf_call_args)
            sum_partials = make_norm_reduction()
            self.score_norms = sum_partials(partials)
            self.score_norm_inv_n = 1.0 / float(H * W)

        if output_radius is None:
            vd.fft.irfft(correlation_signal_out)
            return correlation_signal_out

        # output buffer of the trimmed inverse FFT (only this path needs a second full-size buffer)
        correlation_signal = vd.RFFTBuffer(correlation_shape)

        trimmed_shape = (
            correlation_signal.real_shape[0],
            output_radius * 2,
            correlation_signal.real_shape[2]
        )

        trimmed_shape_2 = (
            correlation_signal.shape[0],
            output_radius * 2,
            correlation_signal.shape[2] * 2
        )

        trimmed_shape_3 = (
            correlation_signal.shape[0],
            output_radius * 2,
            correlation_signal.shape[2]
        )

        corr_shape_2 = (
            correlation_signal.shape[0],
            correlation_signal.shape[1],
            correlation_signal.shape[2] * 2,
        )

        @vd.map
        def irfft_map_input(input_buffer: vc.Buffer[c64]):
            io_index = vd.fft.read_op().io_index

            with vc.if_block(vd.fft.read_op().fft_index >= (vd.fft.read_op().fft_size // 2) + 1):
                io_index[:] = vd.fft.read_op().r2c_inverse_offset - io_index

            coord3d = vc.ravel_index(vd.fft.read_op().io_index, trimmed_shape_3).to_register()

            with vc.if_block(coord3d.y >= output_radius):
              coord3d.y += correlation_signal.shape[1] - output_radius * 2

            new_io_index = vc.unravel_index(coord3d, correlation_signal.shape).to_register()

            vd.fft.read_op().register[:] = input_buffer[new_io_index]

            with vc.if_block(vd.fft.read_op().fft_index >= (vd.fft.read_op().fft_size // 2) + 1):
                vd.fft.read_op().register.imag = -vd.fft.read_op().register.imag

        @vd.map
        def irfft_map_output(output_buffer: vc.Buffer[c64]):
            coord3d = vc.ravel_index(vd.fft.write_op().io_index, trimmed_shape_2).to_register()

            with vc.if_block(vc.any(
                coord3d.z < output_radius,
                coord3d.z >= self.micrographs_buffer.real_shape[2] - output_radius)):

                with vc.if_block(coord3d.y >= output_radius):
                    coord3d.y += correlation_signal.shape[1] - output_radius * 2

                new_io_index = vc.unravel_index(coord3d, corr_shape_2).to_register()

                vd.fft.write_op().write_to_buffer(output_buffer, io_index=new_io_index)

        vd.fft.fft(
            correlation_signal,
            correlation_signal_out,
            buffer_shape=trimmed_shape,
            input_map=irfft_map_input,
            output_map=irfft_map_output,
            r2c=True,
            inverse=True
        )

        return correlation_signal