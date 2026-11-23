import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from ..plan import Results
from ..base_results import valid_center_range
from ..limits import check_buffer_bytes

# To avoid errors from the limited precision of floating point numbers, we use 2 32-bit floats
# to approximate a higher precision accumulator.
# Algorithm modified from this paper: https://link.springer.com/article/10.1007/pl00009321
def double_precision_add_f32(dsa: vc.Const[vc.v2], dsb: vc.Const[vc.f32]) -> vc.v2:
    t2 = (dsa.y + dsb).to_register()
    dsc_x = (dsa.x + t2).to_register()

    return vc.new_vec2_register(dsc_x, t2 - (dsc_x - dsa.x))


def double_precision_add_vec2(dsa: vc.Const[vc.v2], dsb: vc.Const[vc.v2]) -> vc.v2:
    t1 = (dsa.x + dsb.x).to_register()
    e = (t1 - dsa.x).to_register()
    t2 = (((dsb.x - e) + (dsa.x - (t1 - e))) + dsa.y + dsb.y).to_register()

    dsc_x = (t1 + t2).to_register()

    return vc.new_vec2_register(dsc_x, t2 - (dsc_x - t1))


def accumulate(
        template_count: int,
        cross_correlation: vc.Buff[vc.f32],
        cross_corr_index: vc.Var[vc.i32],
        micrograph_span_rfft: int,
        rotation_weights: list,
        scales: list,
        *index_values: vc.Var[vc.i32]):
    """rotation_weights: one weight per slot (constant 1 or graph variables). scales: one factor per slot
    (shader expressions) or None."""

    mip = vc.new_float_register(0, var_name="mip")

    sum_cross = vc.new_vec2_register(0, var_name="sum_cross")
    sum2_cross = vc.new_vec2_register(0, var_name="sum2_cross")

    best_mip = vc.new_float_register(vc.ninf_f32(), var_name="best_mip")
    best_index = vc.new_int_register(-1, var_name="best_index")

    for i in range(template_count):
        if i != 0:
            cross_corr_index[:] = cross_corr_index + micrograph_span_rfft

        with vc.if_block(index_values[i] >= 0):
            mip[:] = cross_correlation[cross_corr_index]
            if scales is not None:
                mip[:] = mip * scales[i]

            sum_cross[:] = double_precision_add_f32(sum_cross, mip * rotation_weights[i])
            sum2_cross[:] = double_precision_add_f32(sum2_cross, mip * mip * rotation_weights[i])

            with vc.if_block(mip > best_mip):
                best_mip[:] = mip
                best_index[:] = index_values[i]

    return best_mip, best_index, sum_cross, sum2_cross


@vd.shader("out.size")
def collapse_sums(out: vc.Buff[vc.v2], sum_cross: vc.Buff[vc.v2], sum2_cross: vc.Buff[vc.v2]):
    """out = (sum hi + lo, sum2 hi + lo) per pixel, on every queue's copy."""
    ind = vc.global_invocation_id().x
    a = sum_cross[ind].to_register()
    b = sum2_cross[ind].to_register()
    out[ind] = vc.new_vec2_register(a.x + a.y, b.x + b.y)


class ResultsPixel(Results):
    max_cross: vd.Buffer
    best_index: vd.Buffer
    sum_cross: vd.Buffer # running mean
    sum2_cross: vd.Buffer # running variance

    output_radius: int

    micrograph_count: int

    compiled: bool

    compiled_mip: np.ndarray
    compiled_best_index_array: np.ndarray
    compiled_sum_cross: np.ndarray
    compiled_sum2_cross: np.ndarray

    templates_count: int

    def __init__(self, shape: tuple, output_radius: int = None, exclude_edges="auto") -> None:
        """exclude_edges: True = report only positions where the template lies fully inside the micrograph
        (Leopard-EM's valid mode; elsewhere the circular correlation wraps the template onto the opposite edge);
        False = every position (circular correlation, as v0.3.0); "auto" (default) = True for micrograph searches,
        False for particle boxes (template as large as the image, or output_radius set); see
        Results.edges_excluded. Excluded positions are NaN in every map (best index -1), with the map shapes
        unchanged."""
        self.exclude_edges_setting = self._check_edge_setting(exclude_edges)
        self.micrograph_shape = (int(shape[1]), int(shape[2]))
        self.valid_mask = None
        assert len(shape) == 3, "Shape must be a 3D tuple (L, H, W)."

        micrograph_count = int(shape[0])
        width = int(shape[1]) if output_radius is None else int(output_radius * 2)
        height = int(shape[2]) if output_radius is None else int(output_radius * 2)

        self.output_radius = output_radius

        self.micrograph_count = micrograph_count

        check_buffer_bytes((micrograph_count, width, height), 8, "per-pixel sum buffer", "Use fewer micrographs per plan.")
        self.max_cross = vd.Buffer((micrograph_count, width, height), vd.float32)
        self.best_index = vd.Buffer((micrograph_count, width, height), vd.int32)
        self.sum_cross = vd.Buffer((micrograph_count, width, height), vd.vec2)
        self.sum2_cross = vd.Buffer((micrograph_count, width, height), vd.vec2)

        self.reset()

    def reset(self):
        self.max_cross.write((np.ones(shape=self.max_cross.shape) * -1000000).astype(np.float32))
        self.best_index.write((np.ones(shape=self.best_index.shape, dtype=np.int32) * -1).astype(np.int32))
        self.sum_cross.write(np.zeros(shape=(*self.sum_cross.shape, 2), dtype=np.float32))
        self.sum2_cross.write(np.zeros(shape=(*self.sum2_cross.shape, 2), dtype=np.float32))

        self.compiled = False
        self.compiled_mip = None
        self.compiled_best_index_array = None
        self.compiled_sum_cross = None
        self.compiled_sum2_cross = None
        self.templates_count = None

    def compile_results(self, templates_count: int, total_weight: float = None):
        """total_weight: sum of the rotation weights over all searched parameters (Plan.run passes it when
        enable_rotation_weights is on). The per-pixel sums are weighted by the rotation weights, so their mean and
        variance divide by this, not by templates_count. None: unweighted (= templates_count)."""
        # one copy of each buffer per queue: merged queue by queue in float32 / int32 (max, first queue wins ties) and
        # float64 (sums of the hi + lo float pairs), then fftshifted once
        queues = self.max_cross.context.queue_count
        mip = best = None
        for q in range(queues):
            m, b = self.max_cross.read(q), self.best_index.read(q)
            if mip is None:
                mip, best = np.array(m, copy=True), np.array(b, copy=True)   # reads may be read-only views
            else:
                better = m > mip
                np.copyto(mip, m, where=better)
                np.copyto(best, b, where=better)
        # the double-float (hi, lo) accumulators are collapsed to one float each on the GPU before readback (float32
        # keeps ~1e-7 relative precision of the final sums; the pairs matter only while accumulating): 16 instead of
        # 24 bytes per pixel and queue cross the slow readback
        if getattr(self, "_collapsed", None) is None:
            self._collapsed = vd.Buffer(self.sum_cross.shape, vd.vec2)
        collapse_sums(self._collapsed, self.sum_cross, self.sum2_cross)
        sums = [np.zeros(self.sum_cross.shape, dtype=np.float64) for _ in range(2)]
        for q in range(queues):
            pair = self._collapsed.read(q)
            sums[0] += pair[..., 0]
            sums[1] += pair[..., 1]

        self.compiled_mip = np.fft.fftshift(mip, axes=(1, 2)).astype(np.float64)
        self.compiled_best_index_array = np.fft.fftshift(best, axes=(1, 2)).astype(np.int32, copy=False)
        self.compiled_sum_cross = np.fft.fftshift(sums[0], axes=(1, 2))
        self.compiled_sum2_cross = np.fft.fftshift(sums[1], axes=(1, 2))

        self.valid_mask = self._valid_mask()
        rows_bad = ~self.valid_mask.any(axis=1)     # the valid region is a rectangle: blank whole rows / columns
        cols_bad = ~self.valid_mask.any(axis=0)
        for arr, fill in ((self.compiled_mip, np.nan), (self.compiled_best_index_array, -1),
                          (self.compiled_sum_cross, np.nan), (self.compiled_sum2_cross, np.nan)):
            arr[:, rows_bad, :] = fill
            arr[:, :, cols_bad] = fill

        self.templates_count = templates_count
        self.total_weight = float(templates_count if total_weight is None else total_weight)

        self.compiled = True

    def get_mip(self):
        assert self.compiled, "Results must be compiled before accessing the MIP."
        return self.compiled_mip

    def get_best_index_array(self):
        assert self.compiled, "Results must be compiled before accessing the best index array."
        return self.compiled_best_index_array

    def get_sum_cross(self):
        assert self.compiled, "Results must be compiled before accessing the sum of cross-correlations."
        return self.compiled_sum_cross

    def get_sum2_cross(self):
        assert self.compiled, "Results must be compiled before accessing the sum of squared cross-correlations."
        return self.compiled_sum2_cross

    def get_templates_count(self):
        assert self.compiled, "Results must be compiled before accessing the templates count."
        return self.templates_count

    def get_total_weight(self):
        """Sum of the weights in the per-pixel sums: the total rotation weight, or the template count if unweighted."""
        assert self.compiled, "Results must be compiled before accessing the total weight."
        return self.total_weight

    def get_mean(self):
        """Per-pixel (weighted) mean of the score over all searched parameters."""
        assert self.compiled, "Results must be compiled first."
        return self.compiled_sum_cross / self.total_weight

    def get_std(self):
        """Per-pixel (weighted) standard deviation of the score over all searched parameters."""
        mean = self.get_mean()
        return np.sqrt(np.maximum(self.compiled_sum2_cross / self.total_weight - mean * mean, 0.0))

    def _valid_mask(self) -> np.ndarray:
        """(rows, cols) bool mask of the output map: True where the template lies fully inside the micrograph."""
        rows, cols = self.compiled_mip.shape[1:]
        if not self.edges_excluded(self.micrograph_shape):
            return np.ones((rows, cols), bool)
        H, W = self.micrograph_shape
        h, w = self.template_shape
        if self.output_radius is None:
            y, x = np.arange(rows), np.arange(cols)
        else:   # the 2r x 2r output box is centred on map index (H // 2, W // 2)
            r = self.output_radius
            y, x = np.arange(rows) + H // 2 - r, np.arange(cols) + W // 2 - r
        ylo, yhi = valid_center_range(H, h)
        xlo, xhi = valid_center_range(W, w)
        return ((y >= ylo) & (y <= yhi))[:, None] & ((x >= xlo) & (x <= xhi))[None, :]

    def get_valid_mask(self):
        """(rows, cols) bool: positions where the template lies fully inside the micrograph (all True if
        edges are not excluded; see Results.edges_excluded). The other positions are NaN in the maps."""
        assert self.compiled, "Results must be compiled first."
        return self.valid_mask

    def get_zscore(self):
        """(MIP - mean) / std per pixel (cisTEM's scaled MIP / Leopard-EM's z-score); 0 where std is 0."""
        std = self.get_std()
        with np.errstate(invalid="ignore"):
            z = np.where(std > 0, (self.compiled_mip - self.get_mean()) / np.where(std > 0, std, 1.0), 0.0)
        z[:, ~self.valid_mask] = np.nan
        return z

    def check_comparison(self, comparison_buffer: vd.RFFTBuffer, rotation_weights: vc.Var[vc.f32], *indices: vc.Var[vc.i32]):
        assert comparison_buffer.shape[0] % self.max_cross.shape[0] == 0, "Comparison buffer size must be a multiple of the number of templates."

        K = len(indices)
        weights_are_vars = not isinstance(rotation_weights, int)
        weight_types = [vc.Var[vc.f32]] * K if weights_are_vars else []
        scale_types = [vc.Buff[vc.f32]] if self.score_norms is not None else []
        inv_n = self.score_norm_inv_n

        # valid positions (template fully inside the micrograph): pixels outside are skipped entirely, so they cost
        # no accumulator traffic (they are NaN after compile anyway)
        H, W = self.micrograph_shape
        n_rows, n_cols = self.max_cross.shape[1], self.max_cross.shape[2]
        if self.edges_excluded((H, W)):
            ylo, yhi = valid_center_range(H, self.template_shape[0])
            xlo, xhi = valid_center_range(W, self.template_shape[1])
        else:
            ylo, yhi, xlo, xhi = -(2 ** 30), 2 ** 30, -(2 ** 30), 2 ** 30
        # map index after compile's fftshift, then the full-map (centred) index
        y_off = 0 if self.output_radius is None else H // 2 - self.output_radius
        x_off = 0 if self.output_radius is None else W // 2 - self.output_radius

        @vd.shader(
                exec_size=self.max_cross.size,
                arg_type_annotations=[
                    vc.Buff[vc.f32],  # max_cross
                    vc.Buff[vc.i32],  # best_index
                    vc.Buff[vc.v2],   # sum_cross
                    vc.Buff[vc.v2],   # sum2_cross
                    vc.Buff[vc.f32],  # cross_correlation
                ] + scale_types + weight_types + [vc.Var[vc.i32]] * K)
        def update_max(max_cross: vc.Buff[vc.f32],
                       best_index: vc.Buff[vc.i32],
                       sum_cross: vc.Buff[vc.v2],
                       sum2_cross: vc.Buff[vc.v2],
                       cross_correlation: vc.Buff[vc.f32],
                       *rest):
            scale_buff = rest[0] if scale_types else None
            rest = rest[len(scale_types):]
            rot_weights = list(rest[:K]) if weights_are_vars else [1.0] * K
            index_values = rest[len(weight_types):]

            ind = vc.global_invocation_id().x

            coord_3 = vc.ravel_index(ind, self.max_cross.shape).to_register()

            Y = ((coord_3.y.to_dtype(vc.i32) + n_rows // 2) % n_rows + y_off).to_register()
            X = ((coord_3.z.to_dtype(vc.i32) + n_cols // 2) % n_cols + x_off).to_register()
            with vc.if_block(vc.any(Y < ylo, Y > yhi, X < xlo, X > xhi)):
                vc.return_statement()

            coord_2 = coord_3.swizzle("yz").to_dtype(vc.iv2).to_register()
            coord_2 += vc.new_ivec2_register(self.max_cross.shape[1] // 2, self.max_cross.shape[2] // 2)
            coord_2[:] = vc.mod(coord_2, vc.new_ivec2_register(self.max_cross.shape[1], self.max_cross.shape[2])).to_dtype(vc.iv2)
            coord_2 -= vc.new_ivec2_register(self.max_cross.shape[1] // 2, self.max_cross.shape[2] // 2)
            coord_2[:] = vc.mod(coord_2, vc.new_ivec2_register(comparison_buffer.real_shape[1], comparison_buffer.real_shape[2])).to_dtype(vc.iv2)

            micrograph_inner_index = vc.unravel_index(coord_2, (comparison_buffer.real_shape[1], comparison_buffer.real_shape[2])).to_dtype(vc.i32).to_register()

            cross_corr_index = (micrograph_inner_index + 2 * (micrograph_inner_index // (comparison_buffer.real_shape[2]))).to_register()

            template_count = comparison_buffer.shape[0] // self.max_cross.shape[0]
            template_offset = comparison_buffer.shape[1] * comparison_buffer.shape[2] * 2 * template_count
            cross_corr_index[:] = cross_corr_index + coord_3.x.to_dtype(vc.i32) * template_offset

            micrograph_span_rfft = comparison_buffer.shape[1] * comparison_buffer.shape[2] * 2

            scales = None
            if scale_buff is not None:
                m = coord_3.x.to_dtype(vc.i32)
                scales = [(1.0 / vc.sqrt(scale_buff[m * template_count + i] * inv_n)).to_register()
                          for i in range(template_count)]

            best_mip_val, best_index_val, sum_cross_val, sum2_cross_val = accumulate(
                template_count,
                cross_correlation,
                cross_corr_index,
                micrograph_span_rfft,
                rot_weights,
                scales,
                *index_values
            )

            sum_cross[ind] = double_precision_add_vec2(sum_cross[ind], sum_cross_val)
            sum2_cross[ind] = double_precision_add_vec2(sum2_cross[ind], sum2_cross_val)

            with vc.if_block(best_mip_val > max_cross[ind]):
                max_cross[ind] = best_mip_val
                best_index[ind] = best_index_val

        update_max(
            self.max_cross,
            self.best_index,
            self.sum_cross,
            self.sum2_cross,
            comparison_buffer,
            *([self.score_norms] if self.score_norms is not None else []),
            *(list(rotation_weights) if weights_are_vars else []),
            *indices
        )

