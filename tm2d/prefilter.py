"""Prefiltered micrographs for coarse-to-fine searches (log entry 29).

The image-side CTF search scores X_d(u) = sum_s conj(T) conj(C_d) Y w e^{2 pi i s.u} / N / ||c_d * t||_w (see
ComparatorCrossCorrelation.compare_template). The image part, g_d = F^-1[Y w conj(C_d)], does not depend on the
template, so it can be computed once for the whole micrograph and cropped: on a crop that holds the template box
around a position, the correlation of the CTF-free template with g_d is exactly the full search's score there (the
score is a real-space sum over the template box). The norm needs the noise weights of g_d, w |C_d|^2 on the full
grid; ComparatorCrossCorrelation(..., prefiltered=True).norm_lag_weights turns them into exact lag-grid weights.

    g, noise_w = prefilter_micrograph(data, radial_weights, ctf_params, defoci, pixel_size)
    comp = ComparatorCrossCorrelation((M, L, L), template_shape, prefiltered=True)
    A = comp.norm_lag_weights(noise_w, data.shape)          # once per defocus
    comp.set_data(crops_of_g, A_for_each_crop)              # Plan(...) renders CTF-free templates for this comparator

The CTF is tm2d's one implementation, ctf.ctf_filter, evaluated on the GPU on the micrograph's rfft2 grid with the
same row convention as the comparator's image-side CTF, so the prefiltered search and the full image-side search use
the same C_d (any CTFParams: phase plates, Zernike terms and complex CTFs included).
"""
import dataclasses

import numpy as np

from .ctf.ctf import ctf_filter
from .ctf.ctf_params import CTFParams
from .weights import RadialWeights, edge_mask


def ctf_rfft(shape: tuple, pixel_size: float, params: CTFParams) -> np.ndarray:
    """2 * CTF (tm2d's convention, ctf.ctf_filter) on the rfft2 half-plane (H, W // 2 + 1) of an (H, W) image, 0 at DC,
    complex128. Evaluated by ctf_filter on the GPU with every field folded in as a constant; rows are signed as in
    ComparatorCrossCorrelation's image-side CTF (row r > H // 2 is ky = r - H, the Nyquist row is +H // 2).
    params: every field fixed (no None)."""
    for f in dataclasses.fields(params):
        if getattr(params, f.name) is None:
            raise ValueError(f"CTF field {f.name} is None (searched): give a fixed value")
    import vkdispatch as vd
    import vkdispatch.codegen as vc
    H, W = (int(n) for n in shape)
    C = W // 2 + 1
    pix = float(pixel_size)
    out = vd.Buffer((H, C), vd.complex64)

    @vd.shader(exec_size=H * C)
    def ctf_shader(buff: vc.Buff[vc.c64]):
        ind = vc.global_invocation_id().x.to_dtype(vc.i32).to_register()
        row = (ind // C).to_register()
        pos = vc.new_vec2_register()
        pos.x = (ind % C).to_dtype(vc.f32)
        pos.y = row.to_dtype(vc.f32)
        with vc.if_block(row > H // 2):
            pos.y = (row - H).to_dtype(vc.f32)
        buff[ind] = ctf_filter((H, C), pos, params, pix, static_params=params)

    ctf_shader(out)
    return out.read(0).reshape(H, C).astype(np.complex128)


def prefilter_micrograph(data: np.ndarray, weights: RadialWeights, ctf_params: CTFParams, defoci, pixel_size: float):
    """The image side of the search, for each defocus: g_d = irfft2(rfft2(data) * w * conj(C_d)), with w the
    comparator's own resampled table (comparator_radial_table) so that crops of g_d reproduce a full fused search.

    data: (H, W) micrograph (mean-subtracted or not: w is 0 at DC). ctf_params: fixed fields except defocus, which
    may be None (taken from `defoci`). Returns (g, noise_weights): g (D, H, W) float32, noise_weights (D, H, W//2+1)
    float64 = w |C_d|^2 (with DC and the Nyquist lines 0), for ComparatorCrossCorrelation.norm_lag_weights."""
    from .comparators.cross_correlation import comparator_radial_table
    data = np.asarray(data, dtype=np.float64)
    H, W = data.shape
    w = comparator_radial_table(weights, (1, H, W)).evaluate((H, W))
    w[edge_mask((H, W))] = 0.0
    Y = np.fft.rfft2(data)
    g = np.empty((len(defoci), H, W), np.float32)
    nw = np.empty((len(defoci), H, W // 2 + 1))
    for i, d in enumerate(defoci):
        p = CTFParams.from_arg_list(*[getattr(ctf_params, f.name) for f in dataclasses.fields(ctf_params)])
        p.defocus = float(d)
        C = ctf_rfft((H, W), pixel_size, p)
        g[i] = np.fft.irfft2(Y * w * np.conj(C), s=(H, W))
        nw[i] = w * np.abs(C) ** 2
    return g, nw


def crop(images: np.ndarray, center_yx, size: int):
    """Square crops (..., size, size) of (..., H, W) images, centred so that crop pixel (size // 2, size // 2) is
    image pixel center_yx; also returns the image coordinates (y0, x0) of crop pixel (0, 0). Raises if the crop
    leaves the image."""
    y0, x0 = int(center_yx[0]) - size // 2, int(center_yx[1]) - size // 2
    H, W = images.shape[-2:]
    if y0 < 0 or x0 < 0 or y0 + size > H or x0 + size > W:
        raise ValueError(f"crop {size} at {tuple(center_yx)} leaves the {H} x {W} image")
    return images[..., y0:y0 + size, x0:x0 + size], (y0, x0)
