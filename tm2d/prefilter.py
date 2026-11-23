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

The CTF here is a NumPy copy of the slim formula of ctf.image_ctf_value (no phase plates, Zernike terms or
magnification distortion); the GPU evaluates it in float32, so the two agree to ~1e-6.
"""
import dataclasses
import math

import numpy as np

from .ctf.ctf import E_COMPTON, E_MASS, image_ctf_fast_ok
from .ctf.ctf_params import CTFParams
from .weights import RadialWeights, edge_mask


def ctf_rfft(shape: tuple, pixel_size: float, params: CTFParams) -> np.ndarray:
    """2 * CTF (tm2d's convention, as ctf_filter / image_ctf_value) on the rfft2 half-plane of an (H, W) image, 0 at DC.
    params: every field fixed (no None); phase plates, Zernike terms and magnification must be zero / identity."""
    for f in dataclasses.fields(params):
        if getattr(params, f.name) is None:
            raise ValueError(f"CTF field {f.name} is None (searched): give a fixed value")
    if not image_ctf_fast_ok(params):
        raise NotImplementedError("ctf_rfft covers the standard CTF only (no phase plates, Zernike terms or "
                                  "magnification distortion)")
    p = params
    H, W = (int(n) for n in shape)
    g = 1 + p.HT / E_MASS
    beta = math.sqrt(1 - 1 / (g * g))
    eps = g / (1 + p.HT / (2 * E_MASS))
    t1 = eps * p.energy_spread_fwhm / (p.HT * (2 * math.sqrt(2 * math.log(2))))
    t2 = eps * p.accel_voltage_std
    lam = E_COMPTON / (g * beta)
    sig_f = p.Cc * math.sqrt(t1 * t1 + t2 * t2 + 4 * p.OL_current_std ** 2)
    sy = np.fft.fftfreq(H)[:, None] / pixel_size
    sx = np.arange(W // 2 + 1)[None, :] / (W * pixel_size)
    s2 = sx * sx + sy * sy
    cws = p.Cs * lam * lam * s2
    ph = 0.5 * cws - p.defocus
    if p.A_mag != 0.0:
        a2 = np.deg2rad(2 * p.A_ang)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos2 = np.where(s2 > 0, ((sx * sx - sy * sy) * math.cos(a2) + 2 * sx * sy * math.sin(a2)) / s2, 0.0)
        ph = ph + p.A_mag * cos2
    gamma = np.pi * lam * s2 * ph
    expo = (0.5 * sig_f ** 2 * np.pi ** 2 * lam ** 2 * s2 * s2
            + 2 * p.beam_semiangle ** 2 * np.pi ** 2 * s2 * (cws - p.defocus) ** 2
            + 2 * np.pi ** 2 * p.johnson_std ** 2 * s2 + 0.25 * p.B * s2)
    c = 2.0 * np.exp(-expo) * np.sin(gamma - p.amp_contrast)
    c[0, 0] = 0.0
    return c


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
        g[i] = np.fft.irfft2(Y * w * C, s=(H, W))          # C is real here, so conj(C) = C
        nw[i] = w * C * C
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
