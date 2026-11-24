import vkdispatch.codegen as vc

import dataclasses
import math

import numpy as np

from .ctf_params import CTFParams

E_MASS = 0.511e6 # eV/c^2
E_COMPTON = 2.42631e-2 # A
#ALPHA = 0.0072973525693
#HBAR = 1.0545718001e-34
#C_REL = 299792458

def do_int_check(other):
    return isinstance(other, int) or np.issubdtype(type(other), np.integer)

def _even_index_to_mn(i: int) -> tuple[int, int]:
    # Z0^0, Z2^-2, Z2^0, Z2^2, Z4^-4, Z4^-2, Z4^0, ...
    count = 0
    n = 0
    while True:
        for m in range(-n, n + 1, 2):
            if count == i:
                return m, n
            count += 1
        n += 2

def _odd_index_to_mn(i: int) -> tuple[int, int]:
    # Z1^-1, Z1^1, Z3^-3, Z3^-1, Z3^1, ...
    count = 0
    n = 1
    while True:
        for m in range(-n, n + 1, 2):
            if count == i:
                return m, n
            count += 1
        n += 2

def _zernike_radial(n: int, m: int, r: vc.ShaderVariable) -> vc.ShaderVariable:
    m = abs(m)
    if (n - m) % 2 != 0:
        return 0

    out = vc.new_float_register()
    for s in range((n - m) // 2 + 1):
        c = ((-1) ** s) * math.factorial(n - s) / (
            math.factorial(s)
            * math.factorial((n + m) // 2 - s)
            * math.factorial((n - m) // 2 - s)
        )
        out += c * r ** (n - 2 * s)
    return out

def _zernike_cart(m: int, n: int, r: vc.ShaderVariable, th: vc.ShaderVariable) -> vc.ShaderVariable:
    R = _zernike_radial(n, m, r)
    if m == 0:
        return R
    if m > 0:
        return R * vc.cos(m * th)
    return R * vc.sin(abs(m) * th)

def _is_num(x) -> bool:
    """A field fixed at code-generation time (Python / NumPy number), as opposed to a shader variable."""
    return isinstance(x, (int, float, np.integer, np.floating))


def _is_zero(x) -> bool:
    return _is_num(x) and float(x) == 0.0


def _fn(x, py, sh):
    """py(x) for a fixed value (folded into the generated code), sh(x) for a shader variable."""
    return py(float(x)) if _is_num(x) else sh(x)


def _sqrt(x):
    return _fn(x, math.sqrt, vc.sqrt)


def _sin(x):
    return _fn(x, math.sin, vc.sin)


def _cos(x):
    return _fn(x, math.cos, vc.cos)


def _sum(terms):
    """Sum of the terms that are present (None = term not generated); None if none is."""
    terms = [t for t in terms if t is not None]
    if not terms:
        return None
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def fold_ctf_params(params: CTFParams, static_params: CTFParams = None) -> CTFParams:
    """params with every field that is fixed (not None) in static_params replaced by its Python float, so that
    ctf_filter folds it into the generated code. static_params is the plan's CTFParams (None = searched field)."""
    if static_params is None:
        return params
    out = CTFParams.from_arg_list(*[getattr(params, f.name) for f in dataclasses.fields(params)])
    for f in dataclasses.fields(static_params):
        v = getattr(static_params, f.name)
        if v is not None:
            setattr(out, f.name, float(v))
    return out


def _zernike_phase(coeffs, index_to_mn, Sx, Sy):
    """sum_i c_i Z_i(r, theta) over the coefficients that can be nonzero (a fixed zero generates no code; a searched
    one is guarded by an if on its value); None if all are fixed at zero."""
    if all(_is_zero(c) for c in coeffs):
        return None
    psi = vc.new_float_register()
    r = vc.sqrt(Sx * Sx + Sy * Sy).to_register()
    th = vc.atan2(Sy, Sx).to_register()
    for i, c in enumerate(coeffs):
        if _is_zero(c):
            continue
        m, n = index_to_mn(i)
        if _is_num(c):
            psi += float(c) * _zernike_cart(m, n, r, th)
        else:
            with vc.if_block(c != 0.0):
                psi += c * _zernike_cart(m, n, r, th)
    return psi


def phase_from_even_zernike(params: CTFParams, Sx_eff: vc.ShaderVariable, Sy_eff: vc.ShaderVariable):
    return _zernike_phase(params.get_even_coeff(), _even_index_to_mn, Sx_eff, Sy_eff)


def phase_from_odd_zernike(params: CTFParams, Sx_eff: vc.ShaderVariable, Sy_eff: vc.ShaderVariable):
    return _zernike_phase(params.get_odd_coeff(), _odd_index_to_mn, Sx_eff, Sy_eff)  # [rad]


def ctf_filter(
        buffer_shape: tuple[int, int],
        pos_2d: vc.ShaderVariable,
        params: CTFParams,
        pixel_size: float,
        disable_B_factor: bool = False,
        static_params: CTFParams = None):
    """2 * CTF (complex if there are odd aberrations) at rfft2 bin pos_2d = (column kx, signed row ky) of an image
    whose rfft2 half-plane is buffer_shape = (H, W // 2 + 1): s = (kx / (W pix), ky / (H pix)); 0 at DC.

    The one CTF implementation of tm2d: templates (apply_ctfs_to_rfft_signals), the image-side CTF of
    ComparatorCrossCorrelation and tm2d.prefilter.ctf_rfft all generate it here. Each field of params is either a
    shader variable or a Python number; static_params (the plan's CTFParams, None = searched) turns every field fixed
    there into a Python number (fold_ctf_params). Fixed fields are folded into the generated code, and terms fixed at
    zero or identity (phase plates, Zernike terms, magnification distortion, astigmatism, envelopes) generate no code,
    so the usual case costs one sin and one exp per value. Astigmatism uses
    cos 2(theta - A_ang) = ((sx^2 - sy^2) cos 2A + 2 sx sy sin 2A) / s^2 (no atan2); the four envelopes are one exp."""
    vc.comment("Calculating CTF")
    p = fold_ctf_params(params, static_params)

    gamma_lorentz = 1 + p.HT / E_MASS
    beta = _sqrt(1 - 1 / (gamma_lorentz * gamma_lorentz))
    epsilon = gamma_lorentz / (1 + p.HT / (2 * E_MASS))
    wlen = E_COMPTON / (gamma_lorentz * beta)  # wavelength in Angstroms
    wlen_2 = wlen * wlen

    H, W = int(buffer_shape[0]), int(buffer_shape[1]) * 2 - 2
    # pos_2d.x: rfft column index kx -> s_x = kx / (W pix); pos_2d.y: signed row index ky -> s_y = ky / (H pix)
    Sx_raw = (pos_2d.x / (W * pixel_size)).to_register()
    Sy_raw = (pos_2d.y / (H * pixel_size)).to_register()
    identity = all(_is_num(getattr(p, f)) and float(getattr(p, f)) == v
                   for f, v in (("mag_00", 1.0), ("mag_01", 0.0), ("mag_10", 0.0), ("mag_11", 1.0)))
    if identity:
        Sx, Sy = Sx_raw, Sy_raw
    else:
        Sx = (p.mag_00 * Sx_raw + p.mag_01 * Sy_raw).to_register()
        Sy = (p.mag_10 * Sx_raw + p.mag_11 * Sy_raw).to_register()

    psi_even = phase_from_even_zernike(p, Sx, Sy)
    psi_odd = phase_from_odd_zernike(p, Sx, Sy)

    Sr_2 = (Sx * Sx + Sy * Sy).to_register()
    CWS = (p.Cs * wlen_2 * Sr_2).to_register()
    defocus = p.defocus
    phase_df = (CWS * 0.5 - defocus).to_register()
    if not _is_zero(p.A_mag):
        # - A cos(2 (atan2(Sy, Sx) + pi/2 - A_ang)) = + A cos(2 (theta - A_ang)), theta = atan2(Sy, Sx)
        a2 = (np.pi / 180 * 2) * p.A_ang
        inv_s2 = vc.new_float_register(0.0)
        with vc.if_block(Sr_2 > 0.0):
            inv_s2[:] = 1.0 / Sr_2
        cos2 = (((Sx * Sx - Sy * Sy) * _cos(a2) + 2.0 * Sx * Sy * _sin(a2)) * inv_s2).to_register()
        phase_df[:] = phase_df + p.A_mag * cos2
    gamma = (np.pi * wlen * Sr_2 * phase_df).to_register()

    # Zernike phase plate
    eta_z = None if _is_zero(p.zernike) else -np.pi / 180 * p.zernike
    # laser phase plate
    eta_l = None
    if not _is_zero(p.lpp):
        lpp_rot_rad = np.pi / 180 * p.lpp_rot  # [rad]
        Dx = p.f_OL * wlen * (Sy * _cos(lpp_rot_rad) + Sx * _sin(lpp_rot_rad))  # bfp spatial x coord [A]
        Dy = p.f_OL * wlen * (-Sy * _sin(lpp_rot_rad) + Sx * _cos(lpp_rot_rad))  # bfp spatial y coord [A]
        w0 = 1.064e4 / (np.pi * p.NA)  # beam waist radius [A]
        zR = w0 / p.NA  # rayleigh range [A]
        w = w0 * vc.sqrt(1 + (Dx / zR) * (Dx / zR))  # waist at Dx [A]
        k = 2 * np.pi / 1.064e4  # wave number [1/A]
        eta_l = (np.pi / 180 * p.lpp) * (0.5 * (w0 / w) * vc.exp(-2 * Dy * Dy / (w * w)) * (1 + vc.cos(2 * k * Dx)) - 1)

    sinchi = vc.sin(_sum([psi_even, eta_z, eta_l, gamma]) - p.amp_contrast).to_register()

    # envelopes: temporal coherence, spatial coherence, Johnson noise, B factor, as one exp
    temp1 = epsilon * p.energy_spread_fwhm / (p.HT * (2 * np.sqrt(2 * np.log(2))))
    temp2 = epsilon * p.accel_voltage_std
    sigma_f = p.Cc * _sqrt(temp1 * temp1 + temp2 * temp2 + 4 * p.OL_current_std * p.OL_current_std)
    expo = _sum([
        None if _is_zero(sigma_f) else 0.5 * sigma_f * sigma_f * np.pi * np.pi * wlen_2 * Sr_2 * Sr_2,
        None if _is_zero(p.beam_semiangle) else
        2 * p.beam_semiangle * p.beam_semiangle * np.pi * np.pi * Sr_2 * (CWS - defocus) * (CWS - defocus),
        None if _is_zero(p.johnson_std) else 2 * np.pi * np.pi * p.johnson_std * p.johnson_std * Sr_2,
        None if (disable_B_factor or _is_zero(p.B)) else 0.25 * p.B * Sr_2,
    ])
    CTF_scale = (2.0 * sinchi if expo is None else 2.0 * vc.exp(-expo) * sinchi).to_register()  # 2 * ctf

    if psi_odd is None:
        CTF = vc.new_complex64_register()
        CTF.real = CTF_scale
        CTF.imag = 0.0
    else:
        CTF = vc.complex_from_euler_angle(psi_odd).to_register()
        CTF.real *= CTF_scale
        CTF.imag *= CTF_scale

    with vc.if_block(vc.all(Sx == 0.0, Sy == 0.0)):
        CTF.real = 0.0
        CTF.imag = 0.0

    return CTF

def apply_ctfs_to_rfft_signals(
        input_value: vc.ShaderVariable,
        registers: list[vc.ShaderVariable],
        indices: list[int],
        ind: vc.ShaderVariable,
        signal_shape: tuple[int, int, int],
        pix_size: vc.ShaderVariable,
        ctf_params: CTFParams,
        args: list[vc.ShaderVariable],
):
    iind = ind.to_dtype(vc.i32)

    ipos_2d = vc.new_ivec2_register()
    ipos_2d.x = iind % signal_shape[2]
    ipos_2d.y = ((iind // signal_shape[2]) + signal_shape[1] // 2) % signal_shape[1]

    pos_2d = ipos_2d.to_dtype(vc.v2).to_register()
    pos_2d.y = pos_2d.y - signal_shape[1] // 2



    ctf_param_list = ctf_params.assemble_params_list_from_args(args, signal_shape[0])

    for i in range(len(registers)):
        # ctf_params (the plan's, None = searched): its fixed fields are folded into the generated code
        registers[i][:] = vc.mult_complex(input_value, ctf_filter(
            signal_shape[1:],
            pos_2d,
            ctf_param_list[indices[i]],
            pix_size,
            static_params=ctf_params
        ))
