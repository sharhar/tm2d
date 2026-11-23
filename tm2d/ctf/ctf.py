import vkdispatch.codegen as vc

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

def phase_from_even_zernike(params: CTFParams, Sx_eff: vc.ShaderVariable, Sy_eff: vc.ShaderVariable) -> vc.ShaderVariable:
    psi = vc.new_float_register()
    r = vc.sqrt(Sx_eff * Sx_eff + Sy_eff * Sy_eff).to_register()
    th = vc.atan2(Sy_eff, Sx_eff).to_register()
    for i in range(9):
        with vc.if_block(params.get_even_coeff()[i] != 0.0):
            m, n = _even_index_to_mn(i)
            psi += params.get_even_coeff()[i] * _zernike_cart(m, n, r, th)

    return psi

def phase_from_odd_zernike(params: CTFParams, Sx_eff: vc.ShaderVariable, Sy_eff: vc.ShaderVariable) -> vc.ShaderVariable:
    psi = vc.new_float_register()
    r = vc.sqrt(Sx_eff * Sx_eff + Sy_eff * Sy_eff).to_register()
    th = vc.atan2(Sy_eff, Sx_eff).to_register()
    for i in range(6):
        with vc.if_block(params.get_odd_coeff()[i] != 0.0):
            m, n = _odd_index_to_mn(i)
            psi += params.get_odd_coeff()[i] * _zernike_cart(m, n, r, th)

    return psi # [rad]

def ctf_filter(
        buffer_shape: tuple[int, int],
        pos_2d: vc.ShaderVariable,
        params: CTFParams,
        pixel_size: float,
        disable_B_factor: bool = False):
    vc.comment("Calculating CTF")
    gamma_lorentz = 1 + params.HT/E_MASS
    beta = vc.sqrt(1 - 1/(gamma_lorentz*gamma_lorentz))

    epsilon = gamma_lorentz/(1 + params.HT/(2*E_MASS))

    temp1 = epsilon*params.energy_spread_fwhm/( params.HT * (2 * np.sqrt(2 * np.log(2))) )
    temp2 = epsilon*params.accel_voltage_std

    wlen = E_COMPTON / (gamma_lorentz * beta) # wavelength in Angstroms

    sigma_f = params.Cc*vc.sqrt(temp1*temp1 + temp2*temp2 + 4*params.OL_current_std*params.OL_current_std)

    Cs = params.Cs # [A]
    sigma_beta = params.beam_semiangle # [rad]
    johnson_std = params.johnson_std # [A]
    amp_contrast = params.amp_contrast # [dimensionless]
    zernike = params.zernike # [deg]
    lpp = params.lpp # [deg]
    NA = params.NA # [dimensionless]
    f_OL = params.f_OL # [A]
    lpp_rot = params.lpp_rot # [deg]

    lpp_rot_rad = np.pi / 180 * lpp_rot # [rad]
    A2_mag = params.A_mag # [A]
    A2_ang = params.A_ang # [deg]
    A2_ang_rad = np.pi / 180 * A2_ang # [rad]

    wave_shape = vc.new_uvec2_register()
    wave_shape.x = buffer_shape[0]
    wave_shape.y = buffer_shape[1] * 2 - 2

    # pos_2d.x is the rfft column index kx -> s_x = kx / (W pix); pos_2d.y is the signed row index ky -> s_y = ky / (H pix)
    # (wave_shape.x = H rows, wave_shape.y = W columns)
    Sx_raw = (pos_2d.x/(wave_shape.y * pixel_size)).to_register()
    Sy_raw = (pos_2d.y/(wave_shape.x * pixel_size)).to_register()

    Sx = (params.mag_00 * Sx_raw + params.mag_01 * Sy_raw).to_register()
    Sy = (params.mag_10 * Sx_raw + params.mag_11 * Sy_raw).to_register()

    psi_even = phase_from_even_zernike(params, Sx, Sy)
    psi_odd = phase_from_odd_zernike(params, Sx, Sy)

    Sr_2 = (Sx*Sx + Sy*Sy).to_register()
    Sa = (vc.atan2(Sy, Sx) + np.pi/2).to_register()

    wlen_2 = wlen*wlen

    CWS = Cs*wlen_2*Sr_2.to_register()

    defocus = params.defocus

    gamma = np.pi*wlen*Sr_2*((CWS*0.5) - defocus - A2_mag*vc.cos(2*(Sa - A2_ang_rad))).to_register()

    # zernike phase plate
    eta_z = -np.pi / 180 * zernike

    # laser phase plate
    Dx = f_OL * wlen * (Sy * vc.cos(lpp_rot_rad) + Sx * vc.sin(lpp_rot_rad)) # bfp spatial x coord [A]
    Dy = f_OL * wlen * (-Sy * vc.sin(lpp_rot_rad) + Sx * vc.cos(lpp_rot_rad)) # bfp spatial y coord [A]
    w0 = 1.064e4 / (np.pi * NA) # beam waist radius [A]
    zR = w0 / NA # rayleigh range [A]
    w = w0 * vc.sqrt(1 + (Dx / zR)*(Dx / zR)) # waist at Dx [A]
    k = 2 * np.pi / 1.064e4 # wave number [1/A]
    eta_l = (np.pi / 180 * lpp) * (0.5 * (w0 / w) * vc.exp(-2 * Dy*Dy / (w*w)) * (1 + vc.cos(2 * k * Dx)) - 1) # [rad]

    # total phase plate phase
    eta = eta_z + eta_l

    sinchi = vc.sin(psi_even + eta + gamma - amp_contrast).to_register()
    E_TC = vc.exp(-0.5 * sigma_f * sigma_f * np.pi * np.pi * wlen_2 * Sr_2 * Sr_2).to_register()
    E_SC = vc.exp(-2 * sigma_beta * sigma_beta * np.pi * np.pi * Sr_2 * (CWS - defocus)*(CWS - defocus)).to_register()
    E_JN = vc.exp(-2 * np.pi * np.pi * johnson_std*johnson_std * Sr_2).to_register()
    E_BF = vc.exp(-0.25 * params.B * Sr_2).to_register() if not disable_B_factor else 1.0

    CTF_scale = (2 * E_TC * E_SC * E_JN * E_BF * sinchi).to_register() # technically 2 * ctf

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
        #index = ind + i * signal_shape[1] * signal_shape[2]

        registers[i][:] = vc.mult_complex(input_value, ctf_filter(
            signal_shape[1:],
            pos_2d,
            ctf_param_list[indices[i]],
            pix_size
        ))

def _is_static_zero(v) -> bool:
    return v is not None and not isinstance(v, vc.ShaderVariable) and float(v) == 0.0


def image_ctf_fast_ok(static_params: CTFParams) -> bool:
    """Whether image_ctf_value can use the slim formula: no phase plates, Zernike terms or magnification distortion,
    each fixed (not searched) at zero / identity in the plan's CTFParams (None = searched)."""
    p = static_params
    zero = ["zernike", "lpp"] + [f"e_zern_{i}" for i in range(9)] + [f"o_zern_{i}" for i in range(6)] + ["mag_01", "mag_10"]
    if not all(_is_static_zero(getattr(p, f)) for f in zero):
        return False
    return all(getattr(p, f) is not None and float(getattr(p, f)) == 1.0 for f in ("mag_00", "mag_11"))


def _mixed(static_value, var):
    """A field's value for code generation: the Python float if fixed in the plan's CTFParams, else the shader
    variable (searched field)."""
    return float(static_value) if static_value is not None else var


def _f(x, py, sh):
    return py(x) if isinstance(x, (int, float)) else sh(x)


def image_ctf_value(H: int, W: int, row_signed: vc.ShaderVariable, col: vc.ShaderVariable,
                    pixel_size, params: CTFParams, static_params: CTFParams = None) -> vc.ShaderVariable:
    """The CTF (ctf_filter's 2 * CTF, complex if there are odd aberrations) at rfft2 bin (signed row ky, column kx)
    of an (H, W) micrograph: s = (kx / (W pix), ky / (H pix)). Used by the image-side CTF path of
    ComparatorCrossCorrelation, which evaluates it per Fourier pixel of the micrograph for every kernel.

    static_params: the plan's CTFParams (None fields = searched). If image_ctf_fast_ok(static_params), the same
    formula as ctf_filter is generated with the zero terms removed and the fixed fields folded in as Python
    constants: astigmatism from cos 2(theta - A_ang) = ((sx^2 - sy^2) cos 2A + 2 sx sy sin 2A) / s^2 (no atan2), and the
    four envelopes as one exp, i.e. one sin and one exp per value. Otherwise the general ctf_filter."""
    pos_2d = vc.new_vec2_register()
    pos_2d.x = col.to_dtype(vc.f32)
    pos_2d.y = row_signed.to_dtype(vc.f32)
    if static_params is None or not image_ctf_fast_ok(static_params):
        return ctf_filter((H, W // 2 + 1), pos_2d, params, pixel_size)

    sp = static_params
    g = {f: _mixed(getattr(sp, f), getattr(params, f)) for f in (
        "HT", "Cs", "Cc", "energy_spread_fwhm", "accel_voltage_std", "OL_current_std", "beam_semiangle",
        "johnson_std", "B", "amp_contrast", "defocus", "A_mag", "A_ang")}
    sqrt = lambda x: _f(x, math.sqrt, vc.sqrt)
    HT = g["HT"]
    gamma_lorentz = 1 + HT / E_MASS
    beta = sqrt(1 - 1 / (gamma_lorentz * gamma_lorentz))
    epsilon = gamma_lorentz / (1 + HT / (2 * E_MASS))
    temp1 = epsilon * g["energy_spread_fwhm"] / (HT * (2 * np.sqrt(2 * np.log(2))))
    temp2 = epsilon * g["accel_voltage_std"]
    wlen = E_COMPTON / (gamma_lorentz * beta)
    sigma_f = g["Cc"] * sqrt(temp1 * temp1 + temp2 * temp2 + 4 * g["OL_current_std"] * g["OL_current_std"])
    wlen_2 = wlen * wlen

    Sx = (pos_2d.x / (W * pixel_size)).to_register()
    Sy = (pos_2d.y / (H * pixel_size)).to_register()
    Sr_2 = (Sx * Sx + Sy * Sy).to_register()
    CWS = (g["Cs"] * wlen_2 * Sr_2).to_register()
    defocus = g["defocus"]
    phase_df = (CWS * 0.5 - defocus).to_register()
    if not _is_static_zero(g["A_mag"]):
        # ctf_filter: - A cos(2 (atan2(Sy, Sx) + pi/2 - A_ang)) = + A cos(2 (theta - A_ang))
        a2 = np.pi / 180 * 2 * g["A_ang"]
        cos2a = _f(a2, math.cos, vc.cos)
        sin2a = _f(a2, math.sin, vc.sin)
        inv_s2 = vc.new_float_register(0.0)
        with vc.if_block(Sr_2 > 0.0):
            inv_s2[:] = 1.0 / Sr_2
        cos2 = (((Sx * Sx - Sy * Sy) * cos2a + 2.0 * Sx * Sy * sin2a) * inv_s2).to_register()
        phase_df[:] = phase_df + g["A_mag"] * cos2
    gamma = (np.pi * wlen * Sr_2 * phase_df).to_register()
    sinchi = vc.sin(gamma - g["amp_contrast"]).to_register()
    cw_df = (CWS - defocus).to_register()
    expo = (0.5 * sigma_f * sigma_f * np.pi * np.pi * wlen_2 * Sr_2 * Sr_2
            + 2 * g["beam_semiangle"] * g["beam_semiangle"] * np.pi * np.pi * Sr_2 * cw_df * cw_df
            + 2 * np.pi * np.pi * g["johnson_std"] * g["johnson_std"] * Sr_2
            + 0.25 * g["B"] * Sr_2).to_register()
    ctf = vc.new_complex64_register()
    ctf.real = 2.0 * vc.exp(-expo) * sinchi
    ctf.imag = 0.0
    with vc.if_block(Sr_2 == 0.0):
        ctf.real = 0.0
    return ctf
