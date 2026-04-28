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

    Sx_raw = (pos_2d.x/(wave_shape.x * pixel_size)).to_register()
    Sy_raw = (pos_2d.y/(wave_shape.y * pixel_size)).to_register()

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
