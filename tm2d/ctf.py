import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import dataclasses

import itertools
import numpy as np

from typing import Generator

E_MASS = 0.511e6 # eV/c^2
E_COMPTON = 2.42631e-2 # A
ALPHA = 0.0072973525693
HBAR = 1.0545718001e-34
C_REL = 299792458

@dataclasses.dataclass
class CTFParams:
    HT: vd.float32
    Cs: vd.float32
    Cc: vd.float32
    energy_spread_fwhm: vd.float32
    beta: vd.float32
    wlen: vd.float32
    sigma_beta: vd.float32
    sigma_s: vd.float32
    sigma_f: vd.float32
    johnson_std: vd.float32
    B: vd.float32
    amp_contrast: vd.float32
    zernike: vd.float32
    lpp: vd.float32
    NA: vd.float32
    f_OL: vd.float32
    lpp_rot: vd.float32
    defocus: vd.float32
    A_mag: vd.float32
    A_ang: vd.float32

    def get_type_list(self):
        fields = dataclasses.fields(self)
        values = [self.__getattribute__(field.name) for field in fields]
        types = [vc.Var[vc.f32] if value is None else vc.Const[vc.f32] for value in values]
        return types
    
    def get_args(self, cmd_stream: vd.CommandStream, template_count: int):
        args = []

        fields = dataclasses.fields(self)
        values = [self.__getattribute__(field.name) for field in fields]

        for batch_id in range(template_count):
            for value, field in zip(values, fields):
                if value is None:
                    args.append(cmd_stream.bind_var(f"{field.name}_{batch_id}"))
                else:
                    args.append(value)

        return args
    
    def assemble_params_list_from_args(self, args_list: list[vc.Var], template_count: int):
        fields = dataclasses.fields(self)
        
        assert len(args_list) == len(fields) * template_count, f"Expected {len(fields) * template_count} args, got {len(args_list)}"

        result = []

        for i in range(template_count):
            result.append(CTFParams(
                *args_list[i * len(fields):(i + 1) * len(fields)],
            ))
            
        return result

    def generate_iteration_information(self, values_dict: dict[str, np.ndarray]):
        fields = dataclasses.fields(self)

        dynamic_fields = [field for field in fields if self.__getattribute__(field.name) is None]

        dynamic_field_names = {field.name for field in dynamic_fields}
        values_dict_keys = set(values_dict.keys())
        if dynamic_field_names != values_dict_keys:
            raise ValueError(f"Dynamic field names {dynamic_field_names} do not match keys in values_dict {values_dict_keys}")
        
        dynamic_values = [values_dict[field.name] for field in dynamic_fields]

        for dyn_val, field in zip(dynamic_values, dynamic_fields):
            assert dyn_val.ndim == 1, f"Dynamic value for field {field.name} must be a 1D array, got {dyn_val.ndim}D array."

        #print(dynamic_values)
        #grid = np.meshgrid(*dynamic_values)
        #print(f"Grid shape: {grid}")
        #combinations_array = np.vstack([g.ravel() for g in grid]).T

        combinations = list(itertools.product(*dynamic_values))
        combinations_array = np.array(combinations)

        return combinations_array, dynamic_fields
        
    def get_values_at_index(self, values_dict: dict[str, np.ndarray], index: int):
        lengths = [len(values_dict[field.name]) for field in dataclasses.fields(self) if self.__getattribute__(field.name) is None]

        individual_indices = [0] * len(lengths)
        current_index = index
        
        # Iterate from the last array to the first
        for i in range(len(lengths) - 1, -1, -1):
            individual_indices[i] = current_index % lengths[i]
            current_index //= lengths[i]

        return_dict = {}
        field_index = 0
        for field in dataclasses.fields(self):
            if self.__getattribute__(field.name) is None:
                return_dict[field.name] = values_dict[field.name][individual_indices[field_index]]
                field_index += 1
            
        return return_dict

def make_ctf_params(
        HT: float = 300e3,
        Cs: float = 2.7e7,
        Cc: float = 2.7e7,
        energy_spread_fwhm: float = 0.3,
        accel_voltage_std: float = 0.07e-6,
        OL_current_std: float = 0.1e-6,
        beam_semiangle: float = 2.5e-6,
        johnson_std: float = 0.33,
        B: float = 0,
        amp_contrast: float = 0.07,
        zernike: float = 0,
        lpp: float = 0,
        NA: float = 0.05,
        f_OL: float = 20e7,
        lpp_rot: float = 0,
        defocus: float = None,
        A_mag: float = 0,
        A_ang: float = 0
    ) -> CTFParams:

    gamma_lorentz = 1 + HT/E_MASS
    beta = np.sqrt(1 - 1/(gamma_lorentz*gamma_lorentz))
    
    epsilon = gamma_lorentz/(1 + HT/(2*E_MASS))

    temp1 = epsilon*energy_spread_fwhm/HT/(2 * np.sqrt(2 * np.log(2)))
    temp2 = epsilon*accel_voltage_std
    
    params = CTFParams(
        HT=HT,
        Cs=Cs,
        Cc=Cc,
        energy_spread_fwhm=energy_spread_fwhm,
        beta=beta,
        wlen=E_COMPTON / (gamma_lorentz * beta),
        sigma_beta=beam_semiangle,
        sigma_s=beam_semiangle/(E_COMPTON / (gamma_lorentz * beta)),
        sigma_f=Cc*np.sqrt(temp1*temp1 + temp2*temp2 + 4*OL_current_std*OL_current_std),
        johnson_std=johnson_std,
        B=B,
        amp_contrast=amp_contrast,
        zernike=zernike,
        lpp=lpp,
        NA=NA,
        f_OL=f_OL,
        lpp_rot=lpp_rot,
        defocus=defocus,
        A_mag=A_mag,
        A_ang=A_ang
    )

    return params

def get_ctf_params_set(microscope_name: str) -> CTFParams:
    if microscope_name == 'titan':
        ctf_params = make_ctf_params(
            HT = 300e3,
            Cs = 4.8e7,
            Cc = 7.6e7,
            energy_spread_fwhm = 0.9,
            accel_voltage_std = 0.07e-6,
            OL_current_std = 0.1e-6,
            beam_semiangle = 2.5e-6,
            johnson_std = 0.37,
            B = 0,
            amp_contrast = 0.07,
            lpp = 90,
            NA = 0.05,
            f_OL = 20e7,
            lpp_rot = 18
        )

    elif microscope_name == 'theia':
        ctf_params = make_ctf_params(
            HT = 300e3,
            Cs = 0,
            Cc = 5.1e7,
            energy_spread_fwhm = 0.3,
            accel_voltage_std = 0.07e-6,
            OL_current_std = 0.1e-6,
            beam_semiangle = 2.5e-6,
            johnson_std = 0.33,
            B = 0,
            amp_contrast = 0.07,
            lpp = 90,
            NA = 0.05,
            f_OL = 14.1e7,
            lpp_rot = 18
        )

    elif microscope_name == 'krios':
        ctf_params = make_ctf_params(
            HT = 300e3,
            Cs = 2.7e7,
            Cc = 2.7e7,
            energy_spread_fwhm = 0.3,
            accel_voltage_std = 0.07e-6,
            OL_current_std = 0.1e-6,
            beam_semiangle = 2.5e-6,
            johnson_std = 0,
            B = 0,
            amp_contrast = 0.07,
            lpp = 0,
        )

    return ctf_params

def ctf_filter(
        buffer_shape: tuple[int, int],
        #defocus_vars: Var[v4],
        pos_2d: vc.ShaderVariable,
        params: CTFParams,
        pixel_size: float,
        disable_B_factor: bool = False):
    wlen = params.wlen # [A]
    Cs = params.Cs # [A]
    sigma_beta = params.sigma_beta # [rad]
    johnson_std = params.johnson_std # [A]
    sigma_f = params.sigma_f # [A]
    amp_contrast = params.amp_contrast # [dimensionless]
    zernike = params.zernike # [deg]
    lpp = params.lpp # [deg]
    # lpp = defocus_vars.w # [deg] laser phase shift
    NA = params.NA # [dimensionless]
    f_OL = params.f_OL # [A]
    lpp_rot = params.lpp_rot # [deg]
    
    lpp_rot_rad = np.pi / 180 * lpp_rot # [rad]
    A2_mag = params.A_mag #params[ctf_params_index_table["A2_mag"]] # [A]
    A2_ang = params.A_ang #params[ctf_params_index_table["A2_ang"]] # [deg]
    A2_ang_rad = np.pi / 180 * A2_ang # [rad]

    wave_shape = vc.new_uvec2()
    wave_shape.x = buffer_shape[0]
    wave_shape.y = buffer_shape[1] * 2 - 2

    Sx = (pos_2d.y/(wave_shape.y * pixel_size)).copy()
    Sy = (-pos_2d.x/(wave_shape.x * pixel_size)).copy()

    Sr_2 = (Sx*Sx + Sy*Sy).copy()
    Sa = vc.atan2(Sy, Sx).copy()
    
    wlen_2 = wlen*wlen

    CWS = Cs*wlen_2*Sr_2

    defocus = params.defocus

    gamma = np.pi*wlen*Sr_2*((CWS*0.5) - defocus - A2_mag*vc.cos(2*(Sa - A2_ang_rad))).copy()

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

    sinchi = vc.sin(eta + gamma - amp_contrast).copy()
    E_TC = vc.exp(-0.5 * sigma_f * sigma_f * np.pi * np.pi * wlen_2 * Sr_2 * Sr_2).copy()
    E_SC = vc.exp(-2 * sigma_beta * sigma_beta * np.pi * np.pi * Sr_2 * (CWS - defocus)*(CWS - defocus)).copy()
    E_JN = vc.exp(-2 * np.pi * np.pi * johnson_std*johnson_std * Sr_2).copy()
    E_BF = vc.exp(-0.25 * params.B * Sr_2).copy() if not disable_B_factor else 1.0

    CTF = (2 * E_TC * E_SC * E_JN * E_BF * sinchi).copy()

    return CTF

def apply_ctf_to_rfft_buffer(buffer: vd.RFFTBuffer, ctf_params: CTFParams, pixel_size: float, defocus: float):
    @vd.shader(exec_size=lambda args: args.buff.size)
    def ctf_apply_shader(buff: vc.Buff[c64], defocus_vars: vc.Var[vc.v4]):
        ind = vc.global_invocation().x.copy()

        pos_2d = vc.new_vec2()
        pos_2d.x = ind % buffer.shape[2]
        pos_2d.y = ((ind / buffer.shape[2]) + buffer.shape[1] // 2) % buffer.shape[1]
        pos_2d.y = pos_2d.y - buffer.shape[1] // 2

        buff[ind] *= ctf_filter(
            buffer.shape[1:],
            defocus_vars,
            pos_2d,
            ctf_params,
            pixel_size
        )

    ctf_apply_shader(buffer, [defocus, 4000, 4000, 0])

def rfft2_to_fft2(rfft_result, original_shape):
    rows, cols = original_shape
    full_fft = np.zeros((rows, cols), dtype=complex)
    full_fft[:, :rfft_result.shape[1]] = rfft_result
    
    if cols % 2 == 0:
        for k in range(1, rfft_result.shape[1] - 1):
            full_fft[:, cols - k] = np.conj(np.roll(rfft_result[:, k], 0, axis=0)[::-1])
    else:
        for k in range(1, rfft_result.shape[1]):
            full_fft[:, cols - k] = np.conj(np.roll(rfft_result[:, k], 0, axis=0)[::-1])
    
    for j in range(1, cols):
        if j < rfft_result.shape[1]:
            continue
        full_fft[1:, j] = np.conj(full_fft[1:, cols - j][::-1])
        full_fft[0, j] = np.conj(full_fft[0, cols - j])
    
    return full_fft

def generate_ctf(box_size: tuple[int, int], pixel_size: float, defocus: float, ctf_params: CTFParams = None) -> np.ndarray:
    result_buffer = vd.RFFTBuffer((1, *box_size))

    ones = np.ones(shape=result_buffer.shape, dtype=np.float32)
    result_buffer.write_fourier((ones + 1j * ones).astype(np.complex64))

    apply_ctf_to_rfft_buffer(
        result_buffer,
        ctf_params if ctf_params is not None else get_ctf_params_set('krios'),
        pixel_size,
        defocus
    )

    rctf2 = result_buffer.read_fourier(0)[0].real
    return np.fft.fftshift(rfft2_to_fft2(rctf2, box_size).real)
