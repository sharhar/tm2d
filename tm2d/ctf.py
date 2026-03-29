import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import *

import dataclasses
import math

import itertools
import numpy as np

E_MASS = 0.511e6 # eV/c^2
E_COMPTON = 2.42631e-2 # A
ALPHA = 0.0072973525693
HBAR = 1.0545718001e-34
C_REL = 299792458

def do_int_check(other):
    return isinstance(other, int) or np.issubdtype(type(other), np.integer)

@dataclasses.dataclass
class CTFSet:
    combinations_array: np.ndarray
    field_names: list[str]
    lengths: list[int]

    def __init__(self):
        raise TypeError("CTFSet is not meant to be instantiated directly. Use CTFParams.make_ctf_set() instead.")

    @classmethod
    def from_compiled_params(cls, combinations_array: np.ndarray, field_names: list[str], lengths: list[int]):
        instance = cls.__new__(cls)

        instance.combinations_array = combinations_array
        instance.field_names = field_names
        instance.lengths = lengths

        return instance
    
    def get_length(self):
        return self.combinations_array.shape[0]
    
    def get_lengths_list(self):
        return self.lengths
    
    def set_ctf_batch(self,
                      index_arrays: list[np.ndarray],
                      input_array: np.ndarray,
                      cmd_graph: vd.CommandGraph,
                      ctf_batch_size: int,
                      rotations_pixels_batch_size: int,
                      rotations_pixels_count: int,
                      template_count: int,
                      ctf_index: int):
        for batch_id in range(template_count):
            start_index = ctf_index + batch_id * ctf_batch_size
            end_index = min(start_index + ctf_batch_size, self.get_length())

            last_index = end_index - start_index
            
            for ii, field in enumerate(self.field_names):
                input_array[:last_index] = self.combinations_array[start_index:end_index, ii]

                cmd_graph.set_var(f"{field}_{batch_id}", np.tile(input_array[:ctf_batch_size], rotations_pixels_batch_size))

            index_arrays[batch_id][0:last_index] = np.arange(start_index, end_index)
            index_arrays[batch_id][last_index:ctf_batch_size] = -self.get_length() * rotations_pixels_count

            for j in range(1, rotations_pixels_batch_size):
                index_arrays[batch_id][ctf_batch_size*j:ctf_batch_size*(j+1)] = index_arrays[batch_id][:ctf_batch_size] + self.get_length() * j

    def get_values_at_index(self, index: int):
        return_dict = {}
        for i, field_name in enumerate(self.field_names):
            return_dict[field_name] = self.combinations_array[index, i]
            
        return return_dict
    
@dataclasses.dataclass
class CTFParams:
    HT: float
    Cs: float
    Cc: float
    energy_spread_fwhm: float
    accel_voltage_std: float
    OL_current_std: float
    beam_semiangle: float
    johnson_std: float
    B: float
    amp_contrast: float
    zernike: float
    lpp: float
    NA: float
    f_OL: float
    lpp_rot: float
    defocus: float
    A_mag: float
    A_ang: float
    mag_00: float
    mag_01: float
    mag_10: float
    mag_11: float
    e_zern_0: float
    e_zern_1: float
    e_zern_2: float
    e_zern_3: float
    e_zern_4: float
    e_zern_5: float
    e_zern_6: float
    e_zern_7: float
    e_zern_8: float
    o_zern_0: float
    o_zern_1: float
    o_zern_2: float
    o_zern_3: float
    o_zern_4: float
    o_zern_5: float

    
    _units = {
        'HT': ('V', 1.0), # stored in V, show in kV
        'Cs': ('A', 1.0),
        'Cc': ('A', 1.0),
        'energy_spread_fwhm': ('eV', 1.0),
        'accel_voltage_std': ('dV/V', 1.0),
        'OL_current_std': ('dI/I', 1.0),
        'beam_semiangle': ('rad', 1.0),
        'johnson_std': ('A', 1.0),
        'B': ('A^2', 1.0),
        'amp_contrast': ('', 1.0),
        'zernike': ('deg', 1.0),
        'lpp': ('deg', 1.0),
        'NA': ('', 1.0),
        'f_OL': ('A', 1.0),
        'lpp_rot': ('deg', 1.0),
        'defocus': ('A', 1.0),
        'A_mag': ('A', 1.0),
        'A_ang': ('deg', 1.0),
        'mag_00': ('', 1.0),
        'mag_01': ('', 1.0),
        'mag_10': ('', 1.0),
        'mag_11': ('', 1.0),
        'e_zern_0': ('', 1.0),
        'e_zern_1': ('', 1.0),
        'e_zern_2': ('', 1.0),
        'e_zern_3': ('', 1.0),
        'e_zern_4': ('', 1.0),
        'e_zern_5': ('', 1.0),
        'e_zern_6': ('', 1.0),
        'e_zern_7': ('', 1.0),
        'e_zern_8': ('', 1.0),
        'o_zern_0': ('', 1.0),
        'o_zern_1': ('', 1.0),
        'o_zern_2': ('', 1.0),
        'o_zern_3': ('', 1.0),
        'o_zern_4': ('', 1.0),
        'o_zern_5': ('', 1.0),
    }

    def __init__(self,
                HT: float = 300e3,
                Cs: float = 2.7e7,
                Cc: float = 2.7e7,
                energy_spread_fwhm: float = 0.3,
                accel_voltage_std: float = 0.07e-6,
                OL_current_std: float = 0.1e-6,
                beam_semiangle: float = 2.5e-6,
                johnson_std: float = 0,
                B: float = 0,
                amp_contrast: float = 0.07,
                zernike: float = 0,
                lpp: float = 0,
                NA: float = 0.05,
                f_OL: float = 20e7,
                lpp_rot: float = 0,
                defocus: float = 10000,
                A_mag: float = 0,
                A_ang: float = 0,
                mag_matrix: np.ndarray = None,
                even_zernike: list[float] = None,
                odd_zernike: list[float] = None
                ):
        
        self.HT = HT
        self.Cs = Cs
        self.Cc = Cc
        self.energy_spread_fwhm = energy_spread_fwhm
        self.accel_voltage_std = accel_voltage_std
        self.OL_current_std = OL_current_std
        self.beam_semiangle = beam_semiangle
        self.johnson_std = johnson_std
        self.B = B
        self.amp_contrast = amp_contrast
        self.zernike = zernike
        self.lpp = lpp
        self.NA = NA
        self.f_OL = f_OL
        self.lpp_rot = lpp_rot
        self.defocus = defocus
        self.A_mag = A_mag
        self.A_ang = A_ang
        if mag_matrix is not None:
            assert mag_matrix.shape == (2, 2), "mag_matrix must be a 2x2 matrix."
            self.mag_00 = mag_matrix[0, 0]
            self.mag_01 = mag_matrix[0, 1]
            self.mag_10 = mag_matrix[1, 0]
            self.mag_11 = mag_matrix[1, 1]
        else:
            self.mag_00 = 1.0
            self.mag_01 = 0.0
            self.mag_10 = 0.0
            self.mag_11 = 1.0
        if even_zernike is not None:
            assert len(even_zernike) == 9, "even_zernike must have 9 elements."
            self.e_zern_0 = even_zernike[0]
            self.e_zern_1 = even_zernike[1]
            self.e_zern_2 = even_zernike[2]
            self.e_zern_3 = even_zernike[3]
            self.e_zern_4 = even_zernike[4]
            self.e_zern_5 = even_zernike[5]
            self.e_zern_6 = even_zernike[6]
            self.e_zern_7 = even_zernike[7]
            self.e_zern_8 = even_zernike[8]
        else:
            self.e_zern_0 = 0.0
            self.e_zern_1 = 0.0
            self.e_zern_2 = 0.0
            self.e_zern_3 = 0.0
            self.e_zern_4 = 0.0
            self.e_zern_5 = 0.0
            self.e_zern_6 = 0.0
            self.e_zern_7 = 0.0
            self.e_zern_8 = 0.0
        if odd_zernike is not None:
            assert len(odd_zernike) == 6, "odd_zernike must have 6 elements."
            self.o_zern_0 = odd_zernike[0]
            self.o_zern_1 = odd_zernike[1]
            self.o_zern_2 = odd_zernike[2]
            self.o_zern_3 = odd_zernike[3]
            self.o_zern_4 = odd_zernike[4]
            self.o_zern_5 = odd_zernike[5]
        else:
            self.o_zern_0 = 0.0
            self.o_zern_1 = 0.0
            self.o_zern_2 = 0.0
            self.o_zern_3 = 0.0
            self.o_zern_4 = 0.0
            self.o_zern_5 = 0.0

    @classmethod
    def from_arg_list(cls, *arg_list):
        base_obj = cls()

        fields = dataclasses.fields(base_obj)
        assert len(arg_list) == len(fields), f"Expected {len(fields)} arguments, got {len(arg_list)}"

        kwargs = {field.name: arg for field, arg in zip(fields, arg_list)}

        for field in fields:
            base_obj.__setattr__(field.name, kwargs[field.name])

        return base_obj

    @classmethod
    def like_titan(cls,
                    HT: float = 300e3,
                    Cs: float = 4.8e7,
                    Cc: float = 7.6e7,
                    energy_spread_fwhm: float = 0.9,
                    accel_voltage_std: float = 0.07e-6,
                    OL_current_std: float = 0.1e-6,
                    beam_semiangle: float = 2.5e-6,
                    johnson_std: float = 0.37,
                    B: float = 0,
                    amp_contrast: float = 0.07,
                    zernike: float = 0,
                    lpp: float = 90,
                    NA: float = 0.05,
                    f_OL: float = 20e7,
                    lpp_rot: float = 18,
                    defocus: float = 10000,
                    A_mag: float = 0,
                    A_ang: float = 0,
                    mag_matrix: np.ndarray = None,
                    even_zernike: list[float] = None,
                    odd_zernike: list[float] = None):
        return cls(
            HT=HT,
            Cs=Cs,
            Cc=Cc,
            energy_spread_fwhm=energy_spread_fwhm,
            accel_voltage_std=accel_voltage_std,
            OL_current_std=OL_current_std,
            beam_semiangle=beam_semiangle,
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
            A_ang=A_ang,
            mag_matrix=mag_matrix,
            even_zernike=even_zernike,
            odd_zernike=odd_zernike
        )
    
    @classmethod
    def like_theia(cls,
                    HT: float = 300e3,
                    Cs: float = 0,
                    Cc: float = 5.1e7,
                    energy_spread_fwhm: float = 0.3,
                    accel_voltage_std: float = 0.07e-6,
                    OL_current_std: float = 0.1e-6,
                    beam_semiangle: float = 2.5e-6,
                    johnson_std: float = 0.33,
                    B: float = 0,
                    amp_contrast: float = 0.07,
                    zernike: float = 0,
                    lpp: float = 90,
                    NA: float = 0.05,
                    f_OL: float = 14.1e7,
                    lpp_rot: float = 0,
                    defocus: float = 10000,
                    A_mag: float = 0,
                    A_ang: float = 0,
                    mag_matrix: np.ndarray = None,
                    even_zernike: list[float] = None,
                    odd_zernike: list[float] = None):
        return cls(
            HT=HT,
            Cs=Cs,
            Cc=Cc,
            energy_spread_fwhm=energy_spread_fwhm,
            accel_voltage_std=accel_voltage_std,
            OL_current_std=OL_current_std,
            beam_semiangle=beam_semiangle,
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
            A_ang=A_ang,
            mag_matrix=mag_matrix,
            even_zernike=even_zernike,
            odd_zernike=odd_zernike
        )

    @classmethod
    def like_krios(cls,
                    HT: float = 300e3,
                    Cs: float = 2.7e7,
                    Cc: float = 2.7e7,
                    energy_spread_fwhm: float = 0.3,
                    accel_voltage_std: float = 0.07e-6,
                    OL_current_std: float = 0.1e-6,
                    beam_semiangle: float = 2.5e-6,
                    johnson_std: float = 0,
                    B: float = 0,
                    amp_contrast: float = 0.07,
                    zernike: float = 0,
                    lpp: float = 0,
                    NA: float = 0.05,
                    f_OL: float = 20e7,
                    lpp_rot: float = 0,
                    defocus: float = 10000,
                    A_mag: float = 0,
                    A_ang: float = 0,
                    mag_matrix: np.ndarray = None,
                    even_zernike: list[float] = None,
                    odd_zernike: list[float] = None):
        return cls(
            HT=HT,
            Cs=Cs,
            Cc=Cc,
            energy_spread_fwhm=energy_spread_fwhm,
            accel_voltage_std=accel_voltage_std,
            OL_current_std=OL_current_std,
            beam_semiangle=beam_semiangle,
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
            A_ang=A_ang,
            mag_matrix=mag_matrix,
            even_zernike=even_zernike,
            odd_zernike=odd_zernike
        )

    def get_even_coeff(self):
        return (
            self.e_zern_0,
            self.e_zern_1,
            self.e_zern_2,
            self.e_zern_3,
            self.e_zern_4,
            self.e_zern_5,
            self.e_zern_6,
            self.e_zern_7,
            self.e_zern_8
        )
    
    def get_odd_coeff(self):
        return (
            self.o_zern_0,
            self.o_zern_1,
            self.o_zern_2,
            self.o_zern_3,
            self.o_zern_4,
            self.o_zern_5,
        )
    
    def get_mag_matrix(self):
        return (
            (self.mag_00, self.mag_01),
            (self.mag_10, self.mag_11)
        )

    def get_type_list(self, template_count: int):
        types = []

        fields = dataclasses.fields(self)

        for field in fields:
            value = self.__getattribute__(field.name)

            if value is not None:
                types.append(vc.Const[vc.f32])

        for _ in range(template_count):
            for field in fields:
                value = self.__getattribute__(field.name)

                if value is None:
                    types.append(vc.Var[vc.f32])

        return types
    
    def get_args(self, cmd_graph: vd.CommandGraph, template_count: int):
        args = []

        fields = dataclasses.fields(self)

        for field in fields:
            value = self.__getattribute__(field.name)

            if value is not None:
                args.append(value)

        for batch_id in range(template_count):
            for field in fields:
                value = self.__getattribute__(field.name)

                if value is None:
                    args.append(cmd_graph.bind_var(f"{field.name}_{batch_id}"))

        return args
    
    def assemble_params_list_from_args(self, args_list: list[vc.Var], template_count: int) -> "list[CTFParams]":
        fields = dataclasses.fields(self)

        static_count = 0
        dynamic_count = 0
        static_values = [None] * len(fields)
        dynamic_indicies = []

        for ii, field in enumerate(fields):
            value = self.__getattribute__(field.name)

            if value is not None:
                static_values[ii] = args_list[static_count]
                static_count += 1
            else:
                dynamic_count += 1
                dynamic_indicies.append(ii)
        
        assert len(args_list) == static_count + dynamic_count * template_count, f"Expected {static_count} + {dynamic_count} * {template_count} args, got {len(args_list)}"

        dynamic_args_list = args_list[static_count:]

        result = []

        for i in range(template_count):
            for ii, dyn_ind in enumerate(dynamic_indicies):
                static_values[dyn_ind] = dynamic_args_list[i * dynamic_count + ii]

            result.append(CTFParams.from_arg_list(*static_values))
            
        return result

    def make_ctf_set(self, **values_dict) -> CTFSet:
        fields = dataclasses.fields(self)

        dynamic_fields = [field for field in fields if self.__getattribute__(field.name) is None]

        dynamic_field_names = {field.name for field in dynamic_fields}
        values_dict_keys = set(values_dict.keys())
        if dynamic_field_names != values_dict_keys:
            raise ValueError(f"Dynamic field names {dynamic_field_names} do not match keys in values_dict {values_dict_keys}")
        
        dynamic_values = [values_dict[field.name] for field in dynamic_fields]

        for dyn_val, field in zip(dynamic_values, dynamic_fields):
            assert dyn_val.ndim == 1, f"Dynamic value for field {field.name} must be a 1D array, got {dyn_val.ndim}D array."

        combinations = list(itertools.product(*dynamic_values))
        combinations_array = np.array(combinations)

        return CTFSet.from_compiled_params(
            combinations_array=combinations_array,
            field_names=[field.name for field in dynamic_fields],
            lengths=[len(values_dict[field.name]) for field in dynamic_fields]
        )

    def print(self):
        fields = dataclasses.fields(self)
        max_len = max(len(f.name) for f in fields)
        print('CTF Parameters:')
        for f in fields:
            value = getattr(self, f.name)
            unit, scale = self._units.get(f.name, ('', 1.0))
            scaled = value * scale
            print(f'  {f.name:<{max_len}} : {scaled:.6g} {unit}')

    def __str__(self):
        fields = dataclasses.fields(self)
        max_len = max(len(f.name) for f in fields)
        lines = []
        for f in fields:
            value = getattr(self, f.name)
            unit, scale = self._units.get(f.name, ('', 1.0))
            scaled = value * scale
            lines.append(f'{f.name:<{max_len}} : {scaled:.6g} {unit}')
        return '\n'.join(lines)

    def to_dict_with_units(self):
        result = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            unit, scale = self._units.get(f.name, ('', 1.0))
            result[f.name] = {
                'value': value * scale,
                'unit': unit
            }
        return result

    def save_to_json(self, file_path: str):
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict_with_units(), f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str) -> "CTFParams":
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)

        kwargs = {}
        for field in dataclasses.fields(cls):
            entry = data[field.name]
            value = entry['value']
            _, scale = cls._units.get(field.name, ('', 1.0))
            kwargs[field.name] = value / scale

        return cls(**kwargs)

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
        vc.if_statement(params.get_even_coeff()[i] != 0.0)
        m, n = _even_index_to_mn(i)
        psi += params.get_even_coeff()[i] * _zernike_cart(m, n, r, th)
        vc.end()
    return psi

def phase_from_odd_zernike(params: CTFParams, Sx_eff: vc.ShaderVariable, Sy_eff: vc.ShaderVariable) -> vc.ShaderVariable:
    psi = vc.new_float_register()
    r = vc.sqrt(Sx_eff * Sx_eff + Sy_eff * Sy_eff).to_register()
    th = vc.atan2(Sy_eff, Sx_eff).to_register()
    for i in range(6):
        vc.if_statement(params.get_odd_coeff()[i] != 0.0)
        m, n = _odd_index_to_mn(i)
        psi += params.get_odd_coeff()[i] * _zernike_cart(m, n, r, th)
        vc.end()
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
    
    #CTF.real = CTF_scale
    #CTF.imag = 0

    is_dc = vc.logical_and(Sx == 0.0, Sy == 0.0)
    vc.if_statement(is_dc)
    CTF.real = 0.0
    CTF.imag = 0.0
    vc.end()
    
    return CTF

def apply_ctf_to_rfft_buffer(buffer: vd.RFFTBuffer, ctf_params: CTFParams, pixel_size: float):
    with vd.shader_context() as ctx:
        shader_args = ctx.declare_input_arguments([Buff[c64]] + ctf_params.get_type_list(1))

        buff = shader_args[0]
        shader_ctf_params = ctf_params.assemble_params_list_from_args(
            shader_args[1:], 1
        )[0]

        ind = vc.global_invocation_id().x.to_register()

        upos_2d = vc.new_uvec2_register()
        upos_2d.x = ind % buffer.shape[2]
        upos_2d.y = ((ind // buffer.shape[2]) + buffer.shape[1] // 2) % buffer.shape[1]

        pos_2d = upos_2d.to_dtype(vc.v2).to_register()
        pos_2d.y = pos_2d.y - buffer.shape[1] // 2

        ctf = ctf_filter(
            buffer.shape[1:],
            pos_2d,
            shader_ctf_params,
            pixel_size
        )

        buff[ind] = vc.mult_complex(buff[ind], ctf)

        ctf_apply_shader = ctx.get_function(
            exec_count=buffer.size,
            name="apply_ctf_to_rfft_buffer"
        )

    ctf_apply_shader(buffer, *ctf_params.get_args(None, 1))

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

def generate_ctf(box_size: tuple[int, int], pixel_size: float, ctf_params: CTFParams = None) -> np.ndarray:
    result_buffer = vd.RFFTBuffer((1, *box_size))

    ones = np.ones(shape=result_buffer.shape, dtype=np.float32)
    result_buffer.write_fourier((ones).astype(np.complex64))

    apply_ctf_to_rfft_buffer(
        result_buffer,
        ctf_params if ctf_params is not None else CTFParams(),
        pixel_size
    )

    rctf2 = result_buffer.read_fourier(0)[0]
    return np.fft.fftshift(rfft2_to_fft2(rctf2, box_size)) / 2 # division due to definition of ctf