import dataclasses
import numpy as np
import itertools
from .ctf_set import CTFSet

import vkdispatch as vd
import vkdispatch.codegen as vc

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
