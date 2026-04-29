import numpy as np
import dataclasses
from typing import Optional

from .ctf.ctf_set import CTFSet

@dataclasses.dataclass
class ParamSet:
    rotations: Optional[np.ndarray]
    rotations_weights: Optional[np.ndarray]
    pixel_sizes: Optional[np.ndarray]
    ctf_set: CTFSet

    def get_rotation_count(self) -> int:
        if self.rotations is None:
            return 1
        return self.rotations.shape[0]

    def get_pixel_size_count(self) -> int:
        if self.pixel_sizes is None:
            return 1
        return self.pixel_sizes.shape[0]

    def get_ctf_count(self) -> int:
        return self.ctf_set.get_length()

    def get_total_count(self) -> int:
        return self.get_rotation_count() * self.get_pixel_size_count() * self.get_ctf_count()

    def get_values_at_index(self, index: int) -> dict[str, np.ndarray]:
        ctf_index = index % self.get_ctf_count()
        pixel_size_index = (index // self.get_ctf_count()) % self.get_pixel_size_count()
        rotation_index = index // (self.get_ctf_count() * self.get_pixel_size_count())

        values = self.ctf_set.get_values_at_index(ctf_index)

        if self.pixel_sizes is not None:
            values["pixel_size"] = self.pixel_sizes[pixel_size_index]

        if self.rotations is not None:
            values["rotation"] = self.rotations[rotation_index, :]

        return values

    def get_tensor_shape(self, micrograph_count: int) -> tuple:
        ctf_lengths = self.ctf_set.get_lengths_list()
        params_count = len(ctf_lengths) + 1

        shape_prefix = None

        if self.rotations is None and self.pixel_sizes is None:
            shape_prefix = tuple()
        elif self.rotations is None:
            shape_prefix = (self.get_pixel_size_count(),)
            params_count += 1
        elif self.pixel_sizes is None:
            shape_prefix = (self.get_rotation_count(),)
            params_count += 3
        else:
            shape_prefix = (self.get_rotation_count(), self.get_pixel_size_count())
            params_count += 4

        return (micrograph_count, ) + shape_prefix + tuple(ctf_lengths) + (params_count,)

    def get_tensor_axes_names(self) -> tuple:
        if self.rotations is None and self.pixel_sizes is None:
            return ["micrograph"] + self.ctf_set.get_field_names()

        elif self.rotations is None:
            return ["micrograph", "pixel_size"] + self.ctf_set.get_field_names()
        elif self.pixel_sizes is None:
            return ["micrograph", "rotation"] + self.ctf_set.get_field_names()

        return ["micrograph", "rotation", "pixel_size"] + self.ctf_set.get_field_names()

    def get_values_tensor(self, mip_list: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        cropped_mip_list = mip_list[:, :self.get_total_count()]

        tensor_shape = self.get_tensor_shape(cropped_mip_list.shape[0])

        has_rotations = self.rotations is not None
        has_pixel_sizes = self.pixel_sizes is not None

        prefix_count = 1 + int(has_rotations) + int(has_pixel_sizes)

        initial_values_shape = (
            *tensor_shape[:prefix_count],
            int(np.prod(tensor_shape[prefix_count:-1])),
            tensor_shape[-1]
        )

        axis_names_dict = {
            field_name: i + prefix_count for i, field_name in enumerate(self.ctf_set.field_names)
        }
        values_tensor = np.zeros(shape=initial_values_shape, dtype=np.float32)

        if self.rotations is None and self.pixel_sizes is None:
            values_tensor[:, :, 0] = cropped_mip_list.reshape(values_tensor.shape[:-1])
            values_tensor[:, :, 1:] = self.ctf_set.combinations_array
        elif self.rotations is None:
            values_tensor[:, :, :, 0] = cropped_mip_list.reshape(values_tensor.shape[:-1])
            values_tensor[:, :, :, 1] = self.pixel_sizes.reshape(1, -1, 1)
            values_tensor[:, :, :, 2:] = self.ctf_set.combinations_array

            axis_names_dict["pixel_size"] = 1

        elif self.pixel_sizes is None:
            values_tensor[:, :, :, 0] = cropped_mip_list.reshape(values_tensor.shape[:-1])
            values_tensor[:, :, :, 1:4] = self.rotations.reshape(1, -1, 1, 3)
            values_tensor[:, :, :, 4:] = self.ctf_set.combinations_array

            axis_names_dict["rotation"] = 1
        else:
            values_tensor[:, :, :, :, 0] = cropped_mip_list.reshape(values_tensor.shape[:-1])
            values_tensor[:, :, :, :, 1:4] = self.rotations.reshape(1, -1, 1, 1, 3)
            values_tensor[:, :, :, :, 4] = self.pixel_sizes.reshape(1, 1, -1, 1)
            values_tensor[:, :, :, :, 5:] = self.ctf_set.combinations_array

            axis_names_dict["rotation"] = 1
            axis_names_dict["pixel_size"] = 2

        return values_tensor.reshape(tensor_shape), axis_names_dict

def make_param_set(ctf_set: CTFSet,
                    rotations: Optional[np.ndarray] = None,
                    rotations_weights: Optional[np.ndarray] = None,
                    pixel_sizes: Optional[np.ndarray] = None) -> 'ParamSet':

    if rotations is not None and rotations_weights is not None:
        if rotations.shape[0] != rotations_weights.shape[0]:
            raise ValueError("Rotations and rotations_weights must have the same number of entries.")
    elif rotations is not None:
        rotations_weights = np.ones(rotations.shape[0], dtype=np.float32)
    elif rotations_weights is not None:
        raise ValueError("rotations_weights cannot be provided without rotations.")

    return ParamSet(
        rotations=rotations,
        rotations_weights=rotations_weights,
        pixel_sizes=pixel_sizes,
        ctf_set=ctf_set
    )