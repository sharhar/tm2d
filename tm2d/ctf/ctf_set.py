import dataclasses
import numpy as np
import vkdispatch as vd
import itertools

from .ctf_params import CTFParams

@dataclasses.dataclass
class CTFSet:
    ctf_params: CTFParams
    combinations_array: np.ndarray
    field_names: list[str]
    lengths: list[int]

    def __init__(self,
                 ctf_params: CTFParams,
                 **values_dict):

        self.ctf_params = ctf_params

        fields = dataclasses.fields(ctf_params)

        dynamic_fields = [field for field in fields if ctf_params.__getattribute__(field.name) is None]

        dynamic_field_names = {field.name for field in dynamic_fields}
        values_dict_keys = set(values_dict.keys())
        if dynamic_field_names != values_dict_keys:
            raise ValueError(f"Dynamic field names {dynamic_field_names} do not match keys in values_dict {values_dict_keys}")

        dynamic_values = [values_dict[field.name] for field in dynamic_fields]

        for dyn_val, field in zip(dynamic_values, dynamic_fields):
            assert dyn_val.ndim == 1, f"Dynamic value for field {field.name} must be a 1D array, got {dyn_val.ndim}D array."

        combinations = list(itertools.product(*dynamic_values))

        self.combinations_array = np.array(combinations)
        self.field_names = [field.name for field in dynamic_fields]
        self.lengths = [len(values_dict[field.name]) for field in dynamic_fields]

    def get_length(self):
        return self.combinations_array.shape[0]

    def get_field_names(self):
        return self.field_names

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

def make_ctf_set(ctf_params: CTFParams, **values_dict) -> CTFSet:
    return CTFSet(ctf_params, **values_dict)