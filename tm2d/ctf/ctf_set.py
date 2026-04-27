import dataclasses
import numpy as np
import vkdispatch as vd

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