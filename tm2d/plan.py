import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np
import tqdm
import numbers

from typing import Union, List, Tuple, Optional

from .ctf import CTFParams, CTFSet

class Template:
    def __init__(self):
        raise NotImplementedError("Template is an abstract class. Please implement it in a subclass.")

    def _make_template(self,
                      rotations: Union[vc.Var[vc.m4], np.ndarray],
                      pixel_size: float,
                      ctf_params: CTFParams,
                      template_count: int = 1,
                      cmd_stream: vd.CommandStream = None,
                      disable_ctf: bool = False) -> vd.RFFTBuffer:
        """
        This abstract method should return a buffer containing the sampled and filtered template in real space.   
        """
        
        raise NotImplementedError("get_template must be implemented in a subclass.")
    
    def make_template(self,
                      rotations: Union[vc.Var[vc.m4], List[int], np.ndarray],
                      pixel_size: float,
                      ctf_params: CTFParams = None,
                      template_count: int = 1,
                      cmd_stream: vd.CommandStream = None,
                      disable_ctf: bool = False) -> vd.RFFTBuffer:
        """
        This abstract method should return a buffer containing the sampled and filtered template in real space.   
        """

        if ctf_params is None:
            ctf_params = CTFParams()
        
        if isinstance(rotations, list):
            rotations = np.array(rotations, dtype=np.float32)

        if isinstance(rotations, np.ndarray):
            if rotations.ndim == 1:
                rotations = rotations.reshape(1, 3)
            
            if rotations.ndim == 2:
                if rotations.shape[1] == 4 and rotations.shape[0] == 4:
                    rotations = rotations.reshape(1, 4, 4)
                else:
                    assert rotations.shape[1] == 3 , "Rotations must be a 2D array with shape (N, 3)."

                    rotations = self.get_rotation_matricies(rotations)

        return self._make_template(
            rotations=rotations,
            pixel_size=pixel_size,
            ctf_params=ctf_params,
            template_count=template_count,
            cmd_stream=cmd_stream,
            disable_ctf=disable_ctf
        )
    
    def _get_rotation_matricies(self, rotations: np.ndarray) -> np.ndarray:
        raise NotImplementedError("get_rotation_matricies must be implemented in a subclass.")
    
    def get_rotation_matricies(self, rotations: Union[np.ndarray, Tuple[float, float, float]]) -> np.ndarray:
        if not isinstance(rotations, np.ndarray):
            rotations = np.array(rotations, dtype=np.float32).reshape(1, 3)
        
        if rotations.ndim == 1:
            rotations = rotations.reshape(1, 3)

        return self._get_rotation_matricies(rotations)
    
    def get_shape(self) -> tuple:
        """
        Return the shape of the template.
        This method should be implemented in subclasses to return the correct shape.
        """
        raise NotImplementedError("get_shape must be implemented in a subclass.")

class Comparator:
    def __init__(self):
        raise NotImplementedError("Comparator is an abstract class. Please implement it in a subclass.")
    
    def set_data(self, data: np.ndarray):
        """
        This method should set the data buffer that will be used for comparison.
        """
        raise NotImplementedError("set_data must be implemented in a subclass.")

    def compare_template(self, template: vd.RFFTBuffer, normalize: bool = True) -> vd.RFFTBuffer:
        raise NotImplementedError("compare_template must be implemented in a subclass.")

class Results:
    def __init__(self):
        raise NotImplementedError("Results is an abstract class. Please implement it in a subclass.")

    def check_comparison(self, comparison_buffer: vd.RFFTBuffer, *indicies: vc.Var[vc.i32]):
        raise NotImplementedError("check_comparison must be implemented in a subclass.")
    
    def reset(self):
        """
        Reset the results to their initial state.
        This method should be implemented in subclasses to reset any internal state.
        """
        raise NotImplementedError("reset must be implemented in a subclass.")

class ParamSet:
    rotations: Optional[np.ndarray] = None
    pixel_sizes: Optional[np.ndarray] = None
    ctf_set: CTFSet

    def __init__(self):
        raise TypeError("CTFSet is not meant to be instantiated directly. Use CTFParams.make_ctf_set() instead.")

    @classmethod
    def from_params(cls,
                    rotations: np.ndarray,
                    pixel_sizes: np.ndarray,
                    ctf_set: CTFSet) -> 'ParamSet':
        instance = cls.__new__(cls)

        instance.rotations = rotations
        instance.pixel_sizes = pixel_sizes
        instance.ctf_set = ctf_set

        return instance
    
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
    
    def _get_tensor_shape(self, micrograph_count: int) -> tuple:
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
        
    def get_values_tensor(self, mip_list: np.ndarray) -> np.ndarray:
        cropped_mip_list = mip_list[:, :self.get_total_count()]

        tensor_shape = self._get_tensor_shape(cropped_mip_list.shape[0])

        has_rotations = self.rotations is not None
        has_pixel_sizes = self.pixel_sizes is not None

        prefix_count = 1 + int(has_rotations) + int(has_pixel_sizes)

        initial_values_shape = (
            *tensor_shape[:prefix_count],
            np.prod(tensor_shape[prefix_count:-1]),
            tensor_shape[-1]
        )

        axis_names_dict = {
            field_name: i + prefix_count for i, field_name in enumerate(self.ctf_set.field_names)
        }
        values_tensor = np.zeros(shape=initial_values_shape, dtype=np.float32)

        if self.rotations is None and self.pixel_sizes is None:
            values_tensor[:, :, 0] = cropped_mip_list
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

        


class Plan:
    template: Template
    comparator: Comparator
    results: Results
    rotation: Optional[np.ndarray]
    pixel_size: Optional[float]
    ctf_params: CTFParams
    template_batch_size: int

    def __init__(self,
                 template: Template,
                 comparator: Comparator,
                 results: Results,
                 rotation: Optional[np.ndarray] = None,
                 pixel_size: Optional[float] = None,
                 ctf_params: Optional[CTFParams] = None,
                 template_batch_size: int = 2):

        self.template_batch_size = template_batch_size

        if ctf_params is None:
            ctf_params = CTFParams()

        self.template = template
        self.comparator = comparator
        self.results = results
        self.rotation = rotation
        self.pixel_size = pixel_size
        self.ctf_params = ctf_params

        self.cmd_stream = vd.CommandStream()

        prev_stream = vd.set_global_cmd_stream(self.cmd_stream)

        self.template_buffer = self.template.make_template(
            self.cmd_stream.bind_var("rotation_matrix") if self.rotation is None else self.template.get_rotation_matricies(self.rotation),
            self.cmd_stream.bind_var("pixel_size") if self.pixel_size is None else self.pixel_size,
            template_count=self.template_batch_size,
            cmd_stream=self.cmd_stream,
            ctf_params=ctf_params
        )

        self.comparison_buffer = self.comparator.compare_template(self.template_buffer)

        self.results.check_comparison(
            self.comparison_buffer,
            *[self.cmd_stream.bind_var(f"index{i}") for i in range(self.template_batch_size)]
        )

        vd.set_global_cmd_stream(prev_stream)
    
    def set_data(self, data: np.ndarray):
        self.comparator.set_data(data)

    def reset(self):
        self.results.reset()

    def run(self,
            #ctf_set: CTFSet,
            #rotations: Optional[np.ndarray] = None,
            #pixel_sizes: Optional[Union[np.ndarray]] = None,
            params: ParamSet,
            enable_progress_bar: bool = False,
            batch_size: int = 32):
        # if rotations is None:
        #     assert self.rotation is not None, "If rotations are not provided, the rotation attribute must be set."
        
        # if pixel_sizes is None:
        #     assert self.pixel_size is not None, "If pixel sizes are not provided, the pixel_size attribute must be set."

        #rotation_count = 1 if rotations is None else rotations.shape[0]
        #pixel_size_count = 1 if pixel_sizes is None else pixel_sizes.shape[0]

        rotations_array = np.zeros(shape=(batch_size, 3), dtype=np.float32)
        pixel_sizes_array = np.zeros(shape=(batch_size,), dtype=np.float32)
        ctf_index_arrays = [np.ones(shape=(batch_size,), dtype=np.int32) * -1 for _ in range(self.template_batch_size)]
        index_arrays = [np.ones(shape=(batch_size,), dtype=np.int32) * -1 for _ in range(self.template_batch_size)]
        
        max_batch_size = self.template_batch_size * batch_size
        input_array = np.zeros(shape=(batch_size,), dtype=np.float32)

        if params.get_ctf_count() == 1 and self.template_batch_size > 1:
            print("Warning: Only one ctf parameter combination provided, but template_batch_size is greater than 1. This will result in suboiptimal performance.")

        if enable_progress_bar:
            status_bar = tqdm.tqdm(total=params.get_total_count()) #rotation_count * pixel_size_count * ctf_set.get_length())
        
        for i in range(0, params.get_ctf_count(), max_batch_size):

            ctf_count = min(max_batch_size, params.get_ctf_count() - i)
            ctf_batch_size = int(np.ceil(ctf_count / self.template_batch_size))
            rotations_pixels_batch_size = int(np.floor(batch_size / ctf_batch_size))

            rotations_batch_size = rotations_pixels_batch_size
            pixel_batch_size = 1

            if params.get_rotation_count() == 1:
                rotations_batch_size = 1
                pixel_batch_size = rotations_pixels_batch_size

            full_batch_size = ctf_batch_size * rotations_batch_size * pixel_batch_size

            params.ctf_set.set_ctf_batch(
                ctf_index_arrays,
                input_array,
                self.cmd_stream,
                ctf_batch_size,
                rotations_pixels_batch_size,
                params.get_rotation_count() * params.get_pixel_size_count(),
                self.template_batch_size,
                i
            )

            for pixel_size_index in range(0, params.get_pixel_size_count(), pixel_batch_size):
                actual_pixel_batch_size = min(pixel_batch_size, params.get_pixel_size_count() - pixel_size_index)

                if params.pixel_sizes is not None:
                    pixel_sizes_array[:actual_pixel_batch_size * ctf_batch_size] = np.repeat(
                        params.pixel_sizes[pixel_size_index:pixel_size_index + actual_pixel_batch_size],
                        repeats=ctf_batch_size,
                        axis=0
                    )

                    pixel_batch_width = pixel_batch_size * ctf_batch_size

                    for k in range(self.template_batch_size):
                        index_arrays[k][:full_batch_size] = ctf_index_arrays[k][:full_batch_size] + params.get_ctf_count() * pixel_size_index

                        for rot_ind in range(rotations_batch_size):
                            index_arrays[k][pixel_batch_width*rot_ind + actual_pixel_batch_size * ctf_batch_size:pixel_batch_width*(rot_ind+1)] = -params.get_total_count()

                    pixel_sizes_array[:full_batch_size] = np.tile(
                        pixel_sizes_array[:pixel_batch_size * ctf_batch_size],
                        rotations_batch_size
                    )

                    self.cmd_stream.set_var("pixel_size", pixel_sizes_array[:full_batch_size])
                elif params.get_pixel_size_count() == 1:
                    for k in range(self.template_batch_size):
                        index_arrays[k][:full_batch_size] = ctf_index_arrays[k][:full_batch_size]
                else:
                    raise ValueError("Something went wrong.")

                for rotation_index in range(0, params.get_rotation_count(), rotations_batch_size):
                    actual_rotation_batch_size = min(rotations_batch_size, params.get_rotation_count() - rotation_index)

                    if params.rotations is not None:
                        rotations_array[:actual_rotation_batch_size * ctf_batch_size * pixel_batch_size] = np.repeat(
                            params.rotations[rotation_index:rotation_index + actual_rotation_batch_size, :],
                            repeats=ctf_batch_size * pixel_batch_size,
                            axis=0
                        )

                        self.cmd_stream.set_var(
                            "rotation_matrix", 
                            self.template.get_rotation_matricies(rotations_array[:full_batch_size, :])
                        )
                    
                    rotation_offset = params.get_ctf_count() * params.get_pixel_size_count() * rotation_index

                    for k in range(self.template_batch_size):
                        self.cmd_stream.set_var(
                            f"index{k}",
                            index_arrays[k][:full_batch_size] + rotation_offset
                        )

                    # submit the command stream with the current batch size
                    self.cmd_stream.submit_any(full_batch_size)

                    if enable_progress_bar:
                        status_bar.update(ctf_count * actual_rotation_batch_size)

        if enable_progress_bar:
            status_bar.close()
    
    def make_param_set(self,
                        rotations: Optional[np.ndarray] = None,
                        pixel_sizes: Optional[np.ndarray] = None,
                        **kwargs) -> ParamSet:
        if rotations is None:
            assert self.rotation is not None, "If rotations are not provided, the rotation attribute must be set."
        else:
            assert self.rotation is None, "If rotations are provided, the rotation attribute must not be set."
        
        if pixel_sizes is None:
            assert self.pixel_size is not None, "If pixel sizes are not provided, the pixel_size attribute must be set."
        else:
            assert self.pixel_size is None, "If pixel sizes are provided, the pixel_size attribute must not be set."

        """
        Create a ParamSet with the given rotations, pixel sizes, and CTF parameters.
        """
        return ParamSet.from_params(
            rotations=rotations,
            pixel_sizes=pixel_sizes,
            ctf_set=self.ctf_params.make_ctf_set(**kwargs)
        )
