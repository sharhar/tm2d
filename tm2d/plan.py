import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np
import tqdm
import numbers

from typing import Union, List, Optional, Generator

from .ctf import CTFParams, get_ctf_params_set

def process_defocus_input(defocus_input: Union[float, List[float], np.ndarray, vc.Var[vc.v4]]) -> np.ndarray:
    if isinstance(defocus_input, (numbers.Number, np.number)):
        defocus_input = np.array([[defocus_input, 0.0, 0.0, 0.0]], dtype=np.float32)
    
    if isinstance(defocus_input, list):
        defocus_input = np.array(defocus_input, dtype=np.float32)

    if isinstance(defocus_input, np.ndarray):
        if defocus_input.ndim == 1:
            defocus_input = defocus_input.reshape(1, 4)

        assert defocus_input.shape[1] == 4, "Defocus vector must be a 2D array with shape (N, 4)."
    
    return defocus_input

class Template:
    def __init__(self):
        raise NotImplementedError("Template is an abstract class. Please implement it in a subclass.")

    def _make_template(self,
                      rotations: Union[vc.Var[vc.m4], np.ndarray],
                      pixel_size: float,
                      #defoci: List[Union[vc.Var[vc.v4], np.ndarray]],
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
                      #*defoci: Union[vc.Var[vc.v4], float, List[float], np.ndarray],
                      ctf_params: CTFParams = None,
                      template_count: int = 1,
                      cmd_stream: vd.CommandStream = None,
                      disable_ctf: bool = False) -> vd.RFFTBuffer:
        """
        This abstract method should return a buffer containing the sampled and filtered template in real space.   
        """

        if ctf_params is None:
            ctf_params = get_ctf_params_set('krios')
        
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
            #defoci=[process_defocus_input(d) for d in defoci],
            ctf_params=ctf_params,
            template_count=template_count,
            cmd_stream=cmd_stream,
            disable_ctf=disable_ctf
        )
    
    def get_rotation_matricies(self, rotations: np.ndarray) -> np.ndarray:
        raise NotImplementedError("get_rotation_matricies must be implemented in a subclass.")
    
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
                 ctf_params: Union[CTFParams, str] = None,
                 template_batch_size: int = 2):

        self.template_batch_size = template_batch_size

        if ctf_params is None:
            ctf_params = get_ctf_params_set('krios')

        if isinstance(ctf_params, str):
            ctf_params = get_ctf_params_set(ctf_params)

        self.template = template
        self.comparator = comparator
        self.results = results
        self.rotation = rotation
        self.pixel_size = pixel_size
        self.ctf_params = ctf_params

        self.cmd_stream = vd.CommandStream()

        prev_stream = vd.set_global_cmd_stream(self.cmd_stream)

        self.template_buffer = self.template.make_template(
            self.cmd_stream.bind_var("rot_mat"),
            pixel_size,
            #*[self.cmd_stream.bind_var(f"defocus{i}") for i in range(self.template_batch_size)],
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
            rotations: np.ndarray,
            ctf_params_dict: dict[str, np.ndarray],
            enable_progress_bar: bool = False,
            batch_size: int = 32):

        combinations_array, dynamic_fields = self.ctf_params.generate_iteration_information(ctf_params_dict)
        
        if enable_progress_bar:
            status_bar = tqdm.tqdm(total=rotations.shape[0] * combinations_array.shape[0])

        rotations_array = np.zeros(shape=(batch_size, 3), dtype=np.float32)
        index_arrays = [np.ones(shape=(batch_size,), dtype=np.int32) * -1 for _ in range(self.template_batch_size)]
        total_batch_size = self.template_batch_size * batch_size
        input_array = np.zeros(shape=(batch_size,), dtype=np.float32)

        if combinations_array.shape[0] == 1 and self.template_batch_size > 1:
            print("Warning: Only one ctf parameter combination provided, but template_batch_size is greater than 1. This will result in suboiptimal performance.")

        for i in range(0, combinations_array.shape[0], total_batch_size):
            param_count = min(total_batch_size, combinations_array.shape[0] - i)

            ctf_batch_size = int(np.ceil(param_count / self.template_batch_size))
            rotations_batch_size = int(np.floor(batch_size / ctf_batch_size))
            full_batch_size = ctf_batch_size * rotations_batch_size

            for batch_id in range(self.template_batch_size):
                start_index = i + batch_id * ctf_batch_size
                end_index = min(start_index + ctf_batch_size, combinations_array.shape[0])

                last_index = end_index - start_index
                
                for ii, field in enumerate(dynamic_fields):
                    input_array[:last_index] = combinations_array[start_index:end_index, ii]

                    self.cmd_stream.set_var(f"{field.name}_{batch_id}", np.tile(input_array[:ctf_batch_size], rotations_batch_size))

                index_arrays[batch_id][0:last_index] = np.arange(start_index, end_index)
                index_arrays[batch_id][last_index:ctf_batch_size] = -combinations_array.shape[0] * rotations.shape[0]

                for j in range(1, rotations_batch_size):
                    index_arrays[batch_id][ctf_batch_size*j:ctf_batch_size*(j+1)] = index_arrays[batch_id][:ctf_batch_size] + combinations_array.shape[0] * j

            for j in range(0, rotations.shape[0], rotations_batch_size):
                actual_batch_size = min(rotations_batch_size, rotations.shape[0] - j)

                rotations_array[:actual_batch_size * ctf_batch_size] = np.repeat(rotations[j:j + actual_batch_size, :], repeats=ctf_batch_size, axis=0)

                self.cmd_stream.set_var(
                    "rot_mat", 
                    self.template.get_rotation_matricies(rotations_array[:full_batch_size, :])
                )

                for k in range(self.template_batch_size):
                    self.cmd_stream.set_var(f"index{k}", index_arrays[k][:full_batch_size] + combinations_array.shape[0] * j)

                # submit the command stream with the current batch size
                self.cmd_stream.submit_any(full_batch_size)

                if enable_progress_bar:
                    status_bar.update(param_count * rotations_batch_size)


        if enable_progress_bar:
            status_bar.close()
    
