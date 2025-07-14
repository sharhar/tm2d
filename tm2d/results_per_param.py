import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

from .plan import Results, ParamSet

from typing import Optional

class ResultsParam(Results):
    best_values_buffer: vd.Buffer
    compiled: bool
    compiled_best_values: np.ndarray

    def __init__(self, batch_count: int, total_indicies: int) -> None:
        self.best_values_buffer = vd.Buffer((batch_count, total_indicies, ), vd.float32)
        self.compiled = False
        self.reset()
    
    def reset(self):
        self.best_values_buffer.write(
            (np.ones(shape=self.best_values_buffer.shape) * -1000000).astype(np.float32)
        )
    
    def check_comparison(self, comparison_buffer: vd.RFFTBuffer, *indicies: vc.Var[vc.i32]):
        template_count = comparison_buffer.shape[0] // self.best_values_buffer.shape[0]

        assert comparison_buffer.shape[0] % self.best_values_buffer.shape[0] == 0, \
            "The comparison buffer shape must be divisible by the best values buffer shape."
        
        assert len(indicies) == template_count, \
            f"The number of indicies ({len(indicies)}) must match the number of templates"
        
        @vd.map_reduce(vd.SubgroupMax, axes=[1, 2])
        def find_best_mip(buf: Buff[c64]) -> f32:
            ind = vc.mapping_index()
            result = vc.new_float()

            vc.if_statement(ind % comparison_buffer.shape[2] < comparison_buffer.shape[2] - 1)
            result[:] = vc.max(buf[ind].x, buf[ind].y)
            vc.else_statement()
            result[:] = 0.0
            vc.end()

            return result

        def update_best_value_func(
                buff: Buff[f32],
                maxes_buff: Buff[f32],
                *indicies: Var[i32]):
            ind = vc.global_invocation().x
            
            for i in range(template_count):
                output_index = ind * self.best_values_buffer.shape[1] + indicies[i]
                input_index = ind * template_count + i

                vc.if_statement(maxes_buff[input_index] > buff[output_index])
                buff[output_index] = maxes_buff[input_index]
                vc.end()

        with vc.builder_context() as builder:
            signature = vd.ShaderSignature.from_type_annotations(builder, [
                Buff[f32],  # buff
                Buff[i32],  # maxes_buff
            ] + [Var[i32]] * len(indicies))

            update_best_value_func(*signature.get_variables())

            update_best_value_shader = vd.ShaderObject(
                builder.build("update_best_value_func"), 
                signature,
                exec_count=self.best_values_buffer.shape[0]
            )

        max_values_buff = find_best_mip(comparison_buffer)

        update_best_value_shader(
            self.best_values_buffer,
            max_values_buff,
            *indicies
        )

    def compile_results(self):
        # We do an element wise max reduction on the buffer because
        # each of the devices will have only written to a portion of the buffer
        # (which is unique to each device). So this will give us the full combined result.
        self.compiled_best_values = np.max(np.array(self.best_values_buffer.read()), axis=0)

        self.compiled = True
    
    def get_mip_list(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_best_values

        # actual_best_values = self.compiled_best_values[:, :params.get_total_count()] # if true_size is None else self.compiled_best_values[:true_size]
        
        # print(params.get_total_count())
        # print(self.compiled_best_values.shape)
        # print(actual_best_values.shape)
        
        # param_values_dict = params.index_to_values(np.arange(0, actual_best_values.shape[1], 1))
        # print(param_values_dict)

        #assert actual_best_values.shape[1] == rotations.shape[0] * defocus_values.shape[0], "The number of best MIPs does not match the number of parameters"
        #assert rotations.shape[0] % IPA_count == 0, "The number of rotations must be divisible by the number of in-plane angles"

        #param_indicies = np.array(range(actual_best_values.shape[1]), dtype=np.int32)
        #defocus_indicies = param_indicies // rotations.shape[0]
        #rotation_indicies = param_indicies % rotations.shape[0]

        #param_list_result = np.zeros((actual_best_values.shape[0], actual_best_values.shape[1], 5), dtype=np.float32)
        #param_list_result[:, :, :3] = rotations[rotation_indicies]
        #param_list_result[:, :, 3] = defocus_values[defocus_indicies]
        #param_list_result[:, :, 4] = actual_best_values

        #return param_list_result.reshape(actual_best_values.shape[0], defocus_values.shape[0], rotations.shape[0] // IPA_count, IPA_count,  5)

