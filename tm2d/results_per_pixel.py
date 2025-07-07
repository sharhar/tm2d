import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

from .plan import Results

# To avoid errors from the limited precision of floating point numbers, we use 2 32-bit floats
# to approximate a higher precision accumulator.
# Algorithm modified from this paper: https://link.springer.com/article/10.1007/pl00009321
def double_precision_add_f32(dsa: Const[v2], dsb: Const[f32]) -> v2:
    t2 = (dsa.y + dsb).copy()
    dsc_x = (dsa.x + t2).copy()

    return vc.new_vec2(dsc_x, t2 - (dsc_x - dsa.x))


def double_precision_add_vec2(dsa: vc.Const[vc.v2], dsb: vc.Const[vc.v2]) -> vc.v2:
    t1 = (dsa.x + dsb.x).copy()
    e = (t1 - dsa.x).copy()
    t2 = (((dsb.x - e) + (dsa.x - (t1 - e))) + dsa.y + dsb.y).copy()

    dsc_x = (t1 + t2).copy()

    return vc.new_vec2(dsc_x, t2 - (dsc_x - t1))


class ResultsPixel(Results):
    max_cross: vd.Buffer
    best_index: vd.Buffer
    sum_cross: vd.Buffer # running mean
    sum2_cross: vd.Buffer # running variance
    count_buffer: vd.Buffer # counter

    count: int

    compiled: bool

    compiled_mip: np.ndarray
    compiled_best_index_array: np.ndarray
    compiled_location_of_best_match: list[tuple[int, int]]
    compiled_index_of_params_match: list[int]
    compiled_sum_cross: np.ndarray
    compiled_sum2_cross: np.ndarray
    compiled_count: np.ndarray
    compiled_z_score: np.ndarray
    compiled_cross_mean: np.ndarray
    compiled_cross_variance: np.ndarray

    def __init__(self, shape: tuple) -> None:

        assert len(shape) == 3, "Shape must be a 3D tuple (L, H, W)."

        count = int(shape[0])
        width = int(shape[1])
        height = int(shape[2])

        self.count = count

        self.max_cross = vd.Buffer((count, width, height), vd.float32)
        self.best_index = vd.Buffer((count, width, height), vd.int32)
        self.sum_cross = vd.Buffer((count, width, height), vd.vec2)
        self.sum2_cross = vd.Buffer((count, width, height), vd.vec2)
        self.count_buffer = vd.Buffer((count, ), vd.int32)
        self.compiled = False
        self.compiled_mip = None
        self.compiled_best_index_array = None
        self.compiled_location_of_best_match = None
        self.compiled_index_of_params_match = None
        self.compiled_sum_cross = None
        self.compiled_sum2_cross = None
        self.compiled_count = None
        self.compiled_z_score = None

        self.reset()

    def reset(self):
        self.max_cross.write((np.ones(shape=self.max_cross.shape) * -1000000).astype(np.float32))
        self.best_index.write((np.ones(shape=self.best_index.shape, dtype=np.int32) * -1).astype(np.int32))
        self.sum_cross.write(np.zeros(shape=(*self.sum_cross.shape, 2), dtype=np.float32))
        self.sum2_cross.write(np.zeros(shape=(*self.sum2_cross.shape, 2), dtype=np.float32))
        self.count_buffer.write(np.zeros(shape=self.count_buffer.shape, dtype=np.int32))

        self.compiled = False

    def compile_results(self):
        max_crosses = self.max_cross.read()
        best_indicies = self.best_index.read()
        sum_crosses = self.sum_cross.read()
        sum2_crosses = self.sum2_cross.read()
        counts = self.count_buffer.read()

        final_results = [np.zeros(shape=(self.max_cross.shape[0], self.max_cross.shape[1], self.max_cross.shape[2], 2), dtype=np.float64) for _ in max_crosses]

        for i in range(len(max_crosses)):
            final_results[i][:, :, :, 0] = np.fft.ifftshift(max_crosses[i], axes=(1, 2))
            final_results[i][:, :, :, 1] = np.fft.ifftshift(best_indicies[i], axes=(1, 2))

        true_final_result = final_results[0]

        for other_result in final_results[1:]:
            true_final_result = np.where(other_result[:, :, :, 0:1] > true_final_result[:, :, :, 0:1], other_result, true_final_result)

        self.compiled_mip = true_final_result[:, :, :, 0]
        self.compiled_best_index_array = true_final_result[:, :, :, 1].astype(np.int32)

        self.compiled_sum_cross = np.fft.ifftshift(np.array(sum_crosses, dtype=np.float64).sum(axis=(0, 4)), axes=(1, 2))
        self.compiled_sum2_cross = np.fft.ifftshift(np.array(sum2_crosses, dtype=np.float64).sum(axis=(0, 4)), axes=(1, 2))
        self.compiled_count = np.array(counts).sum(axis=0)

        self.compiled_cross_mean = self.compiled_sum_cross / self.compiled_count[:, None, None] # per-pixel mean of cross-correlation
        self.compiled_cross_variance = self.compiled_sum2_cross / self.compiled_count[:, None, None] - self.compiled_cross_mean * self.compiled_cross_mean # per-pixel variance of cross-correlation
        self.compiled_z_score = (self.compiled_mip - self.compiled_cross_mean) / np.sqrt(self.compiled_cross_variance)
        
        self.compiled_location_of_best_match = []
        self.compiled_index_of_params_match = []

        for i in range(self.count):
            self.compiled_location_of_best_match.append(np.unravel_index(np.argmax(self.compiled_mip[i]), self.compiled_mip.shape[1:]))
            self.compiled_index_of_params_match.append(self.compiled_best_index_array[i][self.compiled_location_of_best_match[i]])


        self.compiled = True
    
    def get_mip(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_mip

    def get_best_index_array(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_best_index_array

    def get_location_of_best_match(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_location_of_best_match

    def get_index_of_params_match(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_index_of_params_match
    
    def get_sum_cross(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_sum_cross
    
    def get_sum2_cross(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_sum2_cross
    
    def get_count(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_count
    
    def get_z_score(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_z_score

    def get_cross_mean(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_cross_mean
    
    def get_cross_variance(self):
        if not self.compiled:
            self.compile_results()

        return self.compiled_cross_variance
    
    def check_comparison(self, comparison_buffer: vd.Buffer, *indicies: vc.Var[vc.i32]):
        assert comparison_buffer.shape[0] % self.max_cross.shape[0] == 0, "Comparison buffer size must be a multiple of the number of templates."

        template_count = comparison_buffer.shape[0] // self.max_cross.shape[0]
        template_offset = self.max_cross.shape[1] * (self.max_cross.shape[2] + 2) * template_count

        def update_max_func(max_cross: Buff[f32],
                       best_index: Buff[i32],
                       sum_cross: Buff[v2],
                       sum2_cross: Buff[v2],
                       count: Buff[i32],
                       back_buffer: Buff[f32],
                       *index_values: Var[i32]):
            ind = vc.global_invocation().x.cast_to(vc.i32).copy()

            micrograph_index = (ind / (self.max_cross.shape[1] * self.max_cross.shape[2])).copy()
            micrograph_inner_index = (ind % (self.max_cross.shape[1] * self.max_cross.shape[2])).copy()

            back_buffer_offset = (micrograph_inner_index + 2 * (micrograph_inner_index / (self.max_cross.shape[2]))).copy()
            back_buffer_offset[:] = back_buffer_offset + micrograph_index * template_offset

            mip_register = vc.new_float(0, var_name="mip_register")

            sum_cross_register = vc.new_vec2(0, var_name="sum_cross_register")
            sum2_cross_register = vc.new_vec2(0, var_name="sum2_cross_register")

            count_register = vc.new_int(0, var_name="count_register")

            best_mip_register = vc.new_float(vc.ninf_f32, var_name="best_mip_register")
            best_index_register = vc.new_int(-1, var_name="best_index_register")

            for i in range(template_count):
                if i != 0:
                    back_buffer_offset[:] = back_buffer_offset + self.max_cross.shape[1] * (self.max_cross.shape[2] + 2)

                mip_register[:] = back_buffer[back_buffer_offset]

                sum_cross_register[:] = double_precision_add_f32(sum_cross_register, mip_register)
                sum2_cross_register[:] = double_precision_add_f32(sum2_cross_register, mip_register * mip_register)

                vc.if_all(micrograph_inner_index == 0, index_values[i] >= 0)
                count_register[:] = count_register + 1
                vc.end()

                vc.if_all(mip_register > best_mip_register, index_values[i] >= 0)
                best_mip_register[:] = mip_register
                best_index_register[:] = index_values[i]
                vc.end()


            vc.if_statement(micrograph_inner_index == 0)
            count[micrograph_index] += count_register
            vc.end()

            sum_cross[ind] = double_precision_add_vec2(sum_cross[ind], sum_cross_register)
            sum2_cross[ind] = double_precision_add_vec2(sum2_cross[ind], sum2_cross_register)

            vc.if_statement(best_mip_register > max_cross[ind])
            max_cross[ind] = best_mip_register
            best_index[ind] = best_index_register
            vc.end()

        with vc.builder_context() as builder:
            signature = vd.ShaderSignature.from_type_annotations(builder, [
                Buff[f32],  # max_cross
                Buff[i32],  # best_index
                Buff[v2],   # sum_cross
                Buff[v2],   # sum2_cross
                Buff[i32],  # count
                Buff[f32],  # back_buffer
            ] + [Var[i32]] * len(indicies))

            update_max_func(*signature.get_variables())

            update_max_shader = vd.ShaderObject(
                builder.build("update_max_func"), 
                signature,
                exec_count=self.max_cross.size
            )
        
        update_max_shader(
            self.max_cross,
            self.best_index,
            self.sum_cross,
            self.sum2_cross,
            self.count_buffer,
            comparison_buffer,
            *indicies
        )

