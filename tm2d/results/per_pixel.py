import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from ..plan import Results

# To avoid errors from the limited precision of floating point numbers, we use 2 32-bit floats
# to approximate a higher precision accumulator.
# Algorithm modified from this paper: https://link.springer.com/article/10.1007/pl00009321
def double_precision_add_f32(dsa: vc.Const[vc.v2], dsb: vc.Const[vc.f32]) -> vc.v2:
    t2 = (dsa.y + dsb).to_register()
    dsc_x = (dsa.x + t2).to_register()

    return vc.new_vec2_register(dsc_x, t2 - (dsc_x - dsa.x))


def double_precision_add_vec2(dsa: vc.Const[vc.v2], dsb: vc.Const[vc.v2]) -> vc.v2:
    t1 = (dsa.x + dsb.x).to_register()
    e = (t1 - dsa.x).to_register()
    t2 = (((dsb.x - e) + (dsa.x - (t1 - e))) + dsa.y + dsb.y).to_register()

    dsc_x = (t1 + t2).to_register()

    return vc.new_vec2_register(dsc_x, t2 - (dsc_x - t1))


def accumulate(
        template_count: int,
        cross_correlation: vc.Buff[vc.f32],
        cross_corr_index: vc.Var[vc.i32],
        micrograph_span_rfft: int,
        rotation_weights: vc.Var[vc.f32],
        *index_values: vc.Var[vc.i32]):

    mip = vc.new_float_register(0, var_name="mip")

    sum_cross = vc.new_vec2_register(0, var_name="sum_cross")
    sum2_cross = vc.new_vec2_register(0, var_name="sum2_cross")

    best_mip = vc.new_float_register(vc.ninf_f32(), var_name="best_mip")
    best_index = vc.new_int_register(-1, var_name="best_index")

    for i in range(template_count):
        if i != 0:
            cross_corr_index[:] = cross_corr_index + micrograph_span_rfft

        with vc.if_block(index_values[i] >= 0):
            mip[:] = cross_correlation[cross_corr_index]

            sum_cross[:] = double_precision_add_f32(sum_cross, mip * rotation_weights)
            sum2_cross[:] = double_precision_add_f32(sum2_cross, mip * mip * rotation_weights)

            with vc.if_block(mip > best_mip):
                best_mip[:] = mip
                best_index[:] = index_values[i]

    return best_mip, best_index, sum_cross, sum2_cross


class ResultsPixel(Results):
    max_cross: vd.Buffer
    best_index: vd.Buffer
    sum_cross: vd.Buffer # running mean
    sum2_cross: vd.Buffer # running variance

    micrograph_count: int

    compiled: bool

    compiled_mip: np.ndarray
    compiled_best_index_array: np.ndarray
    compiled_sum_cross: np.ndarray
    compiled_sum2_cross: np.ndarray

    templates_count: int

    def __init__(self, shape: tuple) -> None:
        assert len(shape) == 3, "Shape must be a 3D tuple (L, H, W)."

        micrograph_count = int(shape[0])
        width = int(shape[1])
        height = int(shape[2])

        self.micrograph_count = micrograph_count

        self.max_cross = vd.Buffer((micrograph_count, width, height), vd.float32)
        self.best_index = vd.Buffer((micrograph_count, width, height), vd.int32)
        self.sum_cross = vd.Buffer((micrograph_count, width, height), vd.vec2)
        self.sum2_cross = vd.Buffer((micrograph_count, width, height), vd.vec2)

        self.reset()

    def reset(self):
        self.max_cross.write((np.ones(shape=self.max_cross.shape) * -1000000).astype(np.float32))
        self.best_index.write((np.ones(shape=self.best_index.shape, dtype=np.int32) * -1).astype(np.int32))
        self.sum_cross.write(np.zeros(shape=(*self.sum_cross.shape, 2), dtype=np.float32))
        self.sum2_cross.write(np.zeros(shape=(*self.sum2_cross.shape, 2), dtype=np.float32))

        self.compiled = False
        self.compiled_mip = None
        self.compiled_best_index_array = None
        self.compiled_sum_cross = None
        self.compiled_sum2_cross = None
        self.templates_count = None

    def compile_results(self, templates_count: int):
        max_crosses = self.max_cross.read()
        best_indicies = self.best_index.read()
        sum_crosses = self.sum_cross.read()
        sum2_crosses = self.sum2_cross.read()

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

        self.templates_count = templates_count

        self.compiled = True

    def get_mip(self):
        assert self.compiled, "Results must be compiled before accessing the MIP."
        return self.compiled_mip

    def get_best_index_array(self):
        assert self.compiled, "Results must be compiled before accessing the best index array."
        return self.compiled_best_index_array

    def get_sum_cross(self):
        assert self.compiled, "Results must be compiled before accessing the sum of cross-correlations."
        return self.compiled_sum_cross

    def get_sum2_cross(self):
        assert self.compiled, "Results must be compiled before accessing the sum of squared cross-correlations."
        return self.compiled_sum2_cross

    def get_templates_count(self):
        assert self.compiled, "Results must be compiled before accessing the templates count."
        return self.templates_count

    def check_comparison(self, comparison_buffer: vd.Buffer, rotation_weights: vc.Var[vc.f32], *indicies: vc.Var[vc.i32]):
        assert comparison_buffer.shape[0] % self.max_cross.shape[0] == 0, "Comparison buffer size must be a multiple of the number of templates."

        template_count = comparison_buffer.shape[0] // self.max_cross.shape[0]
        template_offset = self.max_cross.shape[1] * (self.max_cross.shape[2] + 2) * template_count

        rotation_weights_type = vc.Const[vc.f32] if isinstance(rotation_weights, int) else vc.Var[vc.f32]

        @vd.shader(
                exec_size=self.max_cross.size,
                arg_type_annotations=[
                    vc.Buff[vc.f32],  # max_cross
                    vc.Buff[vc.i32],  # best_index
                    vc.Buff[vc.v2],   # sum_cross
                    vc.Buff[vc.v2],   # sum2_cross
                    vc.Buff[vc.f32],  # cross_correlation
                    rotation_weights_type,  # rotation_weights
                ] + [vc.Var[vc.i32]] * len(indicies))
        def update_max(max_cross: vc.Buff[vc.f32],
                       best_index: vc.Buff[vc.i32],
                       sum_cross: vc.Buff[vc.v2],
                       sum2_cross: vc.Buff[vc.v2],
                       cross_correlation: vc.Buff[vc.f32],
                       rot_weights: vc.Var[vc.f32],
                       *index_values: vc.Var[vc.i32]):
            ind = vc.global_invocation_id().x.to_dtype(vc.i32).to_register()

            micrograph_span = self.max_cross.shape[1] * self.max_cross.shape[2]
            micrograph_span_rfft = self.max_cross.shape[1] * (self.max_cross.shape[2] + 2)

            micrograph_index = (ind // micrograph_span).to_register()
            micrograph_inner_index = (ind % micrograph_span).to_register()

            cross_corr_index = (micrograph_inner_index + 2 * (micrograph_inner_index // (self.max_cross.shape[2]))).to_register()
            cross_corr_index[:] = cross_corr_index + micrograph_index * template_offset

            best_mip_val, best_index_val, sum_cross_val, sum2_cross_val = accumulate(
                template_count,
                cross_correlation,
                cross_corr_index,
                micrograph_span_rfft,
                rot_weights,
                *index_values
            )

            sum_cross[ind] = double_precision_add_vec2(sum_cross[ind], sum_cross_val)
            sum2_cross[ind] = double_precision_add_vec2(sum2_cross[ind], sum2_cross_val)

            with vc.if_block(best_mip_val > max_cross[ind]):
                max_cross[ind] = best_mip_val
                best_index[ind] = best_index_val

        update_max(
            self.max_cross,
            self.best_index,
            self.sum_cross,
            self.sum2_cross,
            comparison_buffer,
            rotation_weights,
            *indicies
        )

