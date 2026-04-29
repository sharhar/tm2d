import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from ..plan import Results, ParamSet

@vd.map
def reduction_map(buf: vc.Buff[vc.c64]) -> vc.v3:
    ind = vd.reduce.mapped_io_index()
    result = vc.new_vec3_register(0, 0, 0)

    with vc.if_block(ind % buf.shape.z < buf.shape.z - 1):
        val = buf[ind].to_register()

        result.x = vc.max(val.real, val.imag)
        result.y = val.real + val.imag
        result.z = val.real * val.real + val.imag * val.imag

    return result

@vd.reduce.reduce(0, axes=[1, 2], mapping_function=reduction_map)
def calculate_sums_and_max(a: vc.Const[vc.v3], b: vc.Const[vc.v3]) -> vc.v3:
    return vc.new_vec3_register(
        vc.max(a.x, b.x),
        a.y + b.y,
        a.z + b.z
    )

class ResultsParam(Results):
    best_values_buffer: vd.Buffer
    best_mip_values_buffer: vd.Buffer
    compiled: bool
    compiled_best_values: np.ndarray
    compiled_best_mip_values: np.ndarray

    def __init__(self, batch_count: int, total_indicies: int) -> None:
        self.best_values_buffer = vd.Buffer((batch_count, total_indicies, ), vd.float32)
        self.best_mip_values_buffer = vd.Buffer((batch_count, total_indicies, ), vd.float32)
        self.compiled = False
        self.reset()

    def reset(self):
        self.best_values_buffer.write(
            (np.ones(shape=self.best_values_buffer.shape) * -1000000).astype(np.float32)
        )
        self.best_mip_values_buffer.write(
            (np.ones(shape=self.best_mip_values_buffer.shape) * -1000000).astype(np.float32)
        )

    def check_comparison(self, comparison_buffer: vd.RFFTBuffer, rotation_weights: vc.Var[vc.f32], *indicies: vc.Var[vc.i32]):

        template_count = comparison_buffer.shape[0] // self.best_values_buffer.shape[0]

        assert comparison_buffer.shape[0] % self.best_values_buffer.shape[0] == 0, \
            "The comparison buffer shape must be divisible by the best values buffer shape."

        assert len(indicies) == template_count, \
            f"The number of indicies ({len(indicies)}) must match the number of templates"

        pixel_count = comparison_buffer.shape[1] * (comparison_buffer.shape[2] - 1) * 2

        rotation_weights_type = vc.Const[vc.f32] if isinstance(rotation_weights, int) else vc.Var[vc.f32]

        @vd.shader(
            exec_size=self.best_values_buffer.shape[0],
            arg_type_annotations=[
                vc.Buff[vc.f32],  # zscore_buff
                vc.Buff[vc.f32],  # mip_buff
                vc.Buff[vc.v3],  # reduced_values_buff
                rotation_weights_type,  # rotation_weights
            ] + [vc.Var[vc.i32]] * len(indicies)
        )
        def update_best_value(
            zscore_buff: vc.Buff[vc.f32],
            mip_buff: vc.Buff[vc.f32],
            reduced_values_buff: vc.Buff[vc.v3],
            rot_weights: vc.Var[vc.f32],
            *indicies: vc.Var[vc.i32]):
            _ = rot_weights

            ind = vc.global_invocation_id().x.to_dtype(vc.i32).to_register()

            reduced_values = vc.new_vec3_register()

            for i in range(template_count):
                output_index = ind * self.best_values_buffer.shape[1] + indicies[i]
                input_index = ind * template_count + i

                reduced_values[:] = reduced_values_buff[input_index]

                with vc.if_block(indicies[i] >= 0):
                    mean_val = (reduced_values.y / float(pixel_count)).to_register()
                    var_val = vc.max(reduced_values.z / float(pixel_count) - mean_val * mean_val, 0.0).to_register()
                    z_score = ((reduced_values.x - mean_val) / vc.sqrt(var_val + 1e-6)).to_register()

                    with vc.if_block(z_score > zscore_buff[output_index]):
                        zscore_buff[output_index] = z_score
                        mip_buff[output_index] = reduced_values.x

        reduced_values_buff = calculate_sums_and_max(comparison_buffer)

        update_best_value(
            self.best_values_buffer,
            self.best_mip_values_buffer,
            reduced_values_buff,
            rotation_weights,
            *indicies
        )

    def compile_results(self, template_count: int):
        _ = template_count

        # We do an element wise max reduction on the buffer because
        # each of the devices will have only written to a portion of the buffer
        # (which is unique to each device). So this will give us the full combined result.
        all_best_values = np.array(self.best_values_buffer.read())
        all_best_mips = np.array(self.best_mip_values_buffer.read())

        # Select the device slot that produced the winning z-score, then read both
        # z-score and matching MIP from that same slot.
        max_indices = np.argmax(all_best_values, axis=0)
        gather_indices = np.expand_dims(max_indices, axis=0)
        self.compiled_best_values = np.take_along_axis(all_best_values, gather_indices, axis=0)[0]
        self.compiled_best_mip_values = np.take_along_axis(all_best_mips, gather_indices, axis=0)[0]

        self.compiled = True

    def get_zscore_list(self, params: ParamSet):
        flat_zscore = self.compiled_best_values[:, :params.get_total_count()]
        zscore_shape = params.get_tensor_shape(flat_zscore.shape[0])[:-1]
        return flat_zscore.reshape(zscore_shape)

    def get_mip_list(self, params: ParamSet):
        flat_mip = self.compiled_best_mip_values[:, :params.get_total_count()]
        mip_shape = params.get_tensor_shape(flat_mip.shape[0])[:-1]
        return flat_mip.reshape(mip_shape)

