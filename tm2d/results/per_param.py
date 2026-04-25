import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from ..plan import Results, ParamSet

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

    def check_comparison(self, comparison_buffer: vd.RFFTBuffer, *indicies: vc.Var[vc.i32]):
        template_count = comparison_buffer.shape[0] // self.best_values_buffer.shape[0]

        assert comparison_buffer.shape[0] % self.best_values_buffer.shape[0] == 0, \
            "The comparison buffer shape must be divisible by the best values buffer shape."

        assert len(indicies) == template_count, \
            f"The number of indicies ({len(indicies)}) must match the number of templates"

        pixel_count = comparison_buffer.shape[1] * (comparison_buffer.shape[2] - 1) * 2

        @vd.reduce.map_reduce(vd.reduce.SubgroupMax, axes=[1, 2])
        def find_best_mip(buf: vc.Buff[vc.c64]) -> vc.f32:
            ind = vd.reduce.mapped_io_index()
            result = vc.new_float_register(0)

            with vc.if_block(ind % comparison_buffer.shape[2] < comparison_buffer.shape[2] - 1):
                result[:] = vc.max(buf[ind].real, buf[ind].imag)

            with vc.else_block():
                result[:] = 0.0

            return result

        @vd.reduce.map_reduce(vd.reduce.SubgroupAdd, axes=[1, 2])
        def find_sum(buf: vc.Buff[vc.c64]) -> vc.f32:
            ind = vd.reduce.mapped_io_index()
            result = vc.new_float_register(0)

            with vc.if_block(ind % comparison_buffer.shape[2] < comparison_buffer.shape[2] - 1):
                result[:] = buf[ind].real + buf[ind].imag

            with vc.else_block():
                result[:] = 0.0

            return result

        @vd.reduce.map_reduce(vd.reduce.SubgroupAdd, axes=[1, 2])
        def find_sum2(buf: vc.Buff[vc.c64]) -> vc.f32:
            ind = vd.reduce.mapped_io_index()
            result = vc.new_float_register(0)
            val_real = vc.new_float_register(0)
            val_imag = vc.new_float_register(0)

            with vc.if_block(ind % comparison_buffer.shape[2] < comparison_buffer.shape[2] - 1):
                val_real[:] = buf[ind].real
                val_imag[:] = buf[ind].imag
                result[:] = val_real * val_real + val_imag * val_imag

            with vc.else_block():
                result[:] = 0.0

            return result

        def update_best_value_func(
            zscore_buff: vc.Buff[vc.f32],
            mip_buff: vc.Buff[vc.f32],
            maxes_buff: vc.Buff[vc.f32],
            sums_buff: vc.Buff[vc.f32],
            sum2s_buff: vc.Buff[vc.f32],
            *indicies: vc.Var[vc.i32]):
            ind = vc.global_invocation_id().x.to_dtype(vc.i32).to_register()

            for i in range(template_count):
                output_index = ind * self.best_values_buffer.shape[1] + indicies[i]
                input_index = ind * template_count + i

                with vc.if_block(indicies[i] >= 0):
                    mean_val = (sums_buff[input_index] / float(pixel_count)).to_register()
                    var_val = vc.max(sum2s_buff[input_index] / float(pixel_count) - mean_val * mean_val, 0.0).to_register()
                    z_score = ((maxes_buff[input_index] - mean_val) / vc.sqrt(var_val + 1e-6)).to_register()

                    with vc.if_block(z_score > zscore_buff[output_index]):
                        zscore_buff[output_index] = z_score
                        mip_buff[output_index] = maxes_buff[input_index]

        with vc.shader_context() as ctx:
            input_args = ctx.declare_input_arguments([
                vc.Buff[vc.f32],  # zscore_buff
                vc.Buff[vc.f32],  # mip_buff
                vc.Buff[vc.f32],  # maxes_buff
                vc.Buff[vc.f32],  # sums_buff
                vc.Buff[vc.f32],  # sum2s_buff
            ] + [vc.Var[vc.i32]] * len(indicies))

            update_best_value_func(*input_args)

        update_best_value_shader = vd.make_shader_function(
            description=ctx.get_description("update_best_value_func"),
            exec_count=self.best_values_buffer.shape[0]
        )

        max_values_buff = find_best_mip(comparison_buffer)
        sum_values_buff = find_sum(comparison_buffer)
        sum2_values_buff = find_sum2(comparison_buffer)

        update_best_value_shader(
            self.best_values_buffer,
            self.best_mip_values_buffer,
            max_values_buff,
            sum_values_buff,
            sum2_values_buff,
            *indicies
        )

    def compile_results(self):
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
        if not self.compiled:
            self.compile_results()

        flat_zscore = self.compiled_best_values[:, :params.get_total_count()]
        zscore_shape = params._get_tensor_shape(flat_zscore.shape[0])[:-1]
        return flat_zscore.reshape(zscore_shape)

    def get_mip_list(self, params: ParamSet):
        if not self.compiled:
            self.compile_results()

        flat_mip = self.compiled_best_mip_values[:, :params.get_total_count()]
        mip_shape = params._get_tensor_shape(flat_mip.shape[0])[:-1]
        return flat_mip.reshape(mip_shape)

