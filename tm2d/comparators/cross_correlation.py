import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import Buff, f32, c64, v2, Const

import numpy as np
import os

from ..plan import Comparator

@vd.map
def crop_mapping(in_buff: Buff[f32], sum_buffer: Buff[c64], micrograph_shape: Const[vc.iv2]):
    read_op = vd.fft.read_op()

    ind = read_op.io_index.to_dtype(vc.i32)

    read_index_3d = vc.new_ivec3_register()
    read_index_3d.x = ind // (micrograph_shape.x * (micrograph_shape.y + 2))
    read_index_3d.y = (ind // (micrograph_shape.y + 2)) % micrograph_shape.x
    read_index_3d.z = ind % (micrograph_shape.y + 2)

    read_index_3d.y -= micrograph_shape.x // 2
    read_index_3d.z -= micrograph_shape.y // 2

    read_index_3d.y += in_buff.shape.y // 2
    read_index_3d.z += in_buff.shape.z - 1

    with vc.if_block(vc.any(
            read_index_3d.y < 0,
            read_index_3d.y >= in_buff.shape.y,
            read_index_3d.z < 0,
            read_index_3d.z >= (in_buff.shape.z - 1) * 2
        )):
        read_op.register.real = 0.0
        read_op.register.imag = 0.0
    with vc.else_block():
        span_x = in_buff.shape.y * in_buff.shape.z * 2
        span_y = in_buff.shape.z * 2

        read_index_flat = (read_index_3d.x * span_x + read_index_3d.y * span_y + read_index_3d.z).to_register()

        read_op.register[:] = sum_buffer[read_index_3d.x] / (in_buff.shape.y * (in_buff.shape.z - 1) * 2)
        read_op.register.imag = vc.sqrt(read_op.register.imag - read_op.register.real * read_op.register.real)

        read_op.register.real = (in_buff[read_index_flat] - read_op.register.real) / read_op.register.imag
        read_op.register.imag = 0

@vd.shader(exec_size=lambda args: args.buf.size)
def fill_buffer(buf: Buff[c64], val: Const[c64] = 0):
    buf[vc.global_invocation_id().x] = val

@vd.reduce.map_reduce(vd.reduce.SubgroupAdd, axes=[1, 2])
def calc_sums(wave: Buff[c64]) -> v2:
    ind = vd.reduce.mapped_io_index()

    result = vc.new_vec2_register()

    with vc.if_block(ind % wave.shape.z < wave.shape.z - 1):
        result.x = wave[ind].real + wave[ind].imag
        result.y = wave[ind].real * wave[ind].real + wave[ind].imag * wave[ind].imag

    with vc.else_block():
        result.x = 0.0
        result.y = 0.0

    return result


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

class ComparatorCrossCorrelation(Comparator):
    def __init__(self, shape: tuple, template_shape: tuple):
        assert len(shape) == 3, "Shape must be a 3D tuple (N, H, W) but got: {}".format(shape)

        self.micrographs_buffer = vd.RFFTBuffer(shape)

    def set_data(self, data: np.ndarray):
        if data.ndim == 2:
            data = data.reshape((1, *data.shape))

        assert data.ndim == 3, "Micrographs must be a 3D array (N, H, W)."
        data = np.fft.rfft2(data).astype(np.complex64)
        self.micrographs_buffer.write_fourier(data)

    def _make_y_convolved_signal(self, template_buffer: vd.RFFTBuffer) -> vd.RFFTBuffer:
        correlation_shape = (
            self.micrographs_buffer.real_shape[0] * template_buffer.shape[0],
            self.micrographs_buffer.real_shape[1],
            self.micrographs_buffer.real_shape[2]
        )

        correlation_signal = vd.RFFTBuffer(correlation_shape)

        template_sum = calc_sums(template_buffer)

        vd.fft.fft(
            correlation_signal,
            template_buffer,
            template_sum,
            (self.micrographs_buffer.real_shape[1], self.micrographs_buffer.real_shape[2]),
            buffer_shape=(
                template_buffer.shape[0],
                self.micrographs_buffer.real_shape[1],
                self.micrographs_buffer.real_shape[2]
            ),
            input_map=crop_mapping,
            r2c=True
        )

        kernel_offset = int(
            self.micrographs_buffer.shape[1] * self.micrographs_buffer.shape[2]
        )

        @vd.map
        def conv_input_map(input_buffer: vc.Buffer[c64]):
            with vc.if_block(vc.all(
                vd.fft.read_op().fft_index >= self.micrographs_buffer.shape[1] // 2 - template_buffer.shape[1]// 2,
                vd.fft.read_op().fft_index < self.micrographs_buffer.shape[1] // 2 + template_buffer.shape[1] // 2)):
                vd.fft.read_op().read_from_buffer(input_buffer)
            with vc.else_block():
                vd.fft.read_op().register.real = 0.0
                vd.fft.read_op().register.imag = 0.0

        @vd.map
        def convolution_map(kernel_buffer: vc.Buffer[c64]):
            img_val = vd.fft.read_op().register
            read_register = vc.new_complex64_register()

            read_register[:] = kernel_buffer[vd.fft.read_op().io_index % kernel_offset + kernel_offset * vd.fft.mapped_kernel_index()]
            img_val[:] = vc.mult_complex(read_register, img_val.conjugate())

        @vd.map
        def output_map_func(output_buffer: vc.Buffer[c64]):
            out_reg = vd.fft.write_op().register
            output_buffer[vd.fft.write_op().io_index + kernel_offset * template_buffer.shape[0] * vd.fft.mapped_kernel_index()] = out_reg

        vd.fft.convolve(
            correlation_signal,
            correlation_signal,
            self.micrographs_buffer,
            kernel_num=self.micrographs_buffer.shape[0],
            axis=1,
            buffer_shape=(
                template_buffer.shape[0],
                self.micrographs_buffer.shape[1],
                self.micrographs_buffer.shape[2]
            ),
            input_map=conv_input_map,
            kernel_map=convolution_map,
            output_map=output_map_func
        )

        return correlation_signal

    def compare_template(self, template_buffer: vd.RFFTBuffer, normalize: bool = True, output_radius: int = None) -> vd.RFFTBuffer:
        assert normalize, "Normalization is mandatory for cross-correlation."

        correlation_shape = (
            self.micrographs_buffer.real_shape[0] * template_buffer.shape[0],
            self.micrographs_buffer.real_shape[1],
            self.micrographs_buffer.real_shape[2]
        )

        correlation_signal = self._make_y_convolved_signal(template_buffer)

        if output_radius is None:
            vd.fft.irfft(correlation_signal)
            return correlation_signal

        if os.environ.get("TM2D_OUTPUT_RADIUS_MODE") != "local_rows_fft":
            vd.fft.irfft(correlation_signal)
            return correlation_signal

        local_width = output_radius * 2
        local_correlation_signal = vd.RFFTBuffer((
            correlation_shape[0],
            local_width,
            correlation_shape[2],
        ))

        @vd.map
        def local_irfft_input(input_buffer: vc.Buffer[c64]):
            row_spectrum_width = correlation_signal.shape[2]
            local_plane_size = local_width * row_spectrum_width

            io_index = vd.fft.read_op().io_index.to_register()
            conjugate = vc.new_int_register(0)

            with vc.if_block(vd.fft.read_op().fft_index >= (vd.fft.read_op().fft_size // 2) + 1):
                io_index[:] = vd.fft.read_op().r2c_inverse_offset - io_index
                conjugate[:] = 1

            image_index = (io_index // local_plane_size).to_register()
            local_index = (io_index % local_plane_size).to_register()
            local_y = (local_index // row_spectrum_width).to_register()
            kx = (local_index % row_spectrum_width).to_register()
            signed_y = (
                (local_y.to_dtype(vc.i32) + output_radius) % local_width - output_radius
            ).to_register()
            spatial_y = vc.mod(signed_y, correlation_signal.real_shape[1]).to_dtype(vc.i32).to_register()

            source_index = io_index.to_dtype(vc.i32).to_register()
            source_index += image_index.to_dtype(vc.i32) * (
                correlation_signal.shape[1] - local_width
            ) * correlation_signal.shape[2]
            source_index += (
                spatial_y - local_y.to_dtype(vc.i32)
            ) * correlation_signal.shape[2]

            vd.fft.read_op().register[:] = input_buffer[source_index]
            with vc.if_block(conjugate == 1):
                vd.fft.read_op().register.imag = -vd.fft.read_op().register.imag

        vd.fft.fft(
            local_correlation_signal,
            correlation_signal,
            buffer_shape=local_correlation_signal.real_shape,
            input_map=local_irfft_input,
            inverse=True,
            r2c=True
        )

        return local_correlation_signal

    def compare_template_fused_results(
            self,
            template_buffer: vd.RFFTBuffer,
            results,
            rotation_weights: vc.Var[vc.f32],
            *indicies: vc.Var[vc.i32],
            output_radius: int = None):
        assert output_radius is not None, "Fused constrained comparison requires output_radius."
        assert results.output_radius == output_radius, "ResultsPixel output_radius must match comparator output_radius."

        correlation_signal = self._make_y_convolved_signal(template_buffer)

        rotation_weights_type = vc.Const[vc.f32] if isinstance(rotation_weights, int) else vc.Var[vc.f32]
        local_width = output_radius * 2
        template_count = template_buffer.shape[0]
        spectrum_row_width = correlation_signal.shape[2]
        spectrum_plane_size = correlation_signal.shape[1] * correlation_signal.shape[2]
        spectrum_micrograph_stride = template_count * spectrum_plane_size
        norm = float(correlation_signal.real_shape[2])

        if os.environ.get("TM2D_OUTPUT_RADIUS_MODE") == "fused_local_rows_fft_output":
            if template_count != 1:
                comparison_buffer = self.compare_template(
                    template_buffer,
                    output_radius=output_radius
                )
                results.check_comparison(
                    comparison_buffer,
                    rotation_weights,
                    *indicies
                )
                return

            full_height = correlation_signal.real_shape[1]
            full_width = correlation_signal.real_shape[2]
            row_spectrum_width = correlation_signal.shape[2]
            local_packed_row_width = row_spectrum_width * 2
            local_plane_size = local_width * row_spectrum_width

            @vd.map
            def local_rows_irfft_input(input_buffer: vc.Buff[vc.c64]):
                io_index = vd.fft.read_op().io_index.to_register()
                conjugate = vc.new_int_register(0)

                with vc.if_block(vd.fft.read_op().fft_index >= (vd.fft.read_op().fft_size // 2) + 1):
                    io_index[:] = vd.fft.read_op().r2c_inverse_offset - io_index
                    conjugate[:] = 1

                image_index = (io_index // local_plane_size).to_register()
                local_index = (io_index % local_plane_size).to_register()
                local_y = (local_index // row_spectrum_width).to_register()
                kx = (local_index % row_spectrum_width).to_register()
                signed_y = (
                    (local_y.to_dtype(vc.i32) + output_radius) % local_width - output_radius
                ).to_register()
                spatial_y = vc.mod(signed_y, full_height).to_dtype(vc.i32).to_register()

                source_index = (
                    image_index.to_dtype(vc.i32) * full_height * row_spectrum_width
                    + spatial_y * row_spectrum_width
                    + kx.to_dtype(vc.i32)
                ).to_register()

                vd.fft.read_op().register[:] = input_buffer[source_index]
                with vc.if_block(conjugate == 1):
                    vd.fft.read_op().register.imag = -vd.fft.read_op().register.imag

            @vd.map
            def constrained_local_rows_output(max_cross: vc.Buff[vc.f32],
                                              best_index: vc.Buff[vc.i32],
                                              sum_cross: vc.Buff[vc.v2],
                                              sum2_cross: vc.Buff[vc.v2],
                                              rot_weights: rotation_weights_type,
                                              index_value: vc.Var[vc.i32]):
                write_op = vd.fft.write_op()
                io_index = write_op.io_index.to_dtype(vc.i32).to_register()
                spatial_x = write_op.fft_index.to_dtype(vc.i32).to_register()
                row_index = (io_index // local_packed_row_width).to_register()
                local_y = (row_index % local_width).to_dtype(vc.i32).to_register()
                image_i = (row_index // local_width).to_dtype(vc.i32).to_register()

                with vc.if_block(vc.all(
                    index_value >= 0,
                    vc.any(spatial_x < output_radius, spatial_x >= full_width - output_radius),
                )):
                    signed_x = spatial_x.to_register()
                    with vc.if_block(signed_x >= full_width - output_radius):
                        signed_x -= full_width
                    local_x = vc.mod(signed_x, local_width).to_dtype(vc.i32).to_register()

                    result_index = (
                        image_i * local_width * local_width
                        + local_y * local_width
                        + local_x
                    ).to_register()

                    mip = write_op.register.real.to_register()
                    sum_cross[result_index] = double_precision_add_f32(
                        sum_cross[result_index],
                        mip * rot_weights
                    )
                    sum2_cross[result_index] = double_precision_add_f32(
                        sum2_cross[result_index],
                        mip * mip * rot_weights
                    )

                    with vc.if_block(mip > max_cross[result_index]):
                        max_cross[result_index] = mip
                        best_index[result_index] = index_value

            vd.fft.fft(
                results.max_cross,
                results.best_index,
                results.sum_cross,
                results.sum2_cross,
                rotation_weights,
                indicies[0],
                correlation_signal,
                buffer_shape=(
                    correlation_signal.real_shape[0],
                    local_width,
                    full_width,
                ),
                inverse=True,
                normalize_inverse=True,
                r2c=True,
                input_map=local_rows_irfft_input,
                output_map=constrained_local_rows_output,
            )
            return

        if os.environ.get("TM2D_OUTPUT_RADIUS_MODE") == "fused_local_fft_output":
            if template_count != 1:
                comparison_buffer = self.compare_template(
                    template_buffer,
                    output_radius=output_radius
                )
                results.check_comparison(
                    comparison_buffer,
                    rotation_weights,
                    *indicies
                )
                return

            packed_row_width = correlation_signal.shape[2] * 2
            full_height = correlation_signal.real_shape[1]
            full_width = correlation_signal.real_shape[2]

            @vd.map
            def constrained_irfft_output(max_cross: vc.Buff[vc.f32],
                                         best_index: vc.Buff[vc.i32],
                                         sum_cross: vc.Buff[vc.v2],
                                         sum2_cross: vc.Buff[vc.v2],
                                         rot_weights: rotation_weights_type,
                                         index_value: vc.Var[vc.i32]):
                write_op = vd.fft.write_op()
                io_index = write_op.io_index.to_dtype(vc.i32).to_register()
                spatial_x = write_op.fft_index.to_dtype(vc.i32).to_register()
                row_index = (io_index // packed_row_width).to_register()
                spatial_y = (row_index % full_height).to_dtype(vc.i32).to_register()
                image_i = (row_index // full_height).to_dtype(vc.i32).to_register()

                with vc.if_block(vc.all(
                    index_value >= 0,
                    vc.any(spatial_y < output_radius, spatial_y >= full_height - output_radius),
                    vc.any(spatial_x < output_radius, spatial_x >= full_width - output_radius),
                )):
                    signed_y = spatial_y.to_register()
                    signed_x = spatial_x.to_register()

                    with vc.if_block(signed_y >= full_height - output_radius):
                        signed_y -= full_height
                    with vc.if_block(signed_x >= full_width - output_radius):
                        signed_x -= full_width

                    local_y = vc.mod(signed_y, local_width).to_dtype(vc.i32).to_register()
                    local_x = vc.mod(signed_x, local_width).to_dtype(vc.i32).to_register()

                    result_index = (
                        image_i * local_width * local_width
                        + local_y * local_width
                        + local_x
                    ).to_register()

                    mip = write_op.register.real.to_register()
                    sum_cross[result_index] = double_precision_add_f32(
                        sum_cross[result_index],
                        mip * rot_weights
                    )
                    sum2_cross[result_index] = double_precision_add_f32(
                        sum2_cross[result_index],
                        mip * mip * rot_weights
                    )

                    with vc.if_block(mip > max_cross[result_index]):
                        max_cross[result_index] = mip
                        best_index[result_index] = index_value

            vd.fft.fft(
                results.max_cross,
                results.best_index,
                results.sum_cross,
                results.sum2_cross,
                rotation_weights,
                indicies[0],
                correlation_signal,
                buffer_shape=correlation_signal.real_shape,
                inverse=True,
                normalize_inverse=True,
                r2c=True,
                output_map=constrained_irfft_output,
                input_type=correlation_signal.var_type,
            )
            return

        if os.environ.get("TM2D_OUTPUT_RADIUS_MODE") in (
            "fused_local_dft_rows",
            "fused_local_dft_rows_recur",
            "fused_local_dft_rows_table",
        ):
            output_radius_mode = os.environ.get("TM2D_OUTPUT_RADIUS_MODE")
            use_recurrence = output_radius_mode == "fused_local_dft_rows_recur"
            use_table = output_radius_mode == "fused_local_dft_rows_table"
            dft_weights = None
            if use_table:
                signed_x = (np.arange(local_width, dtype=np.int32) + output_radius) % local_width - output_radius
                spatial_x = np.mod(signed_x, correlation_signal.real_shape[2]).astype(np.float32)
                kx = np.arange(spectrum_row_width, dtype=np.float32)
                phase = 2.0 * np.pi * spatial_x[:, None] * kx[None, :] / float(correlation_signal.real_shape[2])
                weights = (np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)
                dft_weights = vd.Buffer(weights.shape, vd.complex64)
                dft_weights.write(weights)

            @vd.shader(
                    exec_size=results.max_cross.shape[0] * local_width,
                    arg_type_annotations=[
                        vc.Buff[vc.f32],  # max_cross
                        vc.Buff[vc.i32],  # best_index
                        vc.Buff[vc.v2],   # sum_cross
                        vc.Buff[vc.v2],   # sum2_cross
                        vc.Buff[vc.c64],  # y-real/x-fourier correlation
                        vc.Buff[vc.c64],  # local DFT weights
                        rotation_weights_type,
                    ] + [vc.Var[vc.i32]] * len(indicies))
            def fused_row_update(max_cross: vc.Buff[vc.f32],
                                 best_index: vc.Buff[vc.i32],
                                 sum_cross: vc.Buff[vc.v2],
                                 sum2_cross: vc.Buff[vc.v2],
                                 spectrum: vc.Buff[vc.c64],
                                 weights_buffer: vc.Buff[vc.c64],
                                 rot_weights: vc.Var[vc.f32],
                                 *index_values: vc.Var[vc.i32]):
                ind = vc.global_invocation_id().x
                image_i = (ind // local_width).to_dtype(vc.i32).to_register()
                local_y = (ind % local_width).to_dtype(vc.i32).to_register()

                signed_y = ((local_y + output_radius) % local_width - output_radius).to_register()
                spatial_y = vc.mod(signed_y, correlation_signal.real_shape[1]).to_dtype(vc.i32).to_register()

                local_x = vc.new_int_register(0)
                with vc.while_block(local_x < local_width):
                    signed_x = ((local_x + output_radius) % local_width - output_radius).to_register()
                    spatial_x = vc.mod(signed_x, correlation_signal.real_shape[2]).to_dtype(vc.i32).to_register()
                    result_index = (
                        image_i * local_width * local_width
                        + local_y * local_width
                        + local_x
                    ).to_register()

                    best_mip = vc.new_float_register(vc.ninf_f32(), var_name="best_mip_row")
                    best_index_val = vc.new_int_register(-1, var_name="best_index_val_row")
                    sum_cross_val = vc.new_vec2_register(0, var_name="sum_cross_val_row")
                    sum2_cross_val = vc.new_vec2_register(0, var_name="sum2_cross_val_row")

                    phase_step = (
                        2.0 * np.pi * spatial_x.to_dtype(vc.f32)
                        / float(correlation_signal.real_shape[2])
                    ).to_register()
                    cos_step = vc.cos(phase_step).to_register()
                    sin_step = vc.sin(phase_step).to_register()

                    for template_i in range(template_count):
                        with vc.if_block(index_values[template_i] >= 0):
                            accum = vc.new_float_register(0.0)
                            cos_phase = vc.new_float_register(1.0)
                            sin_phase = vc.new_float_register(0.0)
                            kx = vc.new_int_register(0)

                            spectrum_index = (
                                image_i * spectrum_micrograph_stride
                                + template_i * spectrum_plane_size
                                + spatial_y * spectrum_row_width
                            ).to_register()
                            weights_index = (local_x * spectrum_row_width).to_register()

                            with vc.while_block(kx < spectrum_row_width):
                                spectrum_val = spectrum[spectrum_index + kx]
                                if use_table:
                                    weight_val = weights_buffer[weights_index + kx]
                                    term = (
                                        spectrum_val.real * weight_val.real
                                        - spectrum_val.imag * weight_val.imag
                                    ).to_register()
                                elif use_recurrence:
                                    term = (
                                        spectrum_val.real * cos_phase
                                        - spectrum_val.imag * sin_phase
                                    ).to_register()
                                else:
                                    phase = (phase_step * kx.to_dtype(vc.f32)).to_register()
                                    term = (
                                        spectrum_val.real * vc.cos(phase)
                                        - spectrum_val.imag * vc.sin(phase)
                                    ).to_register()

                                with vc.if_block(vc.all(
                                    kx > 0,
                                    kx < correlation_signal.real_shape[2] // 2,
                                )):
                                    term *= 2.0

                                accum += term

                                if use_recurrence:
                                    next_cos_phase = (cos_phase * cos_step - sin_phase * sin_step).to_register()
                                    sin_phase[:] = sin_phase * cos_step + cos_phase * sin_step
                                    cos_phase[:] = next_cos_phase

                                kx += 1

                            mip = (accum / norm).to_register()

                            sum_cross_val[:] = double_precision_add_f32(sum_cross_val, mip * rot_weights)
                            sum2_cross_val[:] = double_precision_add_f32(sum2_cross_val, mip * mip * rot_weights)

                            with vc.if_block(mip > best_mip):
                                best_mip[:] = mip
                                best_index_val[:] = index_values[template_i]

                    sum_cross[result_index] = double_precision_add_vec2(sum_cross[result_index], sum_cross_val)
                    sum2_cross[result_index] = double_precision_add_vec2(sum2_cross[result_index], sum2_cross_val)

                    with vc.if_block(best_mip > max_cross[result_index]):
                        max_cross[result_index] = best_mip
                        best_index[result_index] = best_index_val

                    local_x += 1

            fused_row_update(
                results.max_cross,
                results.best_index,
                results.sum_cross,
                results.sum2_cross,
                correlation_signal,
                dft_weights if dft_weights is not None else correlation_signal,
                rotation_weights,
                *indicies
            )
            return

        @vd.shader(
                exec_size=results.max_cross.size,
                arg_type_annotations=[
                    vc.Buff[vc.f32],  # max_cross
                    vc.Buff[vc.i32],  # best_index
                    vc.Buff[vc.v2],   # sum_cross
                    vc.Buff[vc.v2],   # sum2_cross
                    vc.Buff[vc.c64],  # y-real/x-fourier correlation
                    rotation_weights_type,
                ] + [vc.Var[vc.i32]] * len(indicies))
        def fused_update(max_cross: vc.Buff[vc.f32],
                         best_index: vc.Buff[vc.i32],
                         sum_cross: vc.Buff[vc.v2],
                         sum2_cross: vc.Buff[vc.v2],
                         spectrum: vc.Buff[vc.c64],
                         rot_weights: vc.Var[vc.f32],
                         *index_values: vc.Var[vc.i32]):
            ind = vc.global_invocation_id().x
            coord_3 = vc.ravel_index(ind, results.max_cross.shape).to_register()

            local_y = coord_3.y.to_dtype(vc.i32).to_register()
            local_x = coord_3.z.to_dtype(vc.i32).to_register()

            signed_y = ((local_y + output_radius) % local_width - output_radius).to_register()
            signed_x = ((local_x + output_radius) % local_width - output_radius).to_register()
            spatial_y = vc.mod(signed_y, correlation_signal.real_shape[1]).to_dtype(vc.i32).to_register()
            spatial_x = vc.mod(signed_x, correlation_signal.real_shape[2]).to_dtype(vc.i32).to_register()

            best_mip = vc.new_float_register(vc.ninf_f32(), var_name="best_mip")
            best_index_val = vc.new_int_register(-1, var_name="best_index_val")
            sum_cross_val = vc.new_vec2_register(0, var_name="sum_cross_val")
            sum2_cross_val = vc.new_vec2_register(0, var_name="sum2_cross_val")

            phase_step = (
                2.0 * np.pi * spatial_x.to_dtype(vc.f32)
                / float(correlation_signal.real_shape[2])
            ).to_register()
            cos_step = vc.cos(phase_step).to_register()
            sin_step = vc.sin(phase_step).to_register()

            for template_i in range(template_count):
                with vc.if_block(index_values[template_i] >= 0):
                    accum = vc.new_float_register(0.0)
                    cos_phase = vc.new_float_register(1.0)
                    sin_phase = vc.new_float_register(0.0)
                    kx = vc.new_int_register(0)

                    spectrum_index = (
                        coord_3.x.to_dtype(vc.i32) * spectrum_micrograph_stride
                        + template_i * spectrum_plane_size
                        + spatial_y * spectrum_row_width
                    ).to_register()

                    with vc.while_block(kx < spectrum_row_width):
                        spectrum_val = spectrum[spectrum_index + kx]
                        term = (spectrum_val.real * cos_phase - spectrum_val.imag * sin_phase).to_register()

                        with vc.if_block(vc.all(
                            kx > 0,
                            kx < correlation_signal.real_shape[2] // 2,
                        )):
                            term *= 2.0

                        accum += term

                        next_cos_phase = (cos_phase * cos_step - sin_phase * sin_step).to_register()
                        sin_phase[:] = sin_phase * cos_step + cos_phase * sin_step
                        cos_phase[:] = next_cos_phase
                        kx += 1

                    mip = (accum / norm).to_register()

                    sum_cross_val[:] = double_precision_add_f32(sum_cross_val, mip * rot_weights)
                    sum2_cross_val[:] = double_precision_add_f32(sum2_cross_val, mip * mip * rot_weights)

                    with vc.if_block(mip > best_mip):
                        best_mip[:] = mip
                        best_index_val[:] = index_values[template_i]

            sum_cross[ind] = double_precision_add_vec2(sum_cross[ind], sum_cross_val)
            sum2_cross[ind] = double_precision_add_vec2(sum2_cross[ind], sum2_cross_val)

            with vc.if_block(best_mip > max_cross[ind]):
                max_cross[ind] = best_mip
                best_index[ind] = best_index_val

        fused_update(
            results.max_cross,
            results.best_index,
            results.sum_cross,
            results.sum2_cross,
            correlation_signal,
            rotation_weights,
            *indicies
        )
