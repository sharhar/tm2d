import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import Buff, f32, c64, v2, Const

import numpy as np

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

    def compare_template(self, template_buffer: vd.RFFTBuffer, normalize: bool = True, output_radius: int = None) -> vd.RFFTBuffer:
        assert normalize, "Normalization is mandatory for cross-correlation."

        correlation_shape = (
            self.micrographs_buffer.real_shape[0] * template_buffer.shape[0],
            self.micrographs_buffer.real_shape[1],
            self.micrographs_buffer.real_shape[2]
        )

        correlation_signal = vd.RFFTBuffer(correlation_shape)
        correlation_signal_out = vd.RFFTBuffer(correlation_shape)

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
            if output_radius is None:
                out_reg = vd.fft.write_op().register
                output_buffer[vd.fft.write_op().io_index + kernel_offset * template_buffer.shape[0] * vd.fft.mapped_kernel_index()] = out_reg
                return

            with vc.if_block(vc.any(
                vd.fft.write_op().fft_index < output_radius,
                vd.fft.write_op().fft_index >= self.micrographs_buffer.shape[1] - output_radius)):

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

        if output_radius is None:
            vd.fft.irfft(correlation_signal)
            return correlation_signal

        trimmed_shape = (
            correlation_signal.real_shape[0],
            output_radius * 2,
            correlation_signal.real_shape[2]
        )

        trimmed_shape_2 = (
            correlation_signal.shape[0],
            output_radius * 2,
            correlation_signal.shape[2] * 2
        )

        trimmed_shape_3 = (
            correlation_signal.shape[0],
            output_radius * 2,
            correlation_signal.shape[2]
        )

        corr_shape_2 = (
            correlation_signal.shape[0],
            correlation_signal.shape[1],
            correlation_signal.shape[2] * 2,
        )

        @vd.map
        def irfft_map_input(input_buffer: vc.Buffer[c64]):
            io_index = vd.fft.read_op().io_index

            with vc.if_block(vd.fft.read_op().fft_index >= (vd.fft.read_op().fft_size // 2) + 1):
                io_index[:] = vd.fft.read_op().r2c_inverse_offset - io_index

            coord3d = vc.ravel_index(vd.fft.read_op().io_index, trimmed_shape_3).to_register()

            with vc.if_block(coord3d.y >= output_radius):
              coord3d.y += correlation_signal.shape[1] - output_radius * 2

            new_io_index = vc.unravel_index(coord3d, correlation_signal.shape).to_register()

            vd.fft.read_op().register[:] = input_buffer[new_io_index]

            with vc.if_block(vd.fft.read_op().fft_index >= (vd.fft.read_op().fft_size // 2) + 1):
                vd.fft.read_op().register.imag = -vd.fft.read_op().register.imag

        @vd.map
        def irfft_map_output(output_buffer: vc.Buffer[c64]):
            coord3d = vc.ravel_index(vd.fft.write_op().io_index, trimmed_shape_2).to_register()

            with vc.if_block(vc.any(
                coord3d.z < output_radius,
                coord3d.z >= self.micrographs_buffer.real_shape[2] - output_radius)):

                with vc.if_block(coord3d.y >= output_radius):
                    coord3d.y += correlation_signal.shape[1] - output_radius * 2

                new_io_index = vc.unravel_index(coord3d, corr_shape_2).to_register()

                vd.fft.write_op().write_to_buffer(output_buffer, io_index=new_io_index)

        vd.fft.fft(
            correlation_signal,
            correlation_signal,
            buffer_shape=trimmed_shape,
            input_map=irfft_map_input,
            output_map=irfft_map_output,
            r2c=True,
            inverse=True
        )

        return correlation_signal