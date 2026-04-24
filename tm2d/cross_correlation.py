import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import Buff, f32, c64, v2, Const

import numpy as np

from .plan import Comparator

def make_crop_mapping(micrograph_shape: tuple[int, int, int], template_shape: tuple) -> vd.MappingFunction:
    @vd.map
    def crop_mapping(input: Buff[f32], sum_buffer: Buff[c64]):
        read_op = vd.fft.read_op()

        template_index = read_op.io_index // (micrograph_shape[1] * (micrograph_shape[2] + 2))
        ind = vc.new_uint_register(read_op.io_index % (micrograph_shape[1] * (micrograph_shape[2] + 2)))
        ind[:] = ind - 2 * (ind // micrograph_shape[2])

        out_x = (ind // micrograph_shape[2]) % micrograph_shape[1]
        out_y = ind % micrograph_shape[2]

        in_coords = vc.new_uvec2_register()

        with vc.if_block(vc.any(
            vc.all(
                out_x >= template_shape[0] // 2,
                out_x < micrograph_shape[1] - template_shape[0] // 2
            ),
            vc.all(
                out_y >= template_shape[1] // 2,
                out_y < micrograph_shape[2] - template_shape[1] // 2
            )
        )):
            read_op.register.real = 0.0
            read_op.register.imag = 0.0

        with vc.else_block():
            with vc.if_block(out_x < micrograph_shape[1] // 2):
                in_coords.x = out_x
            with vc.else_block():
                in_coords.x = template_shape[0] + out_x - micrograph_shape[1]

            with vc.if_block(out_y < micrograph_shape[2] // 2):
                in_coords.y = out_y
            with vc.else_block():
                in_coords.y = template_shape[1] + out_y - micrograph_shape[2]

            in_coords.x = in_coords.x * template_shape[1] + in_coords.y

            read_op.register[:] = sum_buffer[template_index] / (template_shape[0] * template_shape[1])
            read_op.register.imag = vc.sqrt(read_op.register.imag - read_op.register.real * read_op.register.real)

            in_coords.x = in_coords.x + 2 * (in_coords.x // (template_shape[1]))

            in_coords.x = in_coords.x + template_index * (micrograph_shape[1] * (micrograph_shape[2] + 2))

            read_op.register.real = (input[in_coords.x] - read_op.register.real) / read_op.register.imag
            read_op.register.imag = 0

    return crop_mapping

@vd.shader(exec_size=lambda args: args.buf.size)
def fill_buffer(buf: Buff[c64], val: Const[c64] = 0):
    buf[vc.global_invocation_id().x] = val

class ComparatorCrossCorrelation(Comparator):
    def __init__(self, shape: tuple, template_shape: tuple):
        assert len(shape) == 3, "Shape must be a 3D tuple (N, H, W) but got: {}".format(shape)

        self.micrographs_buffer = vd.RFFTBuffer(shape)

        self.crop_mapping = make_crop_mapping(shape, template_shape)

    def set_data(self, data: np.ndarray):
        if data.ndim == 2:
            data = data.reshape((1, *data.shape))

        assert data.ndim == 3, "Micrographs must be a 3D array (N, H, W)."
        data = np.fft.rfft2(data).astype(np.complex64)
        self.micrographs_buffer.write_fourier(data)

    def compare_template(self, template_buffer: vd.RFFTBuffer, normalize: bool = True) -> vd.RFFTBuffer:
        assert normalize, "Normalization is mandatory for cross-correlation."

        correlation_shape = (
            self.micrographs_buffer.real_shape[0] * template_buffer.shape[0],
            self.micrographs_buffer.real_shape[1],
            self.micrographs_buffer.real_shape[2]
        )

        correlation_signal = vd.RFFTBuffer(correlation_shape)

        @vd.reduce.map_reduce(vd.reduce.SubgroupAdd, axes=[1, 2])
        def calc_sums(wave: Buff[c64]) -> v2:
            ind = vd.reduce.mapped_io_index()

            result = vc.new_vec2_register()

            with vc.if_block(ind % template_buffer.shape[2] < template_buffer.shape[2] - 1):
                result.x = wave[ind].real + wave[ind].imag
                result.y = wave[ind].real * wave[ind].real + wave[ind].imag * wave[ind].imag

            with vc.else_block():
                result.x = 0.0
                result.y = 0.0

            return result

        template_sum = calc_sums(template_buffer)

        in_buffer_shape = (
            template_buffer.shape[0],
            self.micrographs_buffer.real_shape[1],
            self.micrographs_buffer.real_shape[2]
        )

        vd.fft.fft(
            correlation_signal,
            template_buffer,
            template_sum,
            buffer_shape=in_buffer_shape,
            input_map=self.crop_mapping,
            r2c=True
        )

        in_buffer_shape = (
            template_buffer.shape[0],
            self.micrographs_buffer.shape[1],
            self.micrographs_buffer.shape[2]
        )

        kernel_offset = int(
            self.micrographs_buffer.shape[1] * self.micrographs_buffer.shape[2]
        )

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
            buffer_shape=in_buffer_shape,
            kernel_map=convolution_map,
            output_map=output_map_func
        )

        vd.fft.irfft(correlation_signal)

        return correlation_signal