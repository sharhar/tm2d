import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

from .plan import Comparator

def make_crop_mapping(micrograph_shape: tuple[int, int, int], template_shape: tuple) -> vd.MappingFunction:
    @vd.map_registers([c64])
    def crop_mapping(input: Buff[f32], sum_buffer: Buff[v2]):
        template_index = vc.mapping_index() / (micrograph_shape[1] * (micrograph_shape[2] + 2))
        ind = vc.new_uint(vc.mapping_index() % (micrograph_shape[1] * (micrograph_shape[2] + 2)))
        ind[:] = ind - 2 * (ind / micrograph_shape[2])

        out_x = (ind / micrograph_shape[2]) % micrograph_shape[1]
        out_y = ind % micrograph_shape[2]

        in_coords = vc.new_uvec2()
        
        vc.if_any(vc.logical_and(
                    out_x >= template_shape[0] // 2,
                    out_x < micrograph_shape[1] - template_shape[0] // 2),
                vc.logical_and(
                    out_y >= template_shape[1] // 2,
                    out_y < micrograph_shape[2] - template_shape[1] // 2))

        vc.mapping_registers()[0].x = 0.0
        vc.mapping_registers()[0].y = 0.0
        vc.else_statement()

        vc.if_statement(out_x < micrograph_shape[1] // 2)
        in_coords.x = out_x
        vc.else_statement()
        in_coords.x = template_shape[0] + out_x - micrograph_shape[1]
        vc.end()

        vc.if_statement(out_y < micrograph_shape[2] // 2)
        in_coords.y = out_y
        vc.else_statement()
        in_coords.y = template_shape[1] + out_y - micrograph_shape[2]
        vc.end()

        in_coords.x = in_coords.x * template_shape[1] + in_coords.y

        out_reg = vc.mapping_registers()[0]

        out_reg[:] = sum_buffer[template_index] / (template_shape[0] * template_shape[1])
        out_reg.y = vc.sqrt(out_reg.y - out_reg.x * out_reg.x)

        in_coords.x = in_coords.x + 2 * (in_coords.x / (template_shape[1]))

        in_coords.x = in_coords.x + template_index * (micrograph_shape[1] * (micrograph_shape[2] + 2))

        out_reg.x = (input[in_coords.x] - out_reg.x) / out_reg.y
        out_reg.y = 0

        vc.end()

    return crop_mapping

@vd.shader(exec_size=lambda args: args.buf.size)
def fill_buffer(buf: Buff[c64], val: Const[c64] = 0):   
    buf[vc.global_invocation().x] = val

class ComparatorCrossCorrelation(Comparator):
    def __init__(self, shape: tuple, template_shape: tuple):
        assert len(shape) == 3, "Shape must be a 3D tuple (N, H, W)."

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

        @vd.map_reduce(vd.SubgroupAdd, axes=[1, 2])
        def calc_sums(wave: Buff[c64]) -> v2:
            ind = vc.mapping_index()

            result = vc.new_vec2()

            vc.if_statement(ind % template_buffer.shape[2] < template_buffer.shape[2] - 1)

            result.x = wave[ind].x + wave[ind].y
            result.y = wave[ind].x * wave[ind].x + wave[ind].y * wave[ind].y
            vc.else_statement()

            result.x = 0.0
            result.y = 0.0

            vc.end()

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

        @vd.map_registers([c64])
        def convolution_map(kernel_buffer: vc.Buffer[c64]):
            img_val = vc.mapping_registers()[0]
            read_register = vc.mapping_registers()[1]

            read_register[:] = kernel_buffer[vc.mapping_index() % kernel_offset + kernel_offset * vc.kernel_index()]
            img_val[:] = vc.mult_conj_c64(read_register, img_val)

        @vd.map_registers([c64])
        def output_map_func(output_buffer: vc.Buffer[c64]):
            out_reg = vc.mapping_registers()[0]
            output_buffer[vc.mapping_index() + kernel_offset * template_buffer.shape[0] * vc.kernel_index()] = out_reg

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