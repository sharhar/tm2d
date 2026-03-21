import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

import tm2d

from .plan import Comparator

def do_data_crop(io_index: Var[u32],
                 register: Var[f32],
                 micrograph_shape: tuple[int, int, int],
                 template_shape: tuple[int, int, int],
                 input: Buff[f32],
                 sum_buffer: Buff[c64]) -> v2:
    template_index = io_index // (micrograph_shape[1] * (micrograph_shape[2] + 2))
    ind = vc.new_uint_register(io_index % (micrograph_shape[1] * (micrograph_shape[2] + 2)))
    ind[:] = ind - 2 * (ind // micrograph_shape[2])

    out_x = (ind // micrograph_shape[2]) % micrograph_shape[1]
    out_y = ind % micrograph_shape[2]

    in_coords = vc.new_uvec2_register()
    
    vc.if_any(vc.logical_and(
                out_x >= template_shape[0] // 2,
                out_x < micrograph_shape[1] - template_shape[0] // 2),
            vc.logical_and(
                out_y >= template_shape[1] // 2,
                out_y < micrograph_shape[2] - template_shape[1] // 2))

    register[:] = 0.0
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

    sum_vals = (sum_buffer[template_index] / (template_shape[0] * template_shape[1])).to_register()
    sum_vals.imag = vc.sqrt(sum_vals.imag - sum_vals.real * sum_vals.real)

    in_coords.x = in_coords.x + 2 * (in_coords.x // (template_shape[1]))

    in_coords.x = in_coords.x + template_index * (micrograph_shape[1] * (micrograph_shape[2] + 2))

    register[:] = (input[in_coords.x] - sum_vals.real) / sum_vals.imag

    vc.end()

def make_crop_mapping(micrograph_shape: tuple[int, int, int], template_shape: tuple) -> vd.MappingFunction:
    @vd.map
    def crop_mapping(input: Buff[f32], sum_buffer: Buff[c64]):
        read_op = vd.fft.read_op()
        do_data_crop(read_op.io_index, read_op.register.real, micrograph_shape, template_shape, input, sum_buffer)
        read_op.register.imag = 0.0

    return crop_mapping

@vd.shader(exec_size=lambda args: args.buf.size)
def fill_buffer(buf: Buff[c64], val: Const[c64] = 0):   
    buf[vc.global_invocation_id().x] = val

class ComparatorCrossCorrelation(Comparator):
    def __init__(self, shape: tuple, template_shape: tuple):
        assert len(shape) == 3, "Shape must be a 3D tuple (N, H, W) but got: {}".format(shape)

        self.micrographs_buffer = vd.RFFTBuffer(shape)

        self.crop_mapping = make_crop_mapping(shape, template_shape)

        @vd.shader()
        def do_unfused_crop(out: Buff[f32], input: Buff[f32], sum_buffer: Buff[c64]):
            tid = vc.global_invocation_id().x

            do_data_crop(tid, out[tid], shape, template_shape, input, sum_buffer)

        self.unfused_crop_shader = do_unfused_crop

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

            vc.if_statement(ind % template_buffer.shape[2] < template_buffer.shape[2] - 1)

            result.x = wave[ind].real + wave[ind].imag
            result.y = wave[ind].real * wave[ind].real + wave[ind].imag * wave[ind].imag
            vc.else_statement()

            result.x = 0.0
            result.y = 0.0

            vc.end()

            return result
        
        template_sum = calc_sums(template_buffer)

        in_real_buffer_shape = (
            template_buffer.shape[0],
            self.micrographs_buffer.real_shape[1],
            self.micrographs_buffer.real_shape[2]
        )

        if tm2d.disable_kernel_fusion:
            exec_count = template_buffer.shape[0] * self.micrographs_buffer.real_shape[1] * (self.micrographs_buffer.real_shape[2] + 2)

            self.unfused_crop_shader(correlation_signal, template_buffer, template_sum, exec_size=exec_count)
            vd.fft.fft(correlation_signal, buffer_shape=in_real_buffer_shape, r2c=True)
        else:
            vd.fft.fft(
                correlation_signal,
                template_buffer,
                template_sum,
                buffer_shape=in_real_buffer_shape,
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

        if tm2d.disable_kernel_fusion:
            vd.fft.fft(correlation_signal, axis=1, buffer_shape=in_buffer_shape)

            @vd.shader()
            def multiply_conjugate_kernel(corr_buff: Buff[c64], kernel_buffer: Buff[c64]):
                tid = vc.global_invocation_id().x
                img_val = corr_buff[tid]

                for kernel_index in range(self.micrographs_buffer.shape[0]):
                    kernel_val = kernel_buffer[tid % kernel_offset + kernel_offset * kernel_index].to_register()
                    corr_buff[tid + kernel_offset * template_buffer.shape[0] * kernel_index] = vc.mult_complex(kernel_val, img_val.conjugate())

            multiply_conjugate_kernel(correlation_signal, self.micrographs_buffer, exec_size=np.prod(in_buffer_shape))

            vd.fft.ifft(correlation_signal, axis=1)
        else:
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