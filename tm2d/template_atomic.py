import vkdispatch as vd 
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import Tuple, List

import tm2d
import tm2d.utilities as tu

import numpy as np

from .plan import Template
from .ctf import CTFParams, ctf_filter

def make_atomic_template_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    in_matricies = np.zeros(shape=(4, 4, angles.shape[0]), dtype=np.float32)

    cos_phi   = np.cos(np.deg2rad(angles[:, 0]))
    sin_phi   = np.sin(np.deg2rad(angles[:, 0]))
    cos_theta = np.cos(np.deg2rad(angles[:, 1]))
    sin_theta = np.sin(np.deg2rad(angles[:, 1]))

    M00 = cos_phi * cos_theta 
    M01 = -sin_phi 

    M10 = sin_phi * cos_theta 
    M11 = cos_phi 

    M20 = -sin_theta 

    cos_psi_in_plane   = np.cos(np.deg2rad(-angles[:, 2] - 90)) 
    sin_psi_in_plane   = np.sin(np.deg2rad(-angles[:, 2] - 90))

    m00  = cos_psi_in_plane
    m01 = sin_psi_in_plane
    m10 = -sin_psi_in_plane
    m11 = cos_psi_in_plane

    in_matricies[0, 0] = m00 * M00 + m10 * M01
    in_matricies[0, 1] = m00 * M10 + m10 * M11
    in_matricies[0, 2] = m00 * M20
    
    in_matricies[1, 0] = m01 * M00 + m11 * M01
    in_matricies[1, 1] = m01 * M10 + m11 * M11
    in_matricies[1, 2] = m01 * M20

    return in_matricies.T

@vd.shader(exec_size=lambda args: args.buf.size)
def fill_buffer(buf: Buff[c128], val: Const[c128] = 0):
    buf[vc.global_invocation_id().x] = val

def gaussian_filter(buffer_shape: tuple[int, int], tid: vc.ShaderVariable, pixel_size: float, A: float = 100.0):
    pix_size_sq = pixel_size * pixel_size
    
    amp = A / pix_size_sq
    B0 = 8 * np.pi** 2 * (0.27**2 + pix_size_sq / 12) # [A^2] blurring B-factor with contributions from pixel size and physical PSF
    var = 2 * pix_size_sq / B0

    ind = tid.to_dtype(i32).to_register()

    x = (ind // buffer_shape[1]).to_register()
    y = (ind % buffer_shape[1]).to_register()

    x[:] = x + buffer_shape[0] // 2
    x[:] = x % buffer_shape[0]
    x[:] = x - buffer_shape[0] // 2

    x_norm = (x.to_dtype(vd.float32) / buffer_shape[0]).to_register()
    y_norm = (y.to_dtype(vd.float32) / (buffer_shape[1] * 2 - 2)).to_register()

    my_dist = vc.new_float_register()
    my_dist[:] = (x_norm*x_norm + y_norm*y_norm) / ( var * 2 )

    vc.if_statement(my_dist > 100)
    my_dist[:] = 0
    vc.else_statement()
    my_dist[:] = amp * vc.exp(-my_dist) / (buffer_shape[0] * (buffer_shape[1] * 2 - 2))
    vc.end()

    return my_dist


class TemplateAtomic(Template):
    shape: Tuple[int, int]
    atomic_coords: np.ndarray
    atomic_coords_buffer: vd.Buffer
    disable_ctf: bool = False
    disable_convolution: bool = False
    disable_sigma_e: bool = False

    def __init__(self,
                 shape: Tuple[int, int],
                 atomic_coords: np.ndarray,
                 disable_ctf: bool = False,
                 disable_convolution: bool = False,
                 disable_sigma_e: bool = False):
        assert len(shape) == 2, "Shape must be a tuple of two integers (height, width)."
        assert atomic_coords.ndim == 2 and atomic_coords.shape[1] == 3, "Atomic coordinates must be a 2D array with shape (N, 3)."

        self.disable_ctf = disable_ctf
        self.disable_convolution = disable_convolution
        self.disable_sigma_e = disable_sigma_e

        self.shape = (shape[0], shape[1])
        self.atomic_coords = atomic_coords.astype(np.float32)
        self.atomic_coords_buffer = vd.asbuffer(atomic_coords)

    def _make_template(self,
                      rotations: vc.Var[vc.m4],
                      pixel_size: float,
                      ctf_params: CTFParams,
                      template_count: int,
                      cmd_graph: vd.CommandGraph) -> vd.RFFTBuffer:
        
        sigma_e = tu.get_sigmaE(ctf_params.HT)

        template_buffer = vd.RFFTBuffer((template_count, *self.shape), fourier_type=c128)
        template_int_buffer = vd.RFFTBuffer((template_count, *self.shape), fourier_type=c64)

        rotation_type: type = vc.Const[vc.m4] if isinstance(rotations, np.ndarray) else vc.Var[vc.m4]
        pixel_size_type: type = vc.Const[vd.float32] if isinstance(pixel_size, float) else vc.Var[vd.float32]
        
        @vd.shader(exec_size=lambda args: args.atom_coords.shape[0])
        def place_atoms(image: Buff[i32], atom_coords: Buff[f32], rot_matrix: rotation_type, my_pixel_size: pixel_size_type): # type: ignore
            ind = vc.global_invocation_id().x.to_register()

            pos = vc.new_vec4_register()
            pos.x = -atom_coords[3*ind + 1] / my_pixel_size
            pos.y = atom_coords[3*ind + 0] / my_pixel_size
            pos.z = atom_coords[3*ind + 2] / my_pixel_size
            pos.w = 1

            pos[:] = rot_matrix * pos

            image_ind = vc.new_ivec2_register()
            image_ind.y = vc.ceil(pos.y).to_dtype(vd.int32) + (image.shape.y // 2)
            image_ind.x = vc.ceil(-pos.x).to_dtype(vd.int32) + ((image.shape.z * 2 - 2) // 2)

            vc.if_any(image_ind.x < 0, image_ind.x >= image.shape.y, image_ind.y < 0, image_ind.y >= (image.shape.z * 2 - 2))
            vc.return_statement()
            vc.end()

            vc.atomic_add(image[2 * image_ind.x * image.shape.z + image_ind.y], 1)

        fill_buffer(template_int_buffer)
        place_atoms(template_int_buffer, self.atomic_coords_buffer, rotations, pixel_size)

        my_sigma_e = 1.0 if self.disable_sigma_e else sigma_e

        if self.disable_convolution:
            @vd.shader("buff.size * 2")
            def map_int_to_float_shader(buff: Buff[f64], in_buff: Buff[f32]):
                tid = vc.global_invocation_id().x
                buff[tid] = vc.float_bits_to_int(in_buff[tid]).to_dtype(vd.float32) * my_sigma_e

            map_int_to_float_shader(template_buffer, template_int_buffer)

            return template_buffer

        @vd.map
        def map_int_to_float(buff: Buff[f32]):
            read_op = vd.fft.read_op()

            value = vc.float_bits_to_int(buff[read_op.io_index]).to_dtype(vd.float32) * my_sigma_e

            read_op.register.real = value.to_dtype(vd.float64)
            read_op.register.imag = 0.0

        vd.fft.fft(
            template_buffer,
            template_int_buffer,
            buffer_shape=self.shape,
            r2c=True,
            input_map=map_int_to_float,
            compute_type=c128,
            output_type=c128,
        )

        def ctf_map_func(*in_args: Var):
            read_op = vd.fft.read_op()

            tid = read_op.io_index
            value = read_op.register

            my_pixel_size = in_args[0]
            
            value[:] = value * gaussian_filter(
                template_buffer.shape[1:],
                tid,
                my_pixel_size
            )

            if self.disable_ctf:
                return

            upos_2d = vc.new_uvec2_register()
            upos_2d.x = tid % template_buffer.shape[2]
            upos_2d.y = ((tid // template_buffer.shape[2]) + template_buffer.shape[1] // 2) % template_buffer.shape[1]
            
            pos_2d = upos_2d.to_dtype(vc.dv2).to_register()
            pos_2d.y = pos_2d.y - template_buffer.shape[1] // 2

            ctf_param_list = ctf_params.assemble_params_list_from_args(in_args[1:], template_count)

            value[:] = ctf_filter(
                template_buffer.shape[1:],
                pos_2d,
                ctf_param_list[vd.fft.mapped_kernel_index()],
                my_pixel_size
            ) * value

        ctf_map = vd.map(ctf_map_func, input_types = [pixel_size_type] + ctf_params.get_type_list() * template_count)

        @vd.map
        def output_map_func(output_buffer: vc.Buffer[c128]):
            write_op = vd.fft.write_op()

            output_buffer[write_op.io_index + template_buffer.shape[1] * template_buffer.shape[2] * vd.fft.mapped_kernel_index()] = write_op.register

        vd.fft.convolve(
            template_buffer,
            template_buffer,
            pixel_size,
            *ctf_params.get_args(cmd_graph, template_count),
            kernel_map=ctf_map,
            axis=1,
            output_map=output_map_func,
            kernel_num=template_buffer.shape[0],
            buffer_shape=(1, *template_buffer.shape[1:]),
            normalize=False,
            compute_type=c128,
            input_type=c128,
        )

        vd.fft.irfft(template_buffer, normalize=False)

        return template_buffer
    
    def _get_rotation_matricies(self, rotations: np.ndarray) -> np.ndarray:
        return make_atomic_template_rotation_matrix(rotations).astype(np.float32)
    
    def get_shape(self) -> tuple:
        return self.shape