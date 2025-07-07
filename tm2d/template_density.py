import vkdispatch as vd 
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import tm2d

from typing import Tuple, List

import tm2d.utilities as tu

from .plan import Template
from .ctf import CTFParams, ctf_filter

import numpy as np

@vd.shader(exec_size=lambda args: args.input_buff.size * 2)
def fftshift(output: vc.Buff[vc.f32], input_buff: vc.Buff[vc.f32]):

    ind = vc.global_invocation().x.cast_to(vd.int32).copy()

    image_ind = vc.new_int()
    image_ind[:] = ind % (input_buff.shape.y * input_buff.shape.z * 2)

    out_x = (image_ind / (2 * input_buff.shape.z)).copy()
    out_y = (image_ind % (2 * input_buff.shape.z)).copy()

    vc.if_statement(out_y >= 2 * input_buff.shape.z - 2)
    output[ind].x = 0
    output[ind].y = 0
    vc.return_statement()
    vc.end()

    image_ind[:] = ind / (input_buff.shape.y * input_buff.shape.z * 2)

    image_ind[:] = image_ind * (input_buff.shape.y * input_buff.shape.z * 2)

    in_x = ((out_x + input_buff.shape.y / 2) % output.shape.y).copy()
    in_y = ((out_y + input_buff.shape.y / 2) % output.shape.y).copy()

    image_ind += in_x * 2 * input_buff.shape.z + in_y

    output[ind] = input_buff[image_ind]

@vd.shader(exec_size=lambda args: args.buff.size)
def template_slice(buff: Buff[c64], img: Img3[f32], img_shape: Const[iv4], rotation: Var[m4]):
    ind = vc.global_invocation().x.cast_to(i32).copy()
    
    # calculate the planar position of the current buffer pixel
    my_pos = vc.new_vec4(0, 0, 0, 1)
    my_pos.xy[:] = vc.unravel_index(ind, buff.shape).xy
    my_pos.xy += buff.shape.xy / 2
    my_pos.xy[:] = vc.mod(my_pos.xy, buff.shape.xy)
    my_pos.xy -= buff.shape.xy / 2

    # rotate the position to 3D template space
    my_pos[:] = rotation * my_pos
    my_pos.xyz += img_shape.xyz.cast_to(v3) / 2
    
    # sample the 3D image at the current position
    buff[ind] = img.sample(my_pos.xyz).xy

def make_density_template_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    m = np.zeros(shape=(4, 4, angles.shape[0]), dtype=np.float32)

    cos_phi   = np.cos(np.deg2rad(angles[:, 0]))
    sin_phi   = np.sin(np.deg2rad(angles[:, 0]))
    cos_theta = np.cos(np.deg2rad(angles[:, 1]))
    sin_theta = np.sin(np.deg2rad(angles[:, 1]))
    cos_psi   = np.cos(np.deg2rad(angles[:, 2]))
    sin_psi   = np.sin(np.deg2rad(angles[:, 2]))
    m[0][0]   = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi
    m[1][0]   = sin_phi * cos_theta * cos_psi + cos_phi * sin_psi
    m[2][0]   = -sin_theta * cos_psi
    #m[3][0]   = offsets[0]

    m[0][1]   = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi
    m[1][1]   = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi
    m[2][1]   = sin_theta * sin_psi
    #m[3][1]   = offsets[1]    
    
    m[0][2]   = sin_theta * cos_phi
    m[1][2]   = sin_theta * sin_phi
    m[2][2]   = cos_theta
    #m[3][2]   = offsets[2]

    return m.T

def make_density_template(
        rotation_matrix: np.ndarray,
        density_array: np.ndarray,
        transformed: bool = False) -> vd.Buffer:
    assert density_array.ndim == 3, "Density array must be 3D"
    assert density_array.shape[0] == density_array.shape[1] and density_array.shape[0] == density_array.shape[2], "Density array must be cubic"

    if not transformed:
        density_array = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(density_array))).astype(np.complex64)

    image = vd.Image3D(density_array.shape, vd.float32, 2)
    image.write(density_array)

    signal = tm2d.Signal2D(density_array.shape[0], density_array.shape[0], True, False)

    template_slice(signal.buffer(), image, (*image.shape, 0), rotation_matrix)

    return signal


class TemplateDensity(Template):
    shape: Tuple[int, int]
    density_array: np.ndarray
    density_image: vd.Image3D

    def __init__(self, density_array: np.ndarray, transformed: bool = False):
        assert density_array.ndim == 3, "Density array must be a 3D array."

        self.density_array = density_array

        if transformed:
            self.density_array = np.fft.fftn(np.fft.fftshift(density_array)).astype(np.complex64)
        
        self.density_array = self.density_array.astype(np.complex64)

        self.density_image = vd.Image3D(
            self.density_array.shape, vd.float32, 2
        )
        self.density_image.write(self.density_array)

        self.shape = (self.density_array.shape[0], self.density_array.shape[1])
    
    def _make_template(self,
                    rotations: vc.Var[vc.m4],
                    pixel_size: float,
                    defoci: List[vc.Var[vc.v4]],
                    ctf_params: CTFParams,
                    disable_ctf: bool = False) -> vd.RFFTBuffer:
        
        template_buffer_temp = vd.RFFTBuffer((len(defoci), *self.shape))

        rotation_type: type = vc.Const[vc.m4] if isinstance(rotations, np.ndarray) else vc.Var[vc.m4]

        def extract_fft_slices_func(
            buff: vc.Buff[vc.c64],
            img: vc.Img3[vc.f32],
            img_shape: vc.Const[vc.iv4],
            rotation: vc.Var[vc.m4],
            *defocus_values: vc.Var[vc.v4]):

            ind = vc.global_invocation().x.cast_to(vc.i32).copy()

            # calculate the planar position of the current buffer pixel
            my_pos = vc.new_vec4(0, 0, 0, 1)
            my_pos.x = ind % template_buffer_temp.shape[2]
            my_pos.y = ind / template_buffer_temp.shape[2]

            my_pos.y += img_shape.y / 2
            my_pos.y[:] = vc.mod(my_pos.y, img_shape.y)
            my_pos.y -= img_shape.y / 2

            # rotate the position to 3D template space
            my_pos[:] = rotation * my_pos

            vc.if_any(
                my_pos.x < -template_buffer_temp.shape[1],
                my_pos.x > (template_buffer_temp.shape[1] - 1),
                my_pos.y < -template_buffer_temp.shape[1],
                my_pos.y > (template_buffer_temp.shape[1] - 1),
                my_pos.z < -template_buffer_temp.shape[1],
                my_pos.z > (template_buffer_temp.shape[1] - 1))
            for i in range(template_buffer_temp.shape[0]):
                index = ind + i * template_buffer_temp.shape[1] * template_buffer_temp.shape[2]
                buff[index].x = 0
                buff[index].y = 0
            vc.return_statement()
            vc.end()

            my_pos.xy[:] = img.sample(my_pos.xyz).xy

            pos_2d = vc.new_vec2()
            pos_2d.x = ind % template_buffer_temp.shape[2]
            pos_2d.y = ((ind / template_buffer_temp.shape[2]) + template_buffer_temp.shape[1] // 2) % template_buffer_temp.shape[1]
            pos_2d.y = pos_2d.y - template_buffer_temp.shape[1] // 2

            for i in range(template_buffer_temp.shape[0]):
                index = ind + i * template_buffer_temp.shape[1] * template_buffer_temp.shape[2]
                if disable_ctf:
                    buff[index] = my_pos.xy
                else:
                    buff[index] = my_pos.xy * ctf_filter(
                        template_buffer_temp.shape[1:],
                        defocus_values[i],
                        pos_2d,
                        ctf_params,
                        pixel_size
                    )
        
        with vc.builder_context() as builder:
            signature = vd.ShaderSignature.from_type_annotations(builder, [
                vc.Buff[vc.c64], # output buffer
                vc.Img3[vc.f32], # input image
                vc.Const[vc.iv4], # input image shape
                rotation_type, # rotation matrix
            ] + [Var[v4]] * len(defoci))

            extract_fft_slices_func(*signature.get_variables())

            extract_fft_slices_shader = vd.ShaderObject(
                builder.build("extract_fft_slices_shader"), 
                signature,
                exec_count=template_buffer_temp.shape[1] * template_buffer_temp.shape[2]
            )
        
        extract_fft_slices_shader(
            template_buffer_temp,
            self.density_image.sample(
                address_mode=vd.AddressMode.REPEAT,
            ),
            (*self.density_array.shape, 0),
            rotations,
            *defoci
        )

        vd.fft.irfft2(template_buffer_temp)

        template_buffer = vd.RFFTBuffer((len(defoci), *self.shape))

        fftshift(template_buffer, template_buffer_temp)

        return template_buffer
    
    def get_rotation_matricies(self, rotations: np.ndarray) -> np.ndarray:
        return make_density_template_rotation_matrix(rotations).astype(np.float32)
    
    def get_shape(self) -> tuple:
        return self.shape