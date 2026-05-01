import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Tuple

from ..plan import Template
from ..ctf.ctf import CTFParams, ctf_filter

import numpy as np

@vd.shader(exec_size=lambda args: args.input_buff.size * 2)
def fftshift(output: vc.Buff[vc.f32], input_buff: vc.Buff[vc.f32]):
    ind = vc.global_invocation_id().x.to_dtype(vd.int32).to_register()

    image_ind = vc.new_int_register()
    image_ind[:] = ind % (input_buff.shape.y * input_buff.shape.z * 2)

    out_x = (image_ind // (2 * input_buff.shape.z)).to_register()
    out_y = (image_ind % (2 * input_buff.shape.z)).to_register()

    with vc.if_block(out_y >= 2 * input_buff.shape.z - 2):
        output[ind].x = 0
        output[ind].y = 0
        vc.return_statement()

    image_ind[:] = ind // (input_buff.shape.y * input_buff.shape.z * 2)

    image_ind[:] = image_ind * (input_buff.shape.y * input_buff.shape.z * 2)

    in_x = ((out_x + input_buff.shape.y // 2) % output.shape.y).to_register()
    in_y = ((out_y + input_buff.shape.y // 2) % output.shape.y).to_register()

    image_ind += in_x * 2 * input_buff.shape.z + in_y

    output[ind] = input_buff[image_ind]

@vd.shader(exec_size=lambda args: args.buff.size)
def template_slice(buff: vc.Buff[vc.c64], img: vc.Img3[vc.f32], img_shape: vc.Const[vc.iv4], rotation: vc.Var[vc.m4]):
    ind = vc.global_invocation_id().x.to_dtype(vd.int32).to_register()

    # calculate the planar position of the current buffer pixel
    pos_2d = vc.ravel_index(ind, buff.shape).swizzle("xy").to_dtype(vc.v2).to_register()

    my_pos = vc.new_vec4_register(0, 0, 0, 1)
    my_pos.x = pos_2d.x + buff.shape.x / 2
    my_pos.y = pos_2d.y + buff.shape.y / 2

    temp_var = vc.mod(my_pos.swizzle("xy"), buff.shape.swizzle("xy")).to_register()

    my_pos.x = temp_var.x - buff.shape.x / 2
    my_pos.y = temp_var.y - buff.shape.y / 2

    # rotate the position to 3D template space
    my_pos[:] = rotation * my_pos

    my_pos.x += img_shape.x / 2
    my_pos.y += img_shape.y / 2
    my_pos.z += img_shape.z / 2

    # sample the 3D image at the current position
    buff[ind] = img.sample(my_pos.swizzle("xyz")).swizzle("xy").to_dtype(vc.c64)

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

def extract_fft_slices(
    template_buffer: vd.RFFTBuffer,
    sampler: vd.Sampler,
    image_size: int,
    rotations: np.ndarray,
    pixel_size: float,
    base_pixel_size: float,
    ctf_params: CTFParams,
    disable_ctf: bool,
    cmd_graph: vd.CommandGraph
):

    rotation_type: type = vc.Const[vc.m4] if isinstance(rotations, np.ndarray) else vc.Var[vc.m4]
    pixel_size_type: type = vc.Const[vd.float32] if isinstance(pixel_size, float) else vc.Var[vd.float32]

    @vd.shader(
        exec_size=template_buffer.shape[1] * template_buffer.shape[2],
        arg_type_annotations= [
            vc.Buff[vc.c64], # output buffer
            vc.Img3[vc.f32], # input image
            vc.Const[vc.i32], # input image dim
            rotation_type, # rotation matrix
            pixel_size_type, # pixel size
            *ctf_params.get_type_list(template_buffer.shape[0]) # ctf params
        ]
    )
    def extract_fft_slices_shader(
        buff: vc.Buff[vc.c64],
        img: vc.Img3[vc.f32],
        img_dim: vc.Const[vc.i32],
        rotation,
        pix_size,
        *in_args):

        ind = vc.global_invocation_id().x.to_dtype(vc.i32).to_register()

        # calculate the planar position of the current buffer pixel
        ipos = vc.new_ivec4_register(0, 0, 0, 1)
        ipos.x = ind % template_buffer.shape[2]
        ipos.y = ind // template_buffer.shape[2]

        ipos.y += img_dim // 2
        ipos.y[:] = vc.mod(ipos.y, img_dim).to_dtype(vc.i32)
        ipos.y -= img_dim // 2

        my_pos = ipos.to_dtype(vc.v4).to_register()

        # rotate the position to 3D template space
        my_pos[:] = rotation * my_pos * (base_pixel_size / pix_size)

        with vc.if_block(vc.any(
            my_pos.x < -template_buffer.shape[1],
            my_pos.x > (template_buffer.shape[1] - 1),
            my_pos.y < -template_buffer.shape[1],
            my_pos.y > (template_buffer.shape[1] - 1),
            my_pos.z < -template_buffer.shape[1],
            my_pos.z > (template_buffer.shape[1] - 1))):

            for i in range(template_buffer.shape[0]):
                index = ind + i * template_buffer.shape[1] * template_buffer.shape[2]
                buff[index].real = 0
                buff[index].imag = 0
            vc.return_statement()

        value = img.sample(my_pos.swizzle("xyz")).swizzle("xy").to_dtype(vc.c64).to_register()

        ipos_2d = vc.new_ivec2_register()
        ipos_2d.x = ind % template_buffer.shape[2]
        ipos_2d.y = ((ind // template_buffer.shape[2]) + template_buffer.shape[1] // 2) % template_buffer.shape[1]

        pos_2d = ipos_2d.to_dtype(vc.v2).to_register()
        pos_2d.y = pos_2d.y - template_buffer.shape[1] // 2

        ctf_param_list = ctf_params.assemble_params_list_from_args(in_args, template_buffer.shape[0])

        for i in range(template_buffer.shape[0]):
            index = ind + i * template_buffer.shape[1] * template_buffer.shape[2]
            if disable_ctf:
                buff[index] = value
            else:
                buff[index] = vc.mult_complex(value, ctf_filter(
                    template_buffer.shape[1:],
                    pos_2d,
                    ctf_param_list[i],
                    pix_size
                ))

    extract_fft_slices_shader(
        template_buffer,
        sampler,
        image_size,
        rotations,
        pixel_size,
        *ctf_params.get_args(cmd_graph, template_buffer.shape[0])
    )

class TemplateDensity(Template):
    shape: Tuple[int, int]
    density_array: np.ndarray
    density_pixel_size: float
    density_image: vd.Image3D
    disable_ctf: bool

    def __init__(self,
                 density_array: np.ndarray,
                 density_pixel_size: float,
                 transformed: bool = False,
                 disable_ctf: bool = False):
        assert density_array.ndim == 3, "Density array must be a 3D array."

        self.disable_ctf = disable_ctf
        self.density_array = density_array
        self.density_pixel_size = density_pixel_size

        if not transformed:
            self.density_array = np.fft.fftn(np.fft.fftshift(density_array)).astype(np.complex64)

        self.density_array = self.density_array.astype(np.complex64)

        self.density_image = vd.Image3D(
            self.density_array.shape, vd.float32, 2
        )
        self.density_image.write(self.density_array)

        self.density_image.sample

        self.shape = (self.density_array.shape[0], self.density_array.shape[1])

    def _make_template(self,
                    rotations: vc.Var[vc.m4],
                    pixel_size: float,
                    ctf_params: CTFParams,
                    template_count: int,
                    cmd_graph: vd.CommandGraph) -> vd.RFFTBuffer:

        template_buffer_temp = vd.RFFTBuffer((template_count, *self.shape))

        # extract the 2D slices from the 3D density and apply the ctf
        extract_fft_slices(
            template_buffer_temp,
            self.density_image.sample(
                address_mode=vd.AddressMode.REPEAT,
            ),
            self.density_array.shape[1],
            rotations,
            pixel_size,
            self.density_pixel_size,
            ctf_params,
            self.disable_ctf,
            cmd_graph
        )


        vd.fft.irfft2(template_buffer_temp, normalize=True)

        template_buffer = vd.RFFTBuffer((template_count, *self.shape))

        fftshift(template_buffer, template_buffer_temp)

        return template_buffer

    def get_rotation_matricies(self, rotations: np.ndarray) -> np.ndarray:
        return make_density_template_rotation_matrix(rotations).astype(np.float32)

    def get_shape(self) -> tuple:
        return self.shape