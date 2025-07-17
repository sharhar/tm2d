import numpy as np
import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *


def create_circle_array(N, r):
    """
    Create a numpy array of shape (N, N) with a circle of ones within radius r*N/2 centered at the array's center.

    Parameters:
    N (int): Size of the array.
    r (float): Radius of the circle as a fraction of N/2.

    Returns:
    numpy.ndarray: Array with a circle of ones and zeros elsewhere.
    """
    # Ensure r is between 0 and 1
    if not (0 <= r <= 1):
        raise ValueError("r must be between 0 and 1")

    # Create a grid of x, y coordinates
    y, x = np.ogrid[-N/2:N/2, -N/2:N/2]
    # Define the circle mask
    mask = x**2 + y**2 <= (r*N/2)**2

    # Create the array
    array = np.zeros((N, N))
    array[mask] = 1
    return array

def get_psd(im, pad_factor=0, normalize=True):
    im_pad = np.pad(im, pad_factor * im.shape[0], mode='constant')
    im_pad_ft = np.fft.fftshift(np.fft.fft2(im_pad))
    psd_pad = np.abs(im_pad_ft)**2
    if normalize:
        psd_pad /= im.size**2 # account for array size
    return psd_pad

def get_stats_from_psd(psd, pad_factor=0, normalized=True):
    if not normalized:
        psd /= (psd.size / (2 * pad_factor + 1)**2)**2 # account for array size
    im_mean = np.sqrt(psd[psd.shape[0] // 2, psd.shape[1] // 2])
    im_var = psd.sum() / (2 * pad_factor + 1)**2  - im_mean**2
    return (im_mean, im_var)

def whiten_image(im, normalize=True, azim_avg=True, return_filter=False):
    psd = get_psd(im)
    if azim_avg:
        (_, _, psd) = get_azim_avg(psd) # azimuthally average
    im_ft = np.fft.fftshift(np.fft.fft2(im))
    filt = np.sqrt(1 / psd)
    if normalize:
        filt /= np.sqrt(im.size - 1) # variance=1
    im = np.real(np.fft.ifft2(np.fft.ifftshift(im_ft * filt)))
    if normalize:
        im -= im.mean() # mean=0
        filt[filt.shape[0] // 2, filt.shape[1] // 2] = 0 # remove dc
    if return_filter:
        return (im, filt)
    else:
        return im

def get_azim_avg(arr, delta=1):
    ny, nx = arr.shape
    y, x = np.indices((ny, nx))
    center_y, center_x = ny // 2, nx // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r_flat = r.flatten()
    array_flat = arr.flatten()
    r_sorted_indices = np.argsort(r_flat)
    r_sorted = r_flat[r_sorted_indices]
    array_sorted = array_flat[r_sorted_indices]
    r_unique, r_start_indices = np.unique(r_sorted, return_index=True)
    r_end_indices = np.roll(r_start_indices, -1)
    r_end_indices[-1] = len(r_sorted)
    azimuthal_avg_1d = np.array([
        array_sorted[start:end].mean() for start, end in zip(r_start_indices, r_end_indices)
    ])
    azimuthal_avg_2d = np.interp(r_flat, r_unique, azimuthal_avg_1d).reshape(arr.shape)
    r_axis = r_unique * delta # delta adds units to the axis
    return (r_axis, azimuthal_avg_1d, azimuthal_avg_2d)

def read_pixel(buff: Buff[c64],
               x_ind: vc.ShaderVariable,
               y_ind: vc.ShaderVariable,
               batch_index: vc.ShaderVariable):
    result = vc.new_vec2(0.0)

    vc.if_all(x_ind >= 0, y_ind >= 0, x_ind < buff.shape[2], y_ind < buff.shape[1])
    result[:] = buff[batch_index * buff.shape[2] * buff.shape[1] + y_ind * buff.shape[2] + x_ind]
    vc.end()

    return result

def interp_pixel(
        buff: Buff[c64],
        x_coord: vc.ShaderVariable,
        y_coord: vc.ShaderVariable,
        batch_index: vc.ShaderVariable):
    x_ind0 = vc.floor(x_coord).cast_to(vc.i32)
    y_ind0 = vc.floor(y_coord).cast_to(vc.i32)

    x_ind1 = vc.ceil(x_coord).cast_to(vc.i32)
    y_ind1 = vc.ceil(y_coord).cast_to(vc.i32)

    value00 = read_pixel(buff, x_ind0, y_ind0, batch_index)
    value01 = read_pixel(buff, x_ind0, y_ind1, batch_index)
    value10 = read_pixel(buff, x_ind1, y_ind0, batch_index)
    value11 = read_pixel(buff, x_ind1, y_ind1, batch_index)

    dx = x_coord - x_ind0
    dy = y_coord - y_ind0

    r1 = value00 * (1 - dx) + value10 * dx
    r2 = value01 * (1 - dx) + value11 * dx

    interpolated_value = vc.new_vec2(r1 * (1 - dy) + r2 * dy)

    return interpolated_value.x * interpolated_value.x + interpolated_value.y * interpolated_value.y

@vd.map_reduce(reduction=vd.SubgroupMin, axes=[2, ])
def azimuthal_sum(buff: Buff[c64]) -> vc.f32:
    ind = vc.mapping_index()

    template_index = ind % (buff.shape[2] * buff.shape[1])
    batch_index = ind / (buff.shape[2] * buff.shape[1])

    radius_index = vc.new_float(template_index / buff.shape[2])

    result_value = vc.new_float(0.0)

    vc.if_statement(radius_index < buff.shape[2])
    angle_index = vc.new_float(np.pi/2 - (np.pi * (template_index % buff.shape[2])) / buff.shape[2])

    x_coord = vc.new_float(radius_index * vc.cos(angle_index))
    y_coord = vc.new_float(radius_index * vc.sin(angle_index))

    vc.if_statement(y_coord < 0)
    y_coord += buff.shape[1]  # Adjust y_coord to be positive
    vc.end()

    result_value[:] = interp_pixel(buff, x_coord, y_coord, batch_index)
    vc.end()

    return result_value

@vd.shader("buff.size")
def apply_whiten_filter(buff: Buff[c64], sums_buffer: Buff[f32], return_filter: Const[i32] = 0):
    ind = vc.global_invocation().x

    template_index = ind % (buff.shape[2] * buff.shape[1])
    batch_index = ind / (buff.shape[2] * buff.shape[1])

    x_index = template_index / buff.shape[2]
    y_index = template_index % buff.shape[2]

    radius0 = vc.new_float(vc.sqrt(x_index * x_index + y_index * y_index))

    x_index = buff.shape[1] - x_index

    radius1 = vc.new_float(vc.sqrt(x_index * x_index + y_index * y_index))

    true_radius = vc.min(radius0, radius1)

    rad_floor = vc.floor(true_radius).cast_to(vc.i32)
    rad_ceil = vc.ceil(true_radius).cast_to(vc.i32)

    value_floor = sums_buffer[rad_floor + batch_index * buff.shape[1]] / buff.shape[2]
    value_ceil = sums_buffer[rad_ceil + batch_index * buff.shape[1]] / buff.shape[2]

    drad = true_radius - rad_floor
    psd = value_floor * (1 - drad) + value_ceil * drad

    filter_value = vc.new_vec2(0.0)

    vc.if_statement(psd > 1e-10)
    filter_value.x = vc.sqrt(1 / psd)
    filter_value.y = filter_value.x
    vc.end()

    vc.if_statement(return_filter == 1)
    buff[ind] = filter_value
    vc.else_statement()
    buff[ind] *= filter_value
    vc.end()

    #vc.if_statement(psd < 1e-10)
    #buff[ind] = "vec2(0)"
    #vc.else_statement()
    #buff[ind] /= vc.sqrt(psd)
    #vc.end()

def whiten_buffer(buffer: vd.Buffer, return_filter: bool = False):
    vd.fft.rfft2(buffer)
    sums_buffer = azimuthal_sum(buffer)
    apply_whiten_filter(buffer, sums_buffer, 1 if return_filter else 0)
    if not return_filter:
        vd.fft.irfft2(buffer)