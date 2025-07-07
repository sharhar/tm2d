import numpy as np

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