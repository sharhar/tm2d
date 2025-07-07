import numpy as np

from .whitener import whiten_image

def normalize_image(im, n_std=5, remove_outliers=True):
    im_ref = im.copy()
    if remove_outliers:
        outliers_idxs = np.abs(im - im.mean()) > n_std * im.std()
        im_ref[outliers_idxs] = im_ref.mean()
    return (im_ref - im_ref.mean()) / im_ref.std()

def downsample_image(im, N, rescale=False):
    M = im.shape[0] # assume square!
    im_ft = np.fft.fftshift(np.fft.fft2(im)) # fft2

    # get center indices of fft2
    M_center = M // 2
    N_center = N // 2

    # get cropping ranges
    if N % 2 == 0:
        start_M = M_center - N_center
        end_M = M_center + N_center
    else:
        start_M = M_center - N_center
        end_M = M_center + N_center + 1

    im_ft_crop = im_ft[start_M:end_M, start_M:end_M] # crop the fourier transform
    im_ds = np.fft.ifft2(np.fft.ifftshift(im_ft_crop)) # invert fft2
    im_ds_real = np.real(im_ds) # get real part

    # rescale to keep number of electrons constant
    if rescale:
        im_ds_real_sum = im_ds_real.sum()
        if im_ds_real_sum != 0:
            im_ds_real *= im.sum() / im_ds_real_sum

    return im_ds_real

def process_raw_micrograph(
        data: np.ndarray, 
        whiten: bool = True, 
        normalize_input: bool = True, 
        remove_outliers: bool = True, 
        n_std = 5):
    assert data.ndim == 2

    if normalize_input:
        reference_image = data.copy()
        test_image_normalized: np.ndarray = normalize_image(reference_image, n_std=n_std, remove_outliers=remove_outliers)
    else:
        test_image_normalized: np.ndarray = data.copy() # don't actually normalize

    if whiten:
        test_image_normalized = whiten_image(test_image_normalized)

    return test_image_normalized