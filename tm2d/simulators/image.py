import numpy as np
import vkdispatch as vd 
import tm2d
import tm2d.utilities as tu
import tm2d.utilities

from ..plan import Template

from typing import Tuple

from ..ctf import CTFParams

def simulate_pdf(
        template: Template,
        rotation: Tuple[float, float, float],
        defocus: float,
        pixel_size: float,
        ctf_params: CTFParams = None,
        ) -> np.ndarray:
    """
    Simulate a power density function (PDF) for a given template, rotation, defocus, and pixel size.
    """

    if ctf_params is None:
        ctf_params = CTFParams()
    
    return template.make_template(
        template.get_rotation_matricies(np.array([rotation])),
        [defocus, 0, 0, 0],
        pixel_size,
        ctf_params
    ).read_real(0)

def get_im_from_pdf(pdf, dose_per_A2, pix_size, snr=1):
    dose_per_pix = tu.optics_functions.dose_A2ToPix(dose_per_A2, pix_size)
    white_noise = np.random.normal(0, np.sqrt(1 / snr), size=pdf.shape)
    pdf_noisy = pdf + white_noise # add white noise
    pdf_noisy[pdf_noisy < 0] = 0 # enforce non-negativity
    return dose_per_pix * pdf_noisy # [e]