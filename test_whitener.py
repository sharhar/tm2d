import vkdispatch as vd
import tm2d
import numpy as np

import sys

import tm2d.utilities as tu

from matplotlib import pyplot as plt

template_atomic = tm2d.TemplateAtomic(
    (512, 512),
    tu.load_coords_from_npz("data/parsed_5lks_LSU.npz")
)

template_buffer = template_atomic.make_template(
    [188.84183,  78.82107, 326],
    1.056,
    tm2d.CTFParams.like_krios(
        defocus=12870
    ),
    template_count=2
)

template_image_pure = template_buffer.read_real(0)[0]

tu.whiten_buffer(template_buffer)

template_image = template_buffer.read_real(0)[0]

white_image = tu.whiten_image(template_image_pure)

#np.save("filter.npy", np.log(filter))
np.save("template_image.npy", template_image)
np.save("template_pure.npy", template_image_pure)
np.save("white_template.npy", white_image)