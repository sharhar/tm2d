import tm2d
import numpy as np

import tm2d_utils as tu

from matplotlib import pyplot as plt

density = tu.load_density_from_mrc("data/parsed_5lks_LSU_sim_120.mrc")

template_density = tm2d.TemplateDensity(density.density, density.pixel_size)

template_atomic = tm2d.TemplateAtomic(
    (576, 576),
    tu.load_coords_from_npz("data/parsed_5lks_LSU.npz"),
    # fuse_ctf_convolution=True,
)

rot = [45, 78, 123]
pixel_size = 1.06 * 1.34

td = template_density.make_template(
    rot,
    pixel_size=pixel_size,
    template_count=1,
)

ta = template_atomic.make_template(
    rot,
    pixel_size=pixel_size,
    template_count=1,
)

np.save("td.npy", td.read_real(0)[0])
np.save("ta.npy", ta.read_real(0)[0])

