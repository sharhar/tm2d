import vkdispatch as vd
import tm2d
import numpy as np

import mrcfile

import tm2d.utilities as tu

from matplotlib import pyplot as plt

#/BigData/Workspaces/jz/250512-ctfmatch/ 

vd.make_context(max_streams=True)

def bin_array_by_factor_2(arr):
    """
    Bins a 2D NumPy array by a factor of 2 along each axis.

    Args:
        arr (np.ndarray): A 2D NumPy array.

    Returns:
        np.ndarray: A new NumPy array where each axis is half the size of the
                    input array, with values being the mean of the binned
                    elements.

    Raises:
        ValueError: If the input array is not 2D or if either dimension
                    is not even.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if arr.shape[0] % 2 != 0 or arr.shape[1] % 2 != 0:
        raise ValueError("Both dimensions of the input array must be even.")

    rows, cols = arr.shape
    binned_arr = np.zeros((rows // 2, cols // 2))

    for i in range(rows // 2):
        for j in range(cols // 2):
            # Calculate the slice for the 2x2 bin
            row_slice = slice(i * 2, (i + 1) * 2)
            col_slice = slice(j * 2, (j + 1) * 2)
            binned_arr[i, j] = np.mean(arr[row_slice, col_slice])

    return binned_arr

star_file = "/BigData/Workspaces/in_focus/25_05_14_rubisco_2DTM_test01/single_micrograph_test01/laser_off_1um_defocus/J1188_particles.star"

micrograph_location = "/BigData/Workspaces/in_focus/25_05_14_rubisco_2DTM_test01/single_micrograph_test01/laser_off_1um_defocus/RuBisCo_Sq6LaserOff_1umR_May15_00.53.46_patch_aligned.mrc"

with mrcfile.open(micrograph_location, mode='r') as mrc:
    micrograph = mrc.data

    # print pixel size

    print("Pixel size:", mrc.voxel_size)

micrographs = np.array([
    #tu.whiten_image(bin_array_by_factor_2(micrograph[912:1712, 924:1724]))
    tu.whiten_image(micrograph[912:1936, 924:1948])
])

print("Micrograph shape:", micrograph.shape)

template = tm2d.TemplateAtomic(
    micrographs.shape[1:],
    tu.load_coords_from_npz("/home/shaharsandhaus/TEM_LPP_Image_Simulator/NPZs/7smk_rubisco.npz")
)

np.save("template.npy", template.make_template([0, 0, 0], 1, 10000, disable_ctf=True).read_real(0)[0])
np.save("ctf.npy", tm2d.generate_ctf((512, 512), 1.4, 10000))

plan = tm2d.PlanStandard(
    template,
    micrographs.shape,
    0.76,
    ctf_params = tm2d.make_ctf_params(
        HT = 300e3,
        Cs = 2.7e7,
        Cc = 2.7e7,
        energy_spread_fwhm = 0.9,
        accel_voltage_std = 0.07e-6,
        OL_current_std = 0.1e-6,
        beam_semiangle = 2.5e-6,
        johnson_std = 0.37,
        B = 25,
        amp_contrast = 0.07,
        lpp = 0,
        NA = 0.05,
        f_OL = 20e7,
        lpp_rot = 0
    )
)

plan.set_data(micrographs)

# 131.974380 99.891502 -175.805740

rotations = tm2d.get_orientations_healpix(
    angular_step_size=2, 
    psi_step_size=1,
    region=tm2d.OrientationRegion(
        symmetry='D4', 
        phi_min=70,
        phi_max=100,
        theta_min=70,
        theta_max=95,
        psi_min=115,
        psi_max=175
    )
)

defocus_values = np.arange(9500, 10500, 500) #400, 900, 100)

plan.run(
    rotations,
    defocus_values,
    enable_progress_bar=True
)

res: tm2d.ResultsPixel = plan.results

np.save("mip0.npy", res.get_mip()[0])
#np.save("mip1.npy", res.get_mip()[1])

np.save("Z_score0.npy", res.get_z_score()[0])
#np.save("Z_score1.npy", res.get_z_score()[1])

np.save("sum_cross0.npy", res.get_sum_cross()[0])
#np.save("sum_cross1.npy", res.get_sum_cross()[1])

np.save("sum2_cross0.npy", res.get_sum2_cross()[0])
#np.save("sum2_cross1.npy", res.get_sum2_cross()[1])

np.save("best_params.npy0", res.get_best_index_array()[0])
#np.save("best_params.npy1", res.get_best_index_array()[1])

best_rotation_index = res.get_index_of_params_match()[0] % rotations.shape[0]
best_defocus_index = res.get_index_of_params_match()[0] // rotations.shape[0]

print("Found max at:", res.get_location_of_best_match()[0])
print("Max cross correlation:", res.get_z_score()[0][res.get_location_of_best_match()[0]])
print("Phi:", rotations[best_rotation_index][0])
print("Theta:", rotations[best_rotation_index][1])
print("Psi:", rotations[best_rotation_index][2])
    
print("Defocus:", defocus_values[best_defocus_index])

rotation_indicies = res.get_best_index_array()[0] % rotations.shape[0]
defocus_indicies = res.get_best_index_array()[0] // rotations.shape[0]

best_rotations = rotations[rotation_indicies]
best_defocus = defocus_values[defocus_indicies]

np.save("best_phi.npy", best_rotations[:, :, 0])
np.save("best_theta.npy", best_rotations[:, :, 1])
np.save("best_psi.npy", best_rotations[:, :, 2])

np.save("best_defocus.npy", best_defocus)