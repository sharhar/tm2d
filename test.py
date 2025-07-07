import vkdispatch as vd
import tm2d
import numpy as np

import tm2d.utilities as tu

from matplotlib import pyplot as plt

#vd.initialize(debug_mode=True)

vd.make_context(max_streams=True) #devices=[0], queue_families=[[0, 2]])

#vd.initialize(log_level=vd.LogLevel.INFO)

def plot_results(res: tm2d.ResultsPixel, rotations, defocus_values):
    index_of_best_match = res.get_index_of_params_match()
    location_of_best_match = res.get_location_of_best_match()
    mip_values = res.get_mip()

    for i in range(res.count):
        best_rotation_index = index_of_best_match[i] % rotations.shape[0]
        best_defocus_index = index_of_best_match[i] // rotations.shape[0]

        print(f"Data {i + 1}/{res.count}:")

        print("\tFound max at:", location_of_best_match[i])
        print("\tMax cross correlation:", mip_values[i][location_of_best_match[i]])
        print("\tPhi:", rotations[best_rotation_index][0])
        print("\tTheta:", rotations[best_rotation_index][1])
        print("\tPsi:", rotations[best_rotation_index][2])
            
        print("\tDefocus:", defocus_values[best_defocus_index])

    return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot real part
    im1 = axes[0].imshow(res.get_mip(), aspect='auto', cmap='viridis')
    axes[0].set_title("MIP")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    fig.colorbar(im1, ax=axes[0])

    # Plot imaginary part
    im2 = axes[1].imshow(res.get_best_index_array(), aspect='auto', cmap='viridis')
    axes[1].set_title("Best Param Index")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    fig.colorbar(im2, ax=axes[1])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

small_region = tm2d.OrientationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=160, # Default is 0
        phi_max=200, # Default is 360
        theta_min=70, # Default is 0 
        theta_max=90, # Default is 180
        psi_min=300, # Default is 0
        psi_max=340 # Default is 360
    )

medium_region = tm2d.OrientationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=100, # Default is 0
        phi_max=200, # Default is 360
        theta_min=70, # Default is 0 
        theta_max=100, # Default is 180
        #psi_min=300, # Default is 0
        #psi_max=340 # Default is 360
    )

big_region = tm2d.OrientationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=0, # Default is 0
        phi_max=360, # Default is 360
        theta_min=0, # Default is 0 
        theta_max=180, # Default is 180
        psi_min=0, # Default is 0
        psi_max=360 # Default is 360
    )

data_dim = 512

def pad_input(data):
    #padded = np.ones((data_dim, data_dim), dtype=np.float32) * np.mean(data)
    #padded[:data.shape[0], data.shape[1]:] = data
    return data

# A copy of the data folder can be found at:
# /BigData/Workspaces/shahar/data

template_atomic = tm2d.TemplateAtomic(
    (data_dim, data_dim),
    tu.load_coords_from_npz("data/parsed_5lks_LSU.npz")
)

# template_density = tm2d.TemplateDensity(
#     tu.load_density_from_mrc("data/bronwyn/parsed_5lks_LSU_sim_120.mrc"),
#     transformed=True
# )

# np.save("template_density.npy",
#     template_density.make_template(
#         [0, 0, 0], 1, 10000
#     ).read_real(0)[0]
# )

params_template = tm2d.make_ctf_params(
    HT = 300e3,
    Cs = 2.7e7,
    Cc = 2.7e7,
    energy_spread_fwhm = 0.3,
    accel_voltage_std = 0.07e-6,
    OL_current_std = 0.1e-6,
    beam_semiangle = 2.5e-6,
    johnson_std = 0,
    B = 0,
    amp_contrast = 0.07,
    lpp = 0,
    defocus=10000,
)

np.save("template_atomic.npy",
    template_atomic.make_template(
        [0, 0, 0], 1, params_template
    ).read_real(0)[0]
)

#np.save("template_atomic.npy", pad_input(np.load("data/bronwyn/image.npy")))

data_array = np.array(
    [
        # template_atomic.make_template(
        #     [53, 62, 190],
        #     1.056,
        #     13000, ctf_params=tm2d.get_ctf_params_set("krios")).read_real(0)[0]
        tu.whiten_image(pad_input(np.load("data/bronwyn/image.npy"))),
        #pad_input(np.load("data/bronwyn/image.npy"))[::-1, :],
        #pad_input(np.load("data/bronwyn/image.npy"))[:, ::-1],
        #pad_input(np.load("data/bronwyn/image.npy"))[::-1, ::-1]
    ]
)

comparator = tm2d.ComparatorCrossCorrelation(
    data_array.shape, # shape of the micrographs
    template_atomic.get_shape() # shape of the template
)

results = tm2d.ResultsPixel(data_array.shape)

#results = tm2d.ResultsParam(
#    2, # batch size
#    rotations.shape[0] * defocus_values.shape[0] # total number of parameters
#)

params = tm2d.make_ctf_params(
    HT = 300e3,
    Cs = 2.7e7,
    Cc = 2.7e7,
    energy_spread_fwhm = 0.3,
    accel_voltage_std = 0.07e-6,
    OL_current_std = 0.1e-6,
    beam_semiangle = 2.5e-6,
    johnson_std = 0,
    B = None,
    amp_contrast = 0.07,
    lpp = None,
)

plan = tm2d.Plan(
    template_atomic, # template_density,
    comparator,
    results,
    pixel_size=1.056, # pixel size in Angstroms
    ctf_params=params,
    template_batch_size=2
)

plan.set_data(data_array)

# Get rotation list with cube sampling
rotations = tm2d.get_orientations_cube( # Get (phi, theta, psi) angles with cube sampling
    angular_step_size=2, # out of plane rotation step size
    psi_step_size=1, # in plane rotation step size
    region=small_region
)

ctf_params_dict = {
    "defocus": np.arange(12500, 13500, 10),
    "lpp": np.array([0, 90]),
    "B": np.arange(0, 100, 10), # B factor values
}

total_ctf_count = np.prod([len(v) for v in ctf_params_dict.values()])

plan.run(
    rotations,
    ctf_params_dict,
    enable_progress_bar=True
)

for i in range(results.count):
    np.save(f"mip{i}.npy", results.get_mip()[i])
    np.save(f"Z_score{i}.npy", results.get_z_score()[i])
    
    #np.save(f"best_defocus{i}.npy", results.get_best_index_array()[i] // rotations.shape[0])
    #np.save(f"best_rotation{i}.npy", results.get_best_index_array()[i] % rotations.shape[0])
    #np.save(f"sum_cross{i}.npy", results.get_sum_cross()[i])
    #np.save(f"sum2_cross{i}.npy", results.get_sum2_cross()[i])
    #np.save(f"cross_mean{i}.npy", results.get_cross_mean()[i])
    #np.save(f"cross_variance{i}.npy", results.get_cross_variance()[i])

    print(f"Micrograph {i + 1}:")
    print(f"\tBest rotation index: {results.get_index_of_params_match()[i]}")
    print(f"\tBest location: {results.get_location_of_best_match()[i]}")
    print(f"\tMax cross-correlation: {results.get_mip()[i][results.get_location_of_best_match()[i]]}")
    print(f"\tBest rotation: {rotations[results.get_index_of_params_match()[i] // total_ctf_count]}")
    best_index = results.get_index_of_params_match()[i] % total_ctf_count
    print(f"\tBest index: {best_index}")
    print(f"\tBest params: {params.get_values_at_index(ctf_params_dict, best_index)}")

exit()

# 360 is the number of psi angles, which for medium_region is 360 / 1 = 360
param_list = results.get_param_list(rotations, defocus_values, 360)

# param list has dimensions
# (
#   MICROGRAPH_COUNT,
#   DEFOCUS_COUNT,
#   OUT_OF_PLANE_ANGLES,
#   IN_PLANE_ANGLES,
#   5 # (phi, theta, psi, defocus, max_cross_correlation)
# )

# Print the best parameter (across all angles and defoci) for each micrograph
for i in range(param_list.shape[0]):
    best_param = param_list[i].reshape(-1, 5)[np.argmax(param_list[i, :, :, :, 4])]
    print(f"Micrograph {i + 1}:")
    print(f"\tBest rotation: (phi={best_param[0]:.2f}, theta={best_param[1]:.2f}, psi={best_param[2]:.2f})")
    print(f"\tBest defocus: {best_param[3]:.2f}")
    print(f"\tMax cross-correlation: {best_param[4]:.2f}")