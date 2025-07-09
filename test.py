import vkdispatch as vd
import tm2d
import numpy as np

import sys

import tm2d.utilities as tu

from matplotlib import pyplot as plt

#vd.initialize(debug_mode=True)
vd.make_context(max_streams=True)

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
        psi_min=300, # Default is 0
        psi_max=340 # Default is 360
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

# A copy of the data folder can be found at:
# /BigData/Workspaces/shahar/data

output_dir = sys.argv[1]

template_atomic = tm2d.TemplateAtomic(
    (512, 512),
    tu.load_coords_from_npz("data/parsed_5lks_LSU.npz")
)

data_array = np.array(
    [
        tu.whiten_image(np.load("data/bronwyn/image.npy")),
    ]
)

comparator = tm2d.ComparatorCrossCorrelation(
    data_array.shape, # shape of the micrographs
    template_atomic.get_shape() # shape of the template
)

results = tm2d.ResultsPixel(data_array.shape)

params = tm2d.CTFParams.like_krios(
    defocus = None, # 12870
    B = 27.5,
    Cs = None
)

plan = tm2d.Plan(
    template_atomic,
    comparator,
    results,
    #rotation=[188.84183,  78.82107, 326],
    #pixel_size=1.056, # pixel size in Angstroms
    ctf_params=params,
    template_batch_size=2
)

plan.set_data(data_array)

# Get rotation list with cube sampling
rotations = tm2d.get_orientations_cube( # Get (phi, theta, psi) angles with cube sampling
    angular_step_size=2, # out of plane rotation step size
    psi_step_size=1, # in plane rotation step size
    region=small_region # region of interest for the orientations
)

ctf_set = params.make_ctf_set(
    defocus = np.arange(12600, 12900, 2),
    Cs = np.arange(1.9e7, 2.2e7, 5e4),
    #B = np.arange(0, 500, 2.5),
)

pixel_sizes = np.arange(1.006, 1.086, 0.0025) # pixel sizes in Angstroms

plan.run(
    ctf_set,
    rotations=rotations,
    pixel_sizes=pixel_sizes,
    enable_progress_bar=True
)

print("Index of best match:", results.get_index_of_params_match()[0])
print("Ctf Length:", ctf_set.get_length())
print("Best rotation index:", results.get_index_of_params_match()[0] // ctf_set.get_length())
print("Best CTF index:", results.get_index_of_params_match()[0] % ctf_set.get_length())

best_rotations = rotations[results.get_rotation_indicies(ctf_set.get_length(), pixel_sizes.shape[0])]
best_ctf_values = ctf_set.get_values_at_index(results.get_ctf_indicies(ctf_set.get_length()))

for i in range(results.count):
    np.save(f"{output_dir}/mip{i}.npy", results.get_mip()[i])
    np.save(f"{output_dir}/Z_score{i}.npy", results.get_z_score()[i])

    print(f"Micrograph {i + 1}:")
    print(f"\tMax cross-correlation: {results.get_mip()[i][results.get_location_of_best_match()[i]]}")
    #print(f"\tBest rotation: {rotations[results.get_index_of_params_match()[i] // ctf_set.get_length()]}")

    best_ctf_index = results.get_index_of_ctf_match(ctf_set.get_length()) #results.get_index_of_params_match()[i] % ctf_set.get_length()

    print(f"\tBest CTF: {ctf_set.get_values_at_index(best_ctf_index)}")

    best_pixel_size_index = results.get_index_of_pixel_size_match(ctf_set.get_length(), pixel_sizes.shape[0])[i]

    print(f"\tBest pixel size index: {best_pixel_size_index}")
    print(f"\tBest pixel size: {pixel_sizes[best_pixel_size_index]}")

    best_rotation_index = results.get_index_of_rotation_match(ctf_set.get_length(), pixel_sizes.shape[0])[i]

    print(f"\tBest rotation index: {best_rotation_index}")
    print(f"\tBest rotation: {rotations[best_rotation_index]}")

    #print(rotations.shape)
    #print(results.get_rotation_indicies(ctf_set.get_length()).shape)
    #print(best_rotations.shape)

    np.save(f"{output_dir}/phi_{i + 1}.npy", best_rotations[i, :, :, 0])
    np.save(f"{output_dir}/theta_{i + 1}.npy", best_rotations[i, :, :, 1])
    np.save(f"{output_dir}/psi_{i + 1}.npy", best_rotations[i, :, :, 2])

    for k, v in best_ctf_values.items():
        np.save(f"{output_dir}/{k}_{i + 1}.npy", v[i])