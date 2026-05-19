import tm2d
import numpy as np

import sys

import tm2d_utils as tu

pixel_sizes = np.arange(1.04, 1.08, 0.0004)
B_factors = np.arange(0, 250, 0.25)

params = tm2d.make_param_set(
    tm2d.make_ctf_set(
        tu.ctf_like_krios(
            defocus = 12890,
            B = None,
            Cs = 2.7e7
        ),
        B = B_factors
    ),
    rotations=np.array([[188.84183,  78.82107, 326]]),
    #rotations_weights=np.array([1.0]),
    pixel_sizes=pixel_sizes,
)

# A copy of the data folder can be found at:
# /BigData/Workspaces/shahar/data

template_type = sys.argv[1]

micrographs= np.array([tu.whiten_image(np.load("data/bronwyn/image.npy"), double_whiten=True)])

#micrographs = np.ones((1, 1024, 1024), dtype=np.float32)

if template_type == "atomic":
    results = tu.run_tm2d_atomic_pixels(
        micrographs=micrographs,
        param_set=params,
        template_box_size=(256, 256),
        atomic_coords=tu.load_coords_from_npz("data/parsed_5lks_LSU.npz"),
        output_radius=3,
        enable_progress_bar=True
    )
elif template_type == "density":
    results = tu.run_tm2d_density_pixels(
        micrographs=micrographs,
        param_set=params,
        density=tu.load_density_from_mrc("data/parsed_5lks_LSU_sim_120.mrc"),
        output_radius=16,
        enable_progress_bar=True
    )
else:
    raise ValueError(f"Unknown template type: {template_type}")

output_dir = "."

for i in range(results.micrograph_count):
    #np.save(f"{output_dir}/template{i}.npy", plan.template_buffer.read_real(0)[i])
    # np.save(f"{output_dir}/comparison{i}.npy", plan.comparison_buffer.read_real(0)[i])

    np.save(f"{output_dir}/mip{i}.npy", results.get_mip()[i])

    z_score = tu.get_pixel_z_scores(results)[i]

    np.save(f"{output_dir}/Z_score{i}.npy", z_score)

    match_locations, match_indicies = tu.get_locations_and_indicies_of_best_match(results)

    best_indicies = results.get_best_index_array()[i]

    print(f"Micrograph {i}:")
    print(f"\tMax cross-correlation: {results.get_mip()[i][match_locations[i]]}")
    print(f"\tBest Z-score: {z_score[match_locations[i]]}")

    for param_name, param_value in params.get_values_at_index(match_indicies[i]).items():
        print(f"\tBest {param_name}: {param_value}")

    for param_name, param_values in params.get_values_at_index(best_indicies).items():
        np.save(f"{output_dir}/{param_name}_{i}.npy", param_values)