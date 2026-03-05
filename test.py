import vkdispatch as vd
import tm2d
import numpy as np

import sys

import tm2d.utilities as tu

from matplotlib import pyplot as plt

#vd.initialize(debug_mode=True)
#vd.make_context(multi_device=True, multi_queue=True)

small_region = tm2d.OrientationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=180,
        phi_max=190,
        theta_min=70,
        theta_max=80,
        psi_min=310,
        psi_max=330
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
        tu.whiten_image(np.load("data/bronwyn/image.npy"), double_whiten=True),
    ]
)

comparator = tm2d.ComparatorCrossCorrelation(
    data_array.shape,
    template_atomic.get_shape()
)

results = tm2d.ResultsPixel(data_array.shape)

plan = tm2d.Plan(
    template_atomic,
    comparator,
    results,
    ctf_params=tm2d.CTFParams.like_krios(
        defocus = None,
        B = None,
        Cs = 2.7e7
    ),
    whiten_template=False, #True,
    template_batch_size=4,
)

plan.set_data(data_array)

rotations = tm2d.get_orientations_cube(
    angular_step_size=1,
    psi_step_size=0.5,
    region=small_region
)

params = plan.make_param_set(
    rotations=rotations,
    pixel_sizes = np.arange(1.05, 1.06, 0.001),
    defocus = np.arange(12850, 12950, 5),
    B = np.arange(0, 50, 5),
)

plan.run(params, enable_progress_bar=True)

for i in range(results.count):
    np.save(f"{output_dir}/template{i}.npy", plan.template_buffer.read_real(0)[i])
    np.save(f"{output_dir}/comparison{i}.npy", plan.comparison_buffer.read_real(0)[i])

    np.save(f"{output_dir}/mip{i}.npy", results.get_mip()[i])
    np.save(f"{output_dir}/Z_score{i}.npy", results.get_z_score()[i])

    match_index = results.get_index_of_params_match()[i]
    match_location = results.get_location_of_best_match()[i]

    best_indicies = results.get_best_index_array()[i]

    print(f"Micrograph {i}:")
    print(f"\tMax cross-correlation: {results.get_mip()[i][match_location]}")

    for param_name, param_value in params.get_values_at_index(match_index).items():
        print(f"\tBest {param_name}: {param_value}")

    for param_name, param_values in params.get_values_at_index(best_indicies).items():
        np.save(f"{output_dir}/{param_name}_{i}.npy", param_values)