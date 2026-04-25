import tm2d
import numpy as np

import sys

import tm2d_utils as tu

from matplotlib import pyplot as plt

def save_arr_as_png(arr: np.ndarray, title: str, filename: str):
    plt.clf()
    plt.imshow(arr)
    plt.title(title)
    plt.colorbar()
    plt.savefig(filename)

#vd.initialize(debug_mode=True)
#vd.make_context(multi_device=True, multi_queue=True)

small_region = tu.OrientationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=170,
        phi_max=200,
        theta_min=70,
        theta_max=90,
        psi_min=300,
        psi_max=330
    )

rotations = tu.get_orientations_cube(
    angular_step_size=2,
    psi_step_size=1,
    region=small_region
)

ctf_params = tm2d.CTFParams.like_krios(
    defocus = None,
    B = None,
    Cs = 2.7e7
)

params = tm2d.ParamSet.from_params(
    rotations=rotations,
    pixel_sizes=np.arange(1.056, 1.059, 0.002),
    ctf_set=ctf_params.make_ctf_set(
        defocus = np.arange(12870, 12900, 2),
        B = np.arange(4, 10, 2)
    )
)

# A copy of the data folder can be found at:
# /BigData/Workspaces/shahar/data

output_dir = sys.argv[1]

template_atomic = tm2d.TemplateAtomic(
    (512, 512),
    #(576, 576),
    tu.load_coords_from_npz("data/parsed_5lks_LSU.npz"),
    # fuse_ctf_convolution=True,
)

# density = tu.load_density_from_mrc("data/parsed_5lks_LSU_sim_120.mrc")
# template_atomic = tm2d.TemplateDensity(density.density, density.pixel_size)

data_array = np.array(
    [
        tu.whiten_image(np.load("data/bronwyn/image.npy"), double_whiten=True),
    ]
)

comparator = tm2d.ComparatorCrossCorrelation(
    data_array.shape,
    template_atomic.get_shape()
)

#results = tm2d.ResultsPixel(data_array.shape)
results = tm2d.ResultsParam(data_array.shape[0], params.get_total_count())

plan = tm2d.Plan(
    template_atomic,
    comparator,
    results,
    ctf_params=ctf_params,
    template_batch_size=4,
)

plan.set_data(data_array)

plan.run(params, enable_progress_bar=True)

zscore_list = results.get_zscore_list(params)

best_index = np.argmax(zscore_list)
values = params.get_values_at_index(best_index)

print(values)

values = params.get_values_tensor(zscore_list)

print(params.get_tensor_axes_names())

# for i in range(results.micrograph_count):
#     np.save(f"{output_dir}/template{i}.npy", plan.template_buffer.read_real(0)[i])
#     np.save(f"{output_dir}/comparison{i}.npy", plan.comparison_buffer.read_real(0)[i])

#     np.save(f"{output_dir}/mip{i}.npy", results.get_mip()[i])

#     z_score = tu.get_pixel_z_scores(results)[i]

#     np.save(f"{output_dir}/Z_score{i}.npy", z_score)

#     match_locations, match_indicies = tu.get_locations_and_indicies_of_best_match(results)

#     best_indicies = results.get_best_index_array()[i]

#     print(f"Micrograph {i}:")
#     print(f"\tMax cross-correlation: {results.get_mip()[i][match_locations[i]]}")

#     for param_name, param_value in params.get_values_at_index(match_indicies[i]).items():
#         print(f"\tBest {param_name}: {param_value}")

#     for param_name, param_values in params.get_values_at_index(best_indicies).items():
#         np.save(f"{output_dir}/{param_name}_{i}.npy", param_values)

"""

VKDISPATCH_BACKEND=vulkan python3 test.py data/output_sum

Micrograph 0:
        Max cross-correlation: 21996.779296875
        Best B: 8
        Best defocus: 12894
        Best pixel_size: 1.058
        Best rotation: [188.84183  78.82107 326.     ]

"""