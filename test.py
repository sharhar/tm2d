import tm2d
import numpy as np

import sys

import tm2d_utils as tu

from matplotlib import pyplot as plt

ctf_params = tm2d.CTFParams.like_krios(
    defocus = 12890,
    B = None,
    Cs = 2.7e7
)

pixel_sizes = np.arange(1.04, 1.08, 0.00004)
B_factors = np.arange(0, 250, 0.25)

params = tm2d.ParamSet.from_params(
    rotations=np.array([[188.84183,  78.82107, 326]]),
    pixel_sizes=pixel_sizes,
    ctf_set=ctf_params.make_ctf_set(
        B = B_factors
    )
)

# A copy of the data folder can be found at:
# /BigData/Workspaces/shahar/data

template_type = sys.argv[1]

if template_type == "atomic":
    template_obj = tm2d.TemplateAtomic(
        (576, 576),
        tu.load_coords_from_npz("data/parsed_5lks_LSU.npz"),
    )
elif template_type == "density":
    density = tu.load_density_from_mrc("data/parsed_5lks_LSU_sim_120.mrc")
    template_obj = tm2d.TemplateDensity(density.density, density.pixel_size)
else:
    raise ValueError(f"Unknown template type: {template_type}")

data_array = np.array(
    [
        tu.whiten_image(np.load("data/bronwyn/image.npy"), double_whiten=True),
    ]
)

comparator = tm2d.ComparatorCrossCorrelation(
    data_array.shape,
    template_obj.get_shape()
)

#results = tm2d.ResultsPixel(data_array.shape)
results = tm2d.ResultsParam(data_array.shape[0], params.get_total_count())

plan = tm2d.Plan(
    template_obj,
    comparator,
    results,
    ctf_params=ctf_params,
    template_batch_size=4,
)

plan.set_data(data_array)

plan.run(params, enable_progress_bar=True)

zscore_list = results.get_zscore_list(params)
mip_list = results.get_mip_list(params)

plt.imshow(
    zscore_list[0][0],
    extent=[
        min(B_factors), max(B_factors),
        min(pixel_sizes), max(pixel_sizes)
    ],
    aspect="auto",
    origin="lower"
)
plt.colorbar()
plt.title(f"Z-scores {template_type}")
plt.xlabel("B Factor")
plt.ylabel("Pixel Size")
plt.savefig(f"zscore_{template_type}.png")
plt.show()
np.save(f"zscore_{template_type}.npy", zscore_list[0][0])

plt.imshow(
    mip_list[0][0],
    extent=[
        min(B_factors), max(B_factors),
        min(pixel_sizes), max(pixel_sizes)
    ],
    aspect="auto",
    origin="lower"
)
plt.colorbar()
plt.title(f"MIPs {template_type}")
plt.xlabel("B Factor")
plt.ylabel("Pixel Size")
plt.savefig(f"mip_{template_type}.png")
plt.show()
np.save(f"mip_{template_type}.npy", mip_list[0][0])

print(mip_list.shape)

best_index = np.argmax(zscore_list)
values = params.get_values_at_index(best_index)

print(values)

values = params.get_values_tensor(zscore_list)

print(values[0].shape)
print(values[1])

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