import os
import tm2d
import numpy as np

import sys

import tm2d_utils as tu

from matplotlib import pyplot as plt

pixel_sizes = np.arange(1.04, 1.08, 0.0004)
B_factors = np.arange(0, 250, 2.5)

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
    pixel_sizes=pixel_sizes,
)

# A copy of the data folder can be found at:
data_folder = '/BigData/Workspaces/shahar/data'

template_type = sys.argv[1]

if template_type == "atomic":
    template_obj = tm2d.TemplateAtomic(
        (512, 512),
        tu.load_coords_from_npz(os.path.join(data_folder, "parsed_5lks_LSU.npz")),
    )
elif template_type == "density":
    density = tu.load_density_from_mrc(os.path.join(data_folder, "bronwyn/parsed_5lks_LSU_sim_120.mrc"))
    template_obj = tm2d.TemplateDensity(density.density, density.pixel_size)
else:
    raise ValueError(f"Unknown template type: {template_type}")

data_array = np.array(
    [
        tu.whiten_image(np.load(os.path.join(data_folder, "bronwyn/image.npy")), double_whiten=True),
    ]
)

comparator = tm2d.ComparatorCrossCorrelation(
    data_array.shape,
    template_obj.get_shape()
)

results = tm2d.ResultsPixel(data_array.shape, output_radius=4)
# results = tm2d.ResultsParam(data_array.shape[0], params.get_total_count(), output_radius=4)

plan = tm2d.Plan(
    template_obj,
    comparator,
    results,
    ctf_params=params.ctf_set.ctf_params,
    template_batch_size=4,
    output_radius=4,
    #enable_rotation_weights=True
)

plan.set_data(data_array)

plan.run(params, enable_progress_bar=True)

output_dir = "."

for i in range(results.micrograph_count):
    np.save(f"{output_dir}/template{i}.npy", plan.template_buffer.read_real(0)[i])
    np.save(f"{output_dir}/comparison{i}.npy", plan.comparison_buffer.read_real(0)[i])

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


    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(results.get_mip()[i], cmap='gray')
    ax[1].imshow(z_score, cmap='gray')
    fig.savefig(f"{output_dir}/results_{i}.png")


"""

VKDISPATCH_BACKEND=vulkan python3 test.py data/output_sum

Micrograph 0:
        Max cross-correlation: 21996.779296875
        Best B: 8
        Best defocus: 12894
        Best pixel_size: 1.058
        Best rotation: [188.84183  78.82107 326.     ]


Micrograph 0:
        Max cross-correlation: 11009.435546875
        Best Z-score: 2.442804562981274
        Best B: 10.0
        Best pixel_size: 1.058399999999998
        Best rotation: [188.84183  78.82107 326.     ]

"""