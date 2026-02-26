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

template_ex = template_atomic.make_template(
    rotations=np.array([[188.84183,  78.82107, 326]]), # (phi, theta, psi) angles for the template
    pixel_size=1.066, # pixel size in Angstroms
    ctf_params=tm2d.CTFParams.like_krios(
        defocus=12750,
        B=475,
        Cs=2.7e7
    )
)

OUTPUT_DIR = "data/results/"

temp_dat = template_ex.read_real(0)[0]

plt.clf()
plt.imshow(temp_dat)
plt.colorbar()
plt.title("Example template")
plt.savefig(f"{OUTPUT_DIR}/example_template.png")
np.save(f"{OUTPUT_DIR}/example_template.npy", temp_dat)

template_ex2 = template_atomic.make_template(
    rotations=np.array([[198.43495,  88.792274, 300.]]), # (phi, theta, psi) angles for the template
    pixel_size=1.046, # pixel size in Angstroms
    ctf_params=tm2d.CTFParams.like_krios(
        defocus=12700,
        B=0,
        Cs=2.7e7
    )
)

temp_dat2 = template_ex2.read_real(0)[0]

plt.clf()
plt.imshow(temp_dat2)
plt.colorbar()
plt.title("Example template2")
plt.savefig(f"{OUTPUT_DIR}/example_template2.png")
np.save(f"{OUTPUT_DIR}/example_template2.npy", temp_dat2)


data_array = np.array(
    [
        tu.whiten_image(np.load("data/bronwyn/image.npy")),
    ]
)

arr = tm2d.generate_ctf(
    (512, 512),
    1.056,
    tm2d.CTFParams.like_krios(
        defocus=12870,
        B=0,
        Cs=2.7e7
    )
)

plt.clf()
plt.imshow(arr)
plt.colorbar()
plt.savefig(f"{OUTPUT_DIR}/example_ctf.png")
np.save(f"{OUTPUT_DIR}/example_ctf.npy", arr)

# exit()

comparator = tm2d.ComparatorCrossCorrelation(
    data_array.shape, # shape of the micrographs
    template_atomic.get_shape() # shape of the template
)

comparator.set_data(data_array)

comparison = comparator.compare_template(template_ex)

comp_data = np.fft.fftshift(comparison.read_real(0)[0])

plt.clf()
plt.imshow(comp_data)
plt.colorbar()
plt.title("Example comparison")
plt.savefig(f"{OUTPUT_DIR}/example_comparison.png")
np.save(f"{OUTPUT_DIR}/example_comparison.npy", comp_data)

comparison2 = comparator.compare_template(template_ex2)

comp_data2 = np.fft.fftshift(comparison2.read_real(0)[0])

plt.clf()
plt.imshow(comp_data2)
plt.colorbar()
plt.title("Example comparison2")
plt.savefig(f"{OUTPUT_DIR}/example_comparison2.png")
np.save(f"{OUTPUT_DIR}/example_comparison2.npy", comp_data2)

results = tm2d.ResultsPixel(data_array.shape)
# results = tm2d.ResultsParam(data_array.shape[0], 5000000)

plan = tm2d.Plan(
    template_atomic,
    comparator,
    results,
    #rotation=[188.84183,  78.82107, 326],
    #pixel_size=1.056, # pixel size in Angstroms
    ctf_params=tm2d.CTFParams.like_krios(
        defocus = None, # 12870
        B = None,
        Cs = 2.7e7
    ),
    whiten_template=True,
    template_batch_size=4,
    #pixel_size=1.066
)

plan.set_data(data_array)

# Get rotation list with cube sampling
rotations = tm2d.get_orientations_cube( # Get (phi, theta, psi) angles with cube sampling
    angular_step_size=2, # out of plane rotation step size
    psi_step_size=1, # in plane rotation step size
    region=small_region # region of interest for the orientations
)

params = plan.make_param_set(
    rotations=rotations,
    pixel_sizes = np.arange(1.046, 1.066, 0.01),

    defocus = np.arange(12700, 13000, 25),
    # Cs = np.arange(1.9e7, 2.2e7, 1e4),
    B = np.arange(0, 500, 25),
)

plan.run(params, enable_progress_bar=True)

for i in range(results.count):

    plt.clf()
    plt.imshow(plan.template_buffer.read_real(0)[i])
    plt.title(f"Template {i}")
    plt.colorbar()
    plt.savefig(f"{OUTPUT_DIR}/template_{i}.png")
    np.save(f"{OUTPUT_DIR}/template_{i}.npy", plan.template_buffer.read_real(0)[i])

    plt.clf()
    plt.imshow(plan.comparison_buffer.read_real(0)[i])
    plt.title(f"Comparison {i}")
    plt.colorbar()
    plt.savefig(f"{OUTPUT_DIR}/comparison_{i}.png")
    np.save(f"{OUTPUT_DIR}/comparison_{i}.npy", plan.comparison_buffer.read_real(0)[i])

    plt.clf()
    plt.imshow(results.get_z_score()[i])
    plt.title(f"Z-score for micrograph {i}")
    plt.colorbar()
    plt.savefig(f"{OUTPUT_DIR}/z_score_{i}.png")
    np.save(f"{OUTPUT_DIR}/z_score_{i}.npy", results.get_z_score()[i])

    plt.clf()
    plt.imshow(results.get_mip()[i])
    plt.title(f"Max intensity projection for micrograph {i}")
    plt.colorbar()
    plt.savefig(f"{OUTPUT_DIR}/mip_{i}.png")
    np.save(f"{OUTPUT_DIR}/mip_{i}.npy", results.get_mip()[i])

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


"""

Micrograph 0:
        Max cross-correlation: 4653.39892578125
        Best B: 475
        Best defocus: 12750
        Best pixel_size: 1.066
        Best rotation: [188.84183  76.42089 326.     ]

Micrograph 0:
        Max cross-correlation: 4653.39892578125
        Best B: 475
        Best defocus: 12750
        Best pixel_size: 1.066
        Best rotation: [188.84183  76.42089 326.     ]

"""