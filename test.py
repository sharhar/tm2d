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

plan = tm2d.Plan(
    template_atomic,
    comparator,
    results,
    rotation=[188.84183,  78.82107, 326],
    #pixel_size=1.056, # pixel size in Angstroms
    ctf_params=tm2d.CTFParams.like_krios(
        defocus = None, # 12870
        B = 27.5,
        Cs = None
    ),
    template_batch_size=2
)

plan.set_data(data_array)

# Get rotation list with cube sampling
rotations = tm2d.get_orientations_cube( # Get (phi, theta, psi) angles with cube sampling
    angular_step_size=2, # out of plane rotation step size
    psi_step_size=1, # in plane rotation step size
    region=small_region # region of interest for the orientations
)

params = plan.make_param_set(
    #rotations=rotations,
    pixel_sizes=np.arange(1.046, 1.066, 0.0001),

    defocus = np.arange(12600, 12900, 5),
    Cs = np.arange(1.9e7, 2.2e7, 1e4),
    #B = np.arange(0, 500, 2.5),
)

plan.run(params, enable_progress_bar=True)

for i in range(results.count):
    #np.save(f"{output_dir}/mip{i}.npy", results.get_mip()[i])
    #np.save(f"{output_dir}/Z_score{i}.npy", results.get_z_score()[i])

    match_index = results.get_index_of_params_match()[i]
    match_location = results.get_location_of_best_match()[i]

    print(f"Micrograph {i + 1}:")
    print(f"\tMax cross-correlation: {results.get_mip()[i][match_location]}")

    for param_name, param_value in params.index_to_values(match_index).items():
        print(f"\tBest {param_name}: {param_value}")

    #print(f"\tBest params: {params.index_to_values(match_index)}")

    #np.save(f"{output_dir}/phi_{i + 1}.npy", best_rotations[i, :, :, 0])
    #np.save(f"{output_dir}/theta_{i + 1}.npy", best_rotations[i, :, :, 1])
    #np.save(f"{output_dir}/psi_{i + 1}.npy", best_rotations[i, :, :, 2])

    #for k, v in best_ctf_values.items():
    #    np.save(f"{output_dir}/{k}_{i + 1}.npy", v[i])