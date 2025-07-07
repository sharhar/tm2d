import vkdispatch as vd
import tm2d
import numpy as np

import tm2d.utilities as tu

from matplotlib import pyplot as plt

micrograph_arrays = [np.load("data/bronwyn/image.npy"), np.load("data/bronwyn/image2.npy")[::-1, ::-1]]

# Create a command stream of the template matching algorithm
results, micrographs, search_args, template, correlations = tm2d.make_rotation_and_defoci_search_plan(
    (512, 512),
    tu.load_coords_from_npz("data/parsed_5lks_LSU.npz"),
    [micrograph.shape for micrograph in micrograph_arrays],
    1.056
)

# Get rotation list with cube sampling
rotations = tm2d.get_rotations_cube( # Get (phi, theta, psi) angles with cube sampling
    angular_step_size=2, # out of plane rotation step size
    psi_step_size=1, # in plane rotation step size
    region=tm2d.RotationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=100, # Default is 0
        phi_max=200, # Default is 360
        theta_min=70, # Default is 0 
        theta_max=100, # Default is 180
        #psi_min=300, # Default is 0
        #psi_max=340 # Default is 360
        )
    )

# Get defocus values
defocus_values = [12700] # tm2d.get_defocus_from_limits(12500, 12900, 100)

# Execute the template matching algorithm on our rotations and defocus values
tm2d.search_rotations_and_defoci(
    rotations, # Convert (phi, theta, psi) angles to cosine/sine vectors
    defocus_values,
    *search_args,
    enable_progress_bar=True
)

for res in results:
    best_rotation_index = res.get_index_of_params_match() % rotations.shape[0]
    best_defocus_index = res.get_index_of_params_match() // rotations.shape[0]

    print("Found max at:", res.get_location_of_best_match())
    print("Max cross correlation:", res.get_mip()[res.get_location_of_best_match()])
    print("Phi:", rotations[best_rotation_index][0])
    print("Theta:", rotations[best_rotation_index][1])
    print("Psi:", rotations[best_rotation_index][2])
        
    print("Defocus:", defocus_values[best_defocus_index])

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