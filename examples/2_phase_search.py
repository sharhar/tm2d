import os
import numpy as np
import glob
from itertools import chain
import time
import re

import diyplot as dp

import mrcfile

import vkdispatch as vd
import tm2d
import tm2d.utilities as tu

vd.make_context(max_streams=True)

coarse_oop_threshold_quantile = 0.95 # threshold for coarse oop selection
fine_thresh_deg_oop = 3 # proximity threshold for fine oop selection
fine_thresh_deg_ip = 1 # proximity threshold for fine psi selection
whiten = True

mrcs_dir = '/BigData/Workspaces/pnp/24_11_05_lpp_infocus_sim/24_11_17_sim_stack'
settings_fname = os.path.join(mrcs_dir, '24_11_17_settings_snr1_B20_dose30.npz')
results_dir = mrcs_dir + '_analysis_v3'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# mrcs_dir = '/BigData/Workspaces/pnp/24_11_05_lpp_infocus_sim/24_11_18_sim_stack_astig200'
# settings_fname = os.path.join(mrcs_dir, '24_11_18_settings_snr1_B20_dose30.npz')

mrcs_list = glob.glob(mrcs_dir + '/*.mrcs')
print('settings file: ', settings_fname)
settings = np.load(settings_fname)

image_rows = int(settings['image_rows'])
image_cols = image_rows
image_pix_size = settings['image_pix_size']
HT = settings['HT']
B_factor = settings['B_factor']
Cc = settings['Cc']
Cs = settings['Cs']
f_OL = settings['f_OL']
amp_contrast = settings['amp_contrast']
phase_shift = settings['phase_shift']
lpp_rot = settings['lpp_rot']

dose_per_A2 = settings['dose_per_A2']
snr = settings['snr']
A2_mag = settings['A2_mag']
A2_ang = settings['A2_ang']
wlen_L = settings['wlen_L']
defocus_range = settings['defocus_range']

protein_file_dir = '/home/shaharsandhaus/TEM_LPP_Image_Simulator/NPZs'
protein_file_name = '6z6u_apoferritin.npz'
particle_symmetry = 'O'
D = 130 # [A] particle diameter
atoms_npz_path = os.path.join(protein_file_dir, protein_file_name)
atom_coords = tu.load_coords_from_npz(atoms_npz_path)

#%%

def read_mrc(mrc_path):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        im = mrc.data
    return im

def make_matching_plan(ds_factor, image_shape, atom_coords, image_pix_size, HT, B_factor, Cc, Cs, f_OL, amp_contrast, phase_shift, lpp_rot, param_count):
    results, micrographs, search_args, template, correlations = tm2d.make_rotation_and_defoci_search_plan(
        (image_shape[0] // ds_factor, image_shape[1] // ds_factor),
        atom_coords,
        [(image_shape[0] // ds_factor, image_shape[1] // ds_factor)],
        image_pix_size * ds_factor,
        HT=HT,
        B=B_factor,
        Cc=Cc,
        Cs=Cs,
        f_OL=f_OL,
        amp_contrast=amp_contrast,
        do_param_output=True,
        param_count=param_count,
        lpp=phase_shift,
        lpp_rot=lpp_rot,
    )
    
    return results, micrographs, search_args, template, correlations

def get_defocus_step(d_ang, HT):
    wlen_e = tu.optics_functions.e_compton / (tu.get_gammaLorentz(HT) * tu.get_beta(HT)) # wavelength [A]
    return round(d_ang**2 / (2 * wlen_e) / 2) # [A]

def get_symmetric_arange(center, halfwidth, step):
    arr = np.concatenate(((center - np.arange(0, halfwidth + step, step)), (center + np.arange(0, halfwidth + step, step))))
    return np.sort(np.unique(arr))

def generate_octahedral_rotation_matrices():
    """
    Generate all 24 unique 3x3 rotation matrices for the octahedral symmetry group.

    Returns:
    - rotation_matrices: A list of 24 unique 3x3 numpy arrays.
    """
    def rotation_matrix(axis, angle):
        """Generate a rotation matrix for a given axis ('x', 'y', 'z') and angle."""
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'x':
            return np.array([[1, 0, 0],
                             [0, c, -s],
                             [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s],
                             [0, 1, 0],
                             [-s, 0, c]])
        elif axis == 'z':
            return np.array([[c, -s, 0],
                             [s, c, 0],
                             [0, 0, 1]])

    def generate_diagonal_rotations():
        """Generate rotations about diagonal axes (vertices and face centers)."""
        rotations = []
        diagonal_axes = [
            np.array([1, 1, 0]),
            np.array([1, -1, 0]),
            np.array([0, 1, 1]),
            np.array([0, 1, -1]),
            np.array([1, 0, 1]),
            np.array([1, 0, -1]),
        ]
        for axis in diagonal_axes:
            axis = axis / np.linalg.norm(axis)  # Normalize the axis
            c, s = np.cos(np.pi), np.sin(np.pi)  # 180° rotation
            x, y, z = axis
            matrix = np.array([
                [c + (1 - c) * x * x, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
                [(1 - c) * y * x + s * z, c + (1 - c) * y * y, (1 - c) * y * z - s * x],
                [(1 - c) * z * x - s * y, (1 - c) * z * y + s * x, c + (1 - c) * z * z]
            ])
            rotations.append(matrix)

        # Axes through vertices (120° and 240° rotations)
        vertex_axes = [
            np.array([1, 1, 1]),
            np.array([1, -1, 1]),
            np.array([-1, 1, 1]),
            np.array([-1, -1, 1]),
        ]
        for axis in vertex_axes:
            axis = axis / np.linalg.norm(axis)  # Normalize the axis
            for angle in [2 * np.pi / 3, 4 * np.pi / 3]:  # 120° and 240° rotations
                c, s = np.cos(angle), np.sin(angle)
                x, y, z = axis
                matrix = np.array([
                    [c + (1 - c) * x * x, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
                    [(1 - c) * y * x + s * z, c + (1 - c) * y * y, (1 - c) * y * z - s * x],
                    [(1 - c) * z * x - s * y, (1 - c) * z * y + s * x, c + (1 - c) * z * z]
                ])
                rotations.append(matrix)
        return rotations

    # Generate principal axis rotations
    rotation_matrices = [np.eye(3)]  # Identity matrix
    for axis in ['x', 'y', 'z']:
        for angle in [np.pi / 2, np.pi, 3 * np.pi / 2]:
            rotation_matrices.append(rotation_matrix(axis, angle))

    # Add diagonal rotations
    rotation_matrices.extend(generate_diagonal_rotations())

    # Deduplicate by rounding and checking equality
    unique_matrices = []
    for matrix in rotation_matrices:
        rounded_matrix = np.round(matrix, decimals=8)  # Avoid floating-point issues
        if not any(np.allclose(rounded_matrix, m) for m in unique_matrices):
            unique_matrices.append(rounded_matrix)

    return unique_matrices

rotation_matrices = generate_octahedral_rotation_matrices()

def generate_symmetric_points(phi_deg, theta_deg, rotation_matrices):
    """
    Generate all 24 symmetry-equivalent (phi, theta) points for the octahedral group.

    Parameters:
    - phi: Azimuthal angle in radians (0 <= phi < 2*pi)
    - theta: Polar angle in radians (0 <= theta <= pi)

    Returns:
    - symmetric_points: List of (phi, theta) tuples.
    """
    # Convert (phi, theta) to Cartesian coordinates
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    point = np.array([x, y, z])

    # Apply all 24 rotation matrices
    symmetric_points = []
    for matrix in rotation_matrices:
        rotated_point = np.dot(matrix, point)
        # Convert back to spherical coordinates
        r = np.linalg.norm(rotated_point)
        rotated_theta = np.arccos(rotated_point[2] / r)
        rotated_phi = np.arctan2(rotated_point[1], rotated_point[0])
        if rotated_phi < 0:
            rotated_phi += 2 * np.pi  # Ensure phi is in [0, 2*pi)
        symmetric_points.append((rotated_phi, rotated_theta))

    # Deduplicate and return
    symmetric_points = list(set((round(p, 8), round(t, 8)) for p, t in symmetric_points))
    symmetric_points.sort()
    return np.rad2deg(symmetric_points)

def get_oop_kept_inds(test_pt, ref_pts, thresh):
    test_norm = np.linalg.norm(test_pt - ref_pts, axis=1)
    kept_inds = np.where(test_norm < thresh)
    return kept_inds[0]

def get_culled_fine_oop_angles(coarse_oop_angles, fine_oop_domain, thresh_deg):
    really_all_kept_inds = []
    for oop_angles in coarse_oop_angles:
        phi, theta = oop_angles
        equiv_pts = generate_symmetric_points(phi, theta, rotation_matrices)
        all_kept_inds = []
        for equiv_pt in equiv_pts:
            kept_inds = get_oop_kept_inds(equiv_pt, fine_oop_domain, thresh_deg)
            all_kept_inds.extend(kept_inds.tolist())
        really_all_kept_inds.extend(all_kept_inds)
    really_all_kept_inds = np.unique(really_all_kept_inds)
    return fine_oop_domain[really_all_kept_inds]

def filter_points_above_sigma_vectorized(data, n_stds=3):
    row_means = np.mean(data, axis=1, keepdims=True)
    row_stds = np.std(data, axis=1, keepdims=True)
    mask = data > (row_means + n_stds * row_stds)
    filtered_indices = [np.where(row_mask)[0].tolist() for row_mask in mask]
    all_filtered_indices = list(chain.from_iterable(filtered_indices))
    return sorted(set(all_filtered_indices))

def get_culled_fine_ip_angles(candidate_psi, fine_psi_domain, thresh_deg):
    nearby_elements = fine_psi_domain[
        np.any(np.abs(fine_psi_domain[:, None] - candidate_psi[None, :]) <= thresh_deg, axis=1)
    ]
    return nearby_elements

def set_custom_rotation_search_space(matcher, out_of_plane_angles, psi_values):   
    rotation_grid = np.zeros((out_of_plane_angles.shape[0], psi_values.shape[0], 3), dtype=np.float32)
    rotation_grid[:, :, :2] = out_of_plane_angles[:, None, :]
    rotation_grid[:, :, 2] = psi_values[None, :]

    matcher.out_of_plane_angles = out_of_plane_angles
    matcher.psi_values = psi_values
    matcher.out_count = out_of_plane_angles.shape[0]
    matcher.in_count = psi_values.shape[0]
    matcher.rotations = rotation_grid.reshape(-1, 3)

def read_results(results):
    # find max mip across all samples
    max_index = np.argmax(results[:, :, :, 4])
    peak_inds = np.unravel_index(max_index, results.shape[:3]) # unravel index

    # get parameter estimates
    phi_est = results[peak_inds][0]
    theta_est = results[peak_inds][1]
    psi_est = results[peak_inds][2]
    defocus_est = results[peak_inds][3]
    mip_max = results[peak_inds][4]

    return (phi_est, theta_est, psi_est, defocus_est, mip_max)

def plot_tri_grid(ax, x, y, z, n_levels=100):
    import matplotlib.tri as mtri

    triang = mtri.Triangulation(x, y)
    han = ax.tricontourf(triang, z, levels=n_levels)
    ax.triplot(triang,'k.-', linewidth = 0.1)

def plot_coarse_search_fig(out_of_plane_angles, in_plane_angles, defocus_values, mips_oop, mips_f_psi, pose_true, defocus_true):
    fig, ax = dp.subplots(ncols=2)

    plot_tri_grid(ax[0],
        out_of_plane_angles[:, 0],
        out_of_plane_angles[:, 1],
        mips_oop)
    ax[0].scatter(pose_true[0], pose_true[1], c='r', marker='x', s=100)
    ax[0].set_xlabel(r'$\phi$ [deg]')
    ax[0].set_ylabel(r'$\theta$ [deg]')

    plot_tri_grid(ax[1],
        np.repeat(defocus_values, len(in_plane_angles)),
        np.tile(in_plane_angles, len(defocus_values)),
        mips_f_psi.flatten())
    ax[1].scatter(defocus_true, pose_true[2], c='r', marker='x', s=100)
    ax[1].set_xlabel('defocus [A]')
    ax[1].set_ylabel(r'$\psi$ [deg]')

    fig.suptitle('coarse search results')
    fig.tight_layout()

    return fig, ax

def plot_domain_filtering(
    out_of_plane_angles, in_plane_angles, coarse_oop_inds, mips_oop, pose_true, coarse_oop_threshold_quantile,
    fine_oop_angles, fine_psi_values, reduced_fine_oop_domain, fine_thresh_deg_oop,
    mips_filtered, unique_elements, reduced_fine_ip_domain, fine_thresh_deg_ip):

    fig, ax = dp.subplots(ncols=3, nrows=2)
    ax = ax.flatten()

    ax[0].scatter(
        out_of_plane_angles[coarse_oop_inds, 0],
        out_of_plane_angles[coarse_oop_inds, 1],
        c=mips_oop[coarse_oop_inds], cmap='viridis')
    ax[0].scatter(pose_true[0], pose_true[1], c='r', marker='x', s=100)
    ax[0].set_title('coarse oop angles above {:.2f} quantile ({} total)'.format(
        coarse_oop_threshold_quantile, len(coarse_oop_inds)))
    ax[0].set_xlabel(r'$\phi$ [deg]')
    ax[0].set_ylabel(r'$\theta$ [deg]')

    ax[1].scatter(fine_oop_angles[:, 0], fine_oop_angles[:, 1], c='gray')
    ax[1].scatter(reduced_fine_oop_domain[:, 0], reduced_fine_oop_domain[:, 1], c='k')
    ax[1].scatter(pose_true[0], pose_true[1], c='r', marker='x', s=100, label=r'true ($\phi$, $\theta$)')
    ax[1].set_title('threshold: {} deg, angles kept: {} ({:0.2f}%)'.format(
        fine_thresh_deg_oop, len(reduced_fine_oop_domain),
        100 * len(reduced_fine_oop_domain) / len(fine_oop_angles)))
    ax[1].set_xlabel(r'$\phi$ [deg]')
    ax[1].set_ylabel(r'$\theta$ [deg]')
    ax[1].legend()

    dp.imshow_with_cbar(mips_filtered, fig=fig, ax=ax[2], cmap='viridis')
    ax[2].set_aspect(10)
    ax[2].set_xlabel('$\psi$ [deg]')
    ax[2].set_ylabel('oop angles')

    ax[3].plot(in_plane_angles, mips_filtered.sum(axis=0))
    ax[3].axvline(pose_true[2], c='r', label='true $\psi$')
    for idx in unique_elements:
        ax[3].axvline(in_plane_angles[idx], c='k', ls='--', lw=0.5)
    ax[3].set_xlabel('$\psi$ [deg]')
    ax[3].set_ylabel('mip sum over all oop angles')
    ax[3].legend()
    ax[3].set_title('best psi: {} total'.format(len(unique_elements)))

    ax[4].plot(in_plane_angles, mips_filtered.sum(axis=0))
    ax[4].axvline(pose_true[2], c='r', label='true $\psi$')
    for nearby_element in reduced_fine_ip_domain:
        ax[4].axvline(nearby_element, c='g', ls='--', lw=0.5)
    ax[4].set_xlabel('$\psi$ [deg]')
    ax[4].set_ylabel('mip sum over all oop angles')
    ax[4].set_title('threshold: {} deg, angles kept: {} ({:0.2f}%)'.format(
        fine_thresh_deg_ip, len(reduced_fine_ip_domain),
        100 * len(reduced_fine_ip_domain) / len(fine_psi_values)))
    ax[4].legend()

    ax[5].set_axis_off()

    fig.tight_layout()

    return fig, ax

def append_table(fig, ax,
    phi_est_coarse, theta_est_coarse, psi_est_coarse, defocus_est_coarse, t_coarse,
    phi_est_fine, theta_est_fine, psi_est_fine, defocus_est_fine, t_fine,
    pose_true, defocus_true):

    column1 = ['$\phi$ [deg]', r'$\theta$ [deg]', '$\psi$ [deg]', 'f [A]', 't [s]']
    column2 = [round(val, 2) for val in [phi_est_coarse, theta_est_coarse, psi_est_coarse, defocus_est_coarse, t_coarse]]
    column3 = [round(val, 2) for val in [phi_est_fine, theta_est_fine, psi_est_fine, defocus_est_fine, t_fine]]
    column4 = [round(val, 2) for val in [pose_true[0], pose_true[1], pose_true[2], defocus_true]] + [None]
    column5 = [round(val, 2) for val in [phi_est_fine - pose_true[0], theta_est_fine - pose_true[1], psi_est_fine - pose_true[2], defocus_est_fine - defocus_true]] + [None]

    # Combine data into rows
    table_data = list(zip(column1, column2, column3, column4, column5))

    # Create the table
    table = ax_filt[-1].table(
        cellText=table_data,  # Table data as rows
        colLabels=["param", "coarse", "fine", "true", "error"],  # Column headers
        cellLoc="left",  # Center-align text in cells
        loc="center"  # Place the table in the center of the axis
    )
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    return fig, ax

def get_m_ind(mrcs_fname):
    return int(re.search(r'batch(\d+)\.', mrcs_fname).group(1))

#%%

# make coarse search matcher
ds_factor_coarse = 4 # downsample factor
d_coarse = 2 * image_pix_size * ds_factor_coarse # resolution
ang_step_coarse = np.rad2deg(d_coarse / D) # [deg] angular step
defocus_list_coarse = [defocus_range[0], 0, defocus_range[1]] # [A]

coarse_out_of_plane_angles, coarse_in_plane_angles = tm2d.get_rotations_healpix(
    angular_step_size=ang_step_coarse,
    psi_step_size=ang_step_coarse,
    region='O',
    merge_out_and_in_plane=False
)

coarse_rotations = tm2d.make_rotation_array(coarse_out_of_plane_angles, coarse_in_plane_angles)

coarse_defocuses = tm2d.get_defocus_from_limits(defocus_range[0], defocus_range[1] + 1, np.abs(np.diff(defocus_range)) // 2) #, A2_mag=A2_mag, A2_ang=A2_ang)

coarse_results, coarse_micrographs, coarse_search_args, coarse_template, coarse_correlations = make_matching_plan(
    ds_factor_coarse,
    (image_rows, image_cols), 
    atom_coords, 
    image_pix_size, HT,B_factor, 
    Cc, Cs, f_OL, amp_contrast, 
    phase_shift, lpp_rot, 
    coarse_defocuses.shape[0] * coarse_rotations.shape[0]
)

print('coarse search')
print('... resolution: {:.2f} [A]'.format(d_coarse))
print('... angular step: {:.2f} [deg]'.format(ang_step_coarse))
print('... out-of-plane angles: {}'.format(coarse_out_of_plane_angles.shape[0]))
print('... in-plane angles: {}'.format(coarse_in_plane_angles.shape[0]))
print('... total poses: {}'.format(coarse_rotations.shape[0]))
print('... defocus steps: {}'.format(coarse_defocuses.shape[0]))

# make fine search matcher
ds_factor_fine = 1 # downsample factor
d_fine = 2 * image_pix_size * ds_factor_fine # resolution
ang_step_fine = np.rad2deg(d_fine / D) # [deg] angular step
defocus_step_fine = get_defocus_step(d_fine, HT=HT) # [A]
defocus_list_fine = get_symmetric_arange(0, np.max(defocus_range), defocus_step_fine) # [A]

fine_out_of_plane_angles, fine_in_plane_angles = tm2d.get_rotations_healpix(
    angular_step_size=ang_step_fine,
    psi_step_size=ang_step_fine,
    region='O',
    merge_out_and_in_plane=False
)

fine_rotations = tm2d.make_rotation_array(fine_out_of_plane_angles, fine_in_plane_angles)

fine_defocuses = tm2d.get_defocus_from_limits(defocus_list_fine[0], defocus_list_fine[-1] + 1, defocus_step_fine) #, A2_mag=A2_mag, A2_ang=A2_ang)

fine_oop_angles = fine_out_of_plane_angles
fine_psi_values = fine_in_plane_angles
fine_n_templates = fine_rotations.shape[0] * fine_defocuses.shape[0]

fine_results, fine_micrographs, fine_search_args, fine_template, fine_correlations = make_matching_plan(ds_factor_fine,
    (image_rows, image_cols), atom_coords, image_pix_size, HT, B_factor, Cc, Cs, f_OL, amp_contrast, phase_shift, lpp_rot, fine_n_templates
)

print('fine search')
print('... resolution: {:.2f} [A]'.format(d_fine))
print('... angular step: {:.2f} [deg]'.format(ang_step_fine))
print('... out-of-plane angles: {}'.format(fine_out_of_plane_angles.shape[0]))
print('... in-plane angles: {}'.format(fine_in_plane_angles.shape[0]))
print('... total poses: {}'.format(fine_rotations.shape[0]))
print('... defocus spacing: {:.2f} [A]'.format(defocus_step_fine))
print('... defocus steps: {}'.format(fine_defocuses.shape[0]))
print('... total templates: {}'.format(fine_n_templates))

#%%

first_m_ind = get_m_ind(mrcs_list[0])

for mrcs_fname in mrcs_list:
    print('processing: {}'.format(mrcs_fname))

    m_ind = get_m_ind(mrcs_fname)

    # load pose and defocus data
    data_fname = os.path.basename(mrcs_fname).replace('.mrcs', '.npz')
    data_fname = data_fname.replace('stack', 'data')
    data_fname = os.path.join(mrcs_dir, data_fname)
    print('params file: ', data_fname)
    print('mrcs file: ', mrcs_fname)
    data = np.load(data_fname)
    defocuses_true = data['defocuses_true']
    poses_true = data['poses_true']

    # load images
    ims = read_mrc(mrcs_fname)

    # prepare to store results
    defocuses_coarse = np.empty_like(defocuses_true)
    poses_coarse = np.empty_like(poses_true)
    defocuses_fine = np.empty_like(defocuses_true)
    poses_fine = np.empty_like(poses_true)
    mip_maxes_fine = np.empty(len(ims))
    n_templates_fine = np.empty(len(ims))
    times_coarse = np.empty(len(ims))
    times_filter = np.empty(len(ims))
    times_fine = np.empty(len(ims))
    times_housekeeping = np.empty(len(ims))

    results_fname = 'results_batch{}.npz'.format(m_ind)
    results_fpath = os.path.join(results_dir, results_fname)

    for i_ind, im in enumerate(ims):
        print('... image #{}/{}'.format(i_ind + 1, len(ims)))

        # read data
        pose_true = poses_true[i_ind]
        defocus_true = defocuses_true[i_ind]

        # prepare to make and save plots
        if i_ind < 5 and m_ind == first_m_ind:
            make_plots = True
            save_plots = True
            fig_title_str = '{} @ {}'.format(i_ind, os.path.basename(mrcs_fname))
            fig_coarse_fname = 'coarse_search_batch{}_i{}.png'.format(m_ind, i_ind)
            fig_coarse_fpath = os.path.join(results_dir, fig_coarse_fname)
            fig_filt_fname = 'domain_filtering_batch{}_i{}.png'.format(m_ind, i_ind)
            fig_filt_fpath = os.path.join(results_dir, fig_filt_fname)
        else:
            make_plots = False
            save_plots = False

        ### COARSE SEARCH ###################################################################################################
        t_coarse_start = time.time() # start timer

        # prepare reference image
        im_coarse = tu.downsample_image(im, image_rows // ds_factor_coarse) # downsample image
        
        coarse_micrographs[0].set_data(
            tu.process_raw_micrograph(im_coarse, whiten=whiten, normalize_input=True, remove_outliers=True, n_std=5)
        )

        # do template matching
        tm2d.search_rotations_and_defoci(
            coarse_rotations,
            coarse_defocuses,
            *coarse_search_args,
            enable_progress_bar=True
        )

        results_coarse = coarse_results[0].get_param_list(coarse_rotations, coarse_defocuses, coarse_in_plane_angles.shape[0])

        (phi_est_coarse, theta_est_coarse, psi_est_coarse, defocus_est_coarse, _) = read_results(results_coarse)

        # get mips needed for filtering
        mips = results_coarse[:, :, :, 4]
        mips_oop = mips.max(axis=2).max(axis=0) # max mip for each out-of-plane angle

        t_coarse = time.time() - t_coarse_start # end coarse search timer

        ### FILTER SEARCH SPACE #############################################################################################
        t_filter_start = time.time() # start timer

        # get indices of out-of-plane angles above threshold
        coarse_oop_inds = np.where(mips_oop > np.quantile(mips_oop, coarse_oop_threshold_quantile))[0]

        # reduce fine out-of-plane angle search space
        reduced_fine_oop_domain = get_culled_fine_oop_angles(
            coarse_out_of_plane_angles[coarse_oop_inds],
            fine_oop_angles,
            fine_thresh_deg_oop)

        # reduce fine in-plane angle search space
        mips_filtered = mips[:, coarse_oop_inds, :].max(axis=0) # mip(oop, psi) for good oop

        unique_elements = filter_points_above_sigma_vectorized(mips_filtered, n_stds=3)
        reduced_fine_ip_domain = get_culled_fine_ip_angles(
            coarse_in_plane_angles[unique_elements],
            fine_psi_values,
            fine_thresh_deg_ip)

        # results of domain reduction
        n_total_templates_reduced = reduced_fine_oop_domain.shape[0] * reduced_fine_ip_domain.shape[0] *\
            fine_defocuses.shape[0]

        t_filter = time.time() - t_filter_start # end filter search space timer

        ### FINE SEARCH #####################################################################################################
        t_fine_start = time.time() # start timer

        # prepare reference image
        if ds_factor_fine != 1:
            im_fine = tu.downsample_image(im, image_rows // ds_factor_fine) # downsample image
            fine_micrographs[0].set_data(
                tu.process_raw_micrograph(im_fine, whiten=whiten, normalize_input=True, remove_outliers=True, n_std=5)
            )
        else:
            fine_micrographs[0].set_data(
                tu.process_raw_micrograph(im, whiten=whiten, normalize_input=True, remove_outliers=True, n_std=5)
            )

        # make custom fine angular search space
        reduced_rotations = tm2d.make_rotation_array(reduced_fine_oop_domain, reduced_fine_ip_domain)
        
        # do template matching
        tm2d.search_rotations_and_defoci(
            reduced_rotations,
            fine_defocuses,
            *fine_search_args,
            enable_progress_bar=True
        )
        print('... reduced oop: {}'.format(reduced_fine_oop_domain.shape[0]))
        print('... reduced ip: {}'.format(reduced_fine_ip_domain.shape[0]))
        print('... defocus steps: {}'.format(fine_defocuses.shape[0]))
        print('... total templates: {}'.format(n_total_templates_reduced))
        results_fine = fine_results[0].get_param_list(reduced_rotations, fine_defocuses, reduced_fine_ip_domain.shape[0], true_size=n_total_templates_reduced)
        (phi_est_fine, theta_est_fine, psi_est_fine, defocus_est_fine, mip_max_fine) = read_results(results_fine)

        t_fine = time.time() - t_fine_start # end fine search timer

        ### HOUSEKEEPING ####################################################################################################
        t_housekeeping_start = time.time() # start timer

        # make and save figures
        if make_plots:
            mips_f_psi = mips.max(axis=1) # max mip for each in-plane angle and defocus
            fig_coarse, ax_coarse = plot_coarse_search_fig(coarse_out_of_plane_angles, coarse_in_plane_angles, coarse_defocuses, mips_oop, mips_f_psi, pose_true, defocus_true)
            fig_coarse.suptitle(fig_title_str)

            fig_filt, ax_filt = plot_domain_filtering(
                coarse_out_of_plane_angles, coarse_in_plane_angles, coarse_oop_inds, mips_oop, pose_true, coarse_oop_threshold_quantile,
                fine_oop_angles, fine_psi_values, reduced_fine_oop_domain, fine_thresh_deg_oop,
                mips_filtered, unique_elements, reduced_fine_ip_domain, fine_thresh_deg_ip)
            fig_filt, ax_filt = append_table(fig_filt, ax_filt,
                phi_est_coarse, theta_est_coarse, psi_est_coarse, defocus_est_coarse, t_coarse,
                phi_est_fine, theta_est_fine, psi_est_fine, defocus_est_fine, t_fine,
                pose_true, defocus_true)
            fig_filt.suptitle(fig_title_str)
            fig_filt.tight_layout()
        if make_plots and save_plots:
            fig_coarse.savefig(fig_coarse_fpath)
            fig_filt.savefig(fig_filt_fpath)

        # reset accumulators
        for cr in coarse_results:
            cr.reset()
        
        for fr in fine_results:
            fr.reset()

        t_housekeeping = time.time() - t_housekeeping_start # end housekeeping timer

        # store results
        defocuses_coarse[i_ind] = defocus_est_coarse
        poses_coarse[i_ind] = (phi_est_coarse, theta_est_coarse, psi_est_coarse)
        defocuses_fine[i_ind] = defocus_est_fine
        poses_fine[i_ind] = (phi_est_fine, theta_est_fine, psi_est_fine)
        mip_maxes_fine[i_ind] = mip_max_fine
        n_templates_fine[i_ind] = n_total_templates_reduced
        times_coarse[i_ind] = t_coarse
        times_filter[i_ind] = t_filter
        times_fine[i_ind] = t_fine
        times_housekeeping[i_ind] = t_housekeeping

    # save results
    np.savez(results_fpath,
        defocuses_coarse=defocuses_coarse,
        poses_coarse=poses_coarse,
        defocuses_fine=defocuses_fine,
        poses_fine=poses_fine,
        mip_maxes_fine=mip_maxes_fine,
        n_templates_fine=n_templates_fine,
        times_coarse=times_coarse,
        times_filter=times_filter,
        times_fine=times_fine,
        times_housekeeping=times_housekeeping,
    )
    print('results saved to: {}'.format(results_fpath))