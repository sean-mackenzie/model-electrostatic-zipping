# imports
import os
from os.path import join
import numpy as np
import pandas as pd
from utils import empirical, settings, materials
from utils.empirical import dict_from_tck, plot_final_tck_profile
from tests.test_manually_fit_tck_to_surface_profile import manually_fit_tck
from utils import model_zipping


# ------------------- WAFER + MEMBRANE SETTINGS

def set_up_model_directories_and_get_surface_profile(root_dir, test_config, wid, save_sub_dir,
                                                     dict_override=None, use_radial_displacement_settings=False):

    base_dir = join(root_dir, test_config)
    dict_surface = get_surface_profile_settings(wid, test_config, dict_override, use_radial_displacement_settings)
    # -
    analyses_dir = join(base_dir, 'analyses')
    read_settings = join(analyses_dir, 'settings')
    fp_settings = join(read_settings, 'dict_settings.xlsx')
    dict_settings = settings.get_settings(fp_settings=fp_settings, name='settings', update_dependent=False)
    fid_process_profile = dict_settings['fid_process_profile']
    fid = fid_process_profile
    dict_settings_radius = dict_settings['radius_microns']
    include_through_hole = True
    df_surface = empirical.read_surface_profile(
        dict_settings,
        subset='full',  # this should always be 'full', because profile will get sliced during tck
        hole=include_through_hole,
        fid_override=fid,
    )

    save_id = 'wid{}_fid{}'.format(wid, fid)
    save_dir = join(analyses_dir, 'modeling', save_sub_dir)
    read_tck, fn_tck = join(save_dir, 'tck'), 'fid{}_tc_k=3.xlsx'.format(fid)
    fp_tck = join(read_tck, fn_tck)
    for pth in [save_dir, read_tck]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # -
    depth = df_surface['z'].abs().max()
    surface_profile_radius = dict_settings_radius + dict_surface['surface_profile_radius_adjust']
    units, num_points, degree = 1e-6, 500, 3
    tck, rmin, rmax = manually_fit_tck(
        df=df_surface,
        subset=dict_surface['surface_profile_subset'],
        radius=surface_profile_radius,
        smoothing=dict_surface['tck_smoothing'],
        num_points=num_points,
        degree=degree,
        path_save=read_tck,
        show_plots=False,
    )
    dict_tck_settings = {
        'wid': wid,
        'fid': fid,
        'depth': depth,
        'radius': surface_profile_radius,
        'radius_min': rmin,
        'radius_max': rmax,
        'subset': dict_surface['surface_profile_subset'],
        'smoothing': dict_surface['tck_smoothing'],
        'num_points': num_points,
        'degree': degree,
    }
    # export tck
    df_tck = pd.DataFrame(np.vstack([tck[0], tck[1]]).T, columns=['t', 'c'])
    df_tck_settings = pd.DataFrame.from_dict(data=dict_tck_settings, orient='index', columns=['v'])
    with pd.ExcelWriter(fp_tck) as writer:
        for sheet_name, df, idx, idx_lbl in zip(['tck', 'settings'], [df_tck, df_tck_settings], [False, True], [None, 'k']):
            df.to_excel(writer, sheet_name=sheet_name, index=idx, index_label=idx_lbl)
    dict_fid = dict_from_tck(wid, fid, depth, surface_profile_radius, units, dict_surface['num_segments'], fp_tck, rmin)
    # FID, DEPTH, SURFACE_PROFILE_RADIUS, UNITS, NUM_SEGMENTS, fp_tck=FP_TCK, r_min=rmin
    # profile to model
    px, py = dict_fid['r'], dict_fid['z']
    profile_x, profile_y = px, py

    # ALWAYS PLOT THE PROFILE!!!
    # if not os.path.exists(join(read_tck, save_id + '_profile.png')):
    plot_final_tck_profile(px=px, py=py, save_id=save_id, save_dir=read_tck)

    dict_surface.update({
        'depth_um': dict_fid['depth'],
        'radius_um': dict_fid['radius'],
        'depth': dict_fid['depth'] * 1e-6,
        'radius': dict_fid['radius'] * 1e-6,
        'r': dict_fid['r'],
        'z': dict_fid['z'],
    }) # # , profile_x, profile_y

    return save_id, save_dir, dict_surface


def set_up_model(test_config, wid, memb_id, use_memb_or_comp, root_dir, save_sub_dir,
                 dict_override=None, use_radial_displacement_settings=False):
    # set up model directories and get surface profile
    save_id, path_save_dir, dict_surf = set_up_model_directories_and_get_surface_profile(
        root_dir, test_config, wid, save_sub_dir, dict_override, use_radial_displacement_settings,
    )
    # get membrane settings
    dict_memb = materials.get_membrane_settings(memb_id)
    # stack all settings into a single dictionary
    dict_model_settings = {
        'test_config': test_config,
        'wid': wid,
        'memb_id': memb_id,
        'save_id': save_id,
        'save_dir': path_save_dir,
    }
    dict_model_settings.update(dict_surf)
    dict_model_settings.update(dict_memb)
    # save settings
    df_model_settings = pd.DataFrame.from_dict(data=dict_model_settings, orient='index', columns=['v'])
    df_model_settings.to_excel(
        join(dict_model_settings['save_dir'], dict_model_settings['save_id'] + '_model_settings.xlsx'),
        index=True,
        index_label='k',
    )

    if use_memb_or_comp == 'memb':
        memb_t0 = dict_memb['memb_t0']
        memb_t = dict_memb['memb_t']
        memb_ps = dict_memb['memb_ps']
    elif use_memb_or_comp == 'comp':
        memb_t0 = dict_memb['comp_t0']
        memb_t = dict_memb['comp_t']
        memb_ps = dict_memb['comp_ps']
    else:
        raise ValueError('use_memb_or_comp must be "memb" or "comp"')

    dict_geometry = {
        'shape': dict_surf['shape'],
        'radius': dict_surf['radius'],
        'profile_x': dict_surf['r'],
        'profile_z': dict_surf['z'],
        'memb_t0': memb_t0,
        'memb_t': memb_t,
        'memb_ps': memb_ps,
        'met_t': dict_memb['met_t'],
        'met_ps': dict_memb['met_ps'],
    }
    dict_mechanical = {
        'memb_E': dict_memb['memb_E'],
        'memb_J': dict_memb['memb_J'],
        'met_E': dict_memb['met_E'],
        'met_nu': dict_memb['met_nu'],
        'comp_E': dict_memb['comp_E'],
    }
    dict_electrostatic = {
        'sd_eps_r': dict_surf['sd_eps_r'],
        'sd_t': dict_surf['sd_t'],
        'sd_sr': dict_surf['sd_sr'],
    }

    dict_model_solve = {**dict_geometry, **dict_mechanical, **dict_electrostatic}

    return save_id, path_save_dir, dict_model_solve


def get_surface_profile_settings(wid, test_config, dict_override=None, use_radial_displacement_settings=False):
    shape = 'circle'
    surface_dielectric_eps_r = 3.9
    surface_dielectric_surface_roughness = 1e-9
    surface_dielectric_thickness = 2e-6
    num_segments = 2000

    surface_profile_radius_adjust = None
    surface_profile_subset = None
    tck_smoothing = None

    if wid == 5:
        if use_radial_displacement_settings:
            if test_config in ['01082025_W5-D1_C9-0pT']:
                surface_profile_radius_adjust = 5
            elif test_config in ['01272025_W5-D1_C7-20pT']:
                surface_profile_radius_adjust = -2
            surface_profile_subset = 'average_right_half'
            tck_smoothing = 5.0
        elif test_config in ['01082025_W5-D1_C9-0pT']:
            surface_profile_radius_adjust = 20
            surface_profile_subset = 'left_half'
            tck_smoothing = 20
        elif test_config in ['01272025_W5-D1_C7-20pT']:
            surface_profile_radius_adjust = -10
            surface_profile_subset = 'right_half'  # 'right_half' 'left_half'
            tck_smoothing = 0.0
    elif wid == 10:
        if use_radial_displacement_settings:
            if test_config in ['01092025_W10-A1_C9-0pT']:
                surface_profile_radius_adjust = 20
            elif test_config in ['01262025_W10-A1_C7-20pT']:
                surface_profile_radius_adjust = 25
            else:
                raise ValueError()
            surface_profile_subset = 'right_half'
            tck_smoothing = 1.0
        elif test_config in ['01092025_W10-A1_C9-0pT', '02202025_W10-A1_C21-15pT', '01262025_W10-A1_C7-20pT',
                           '02142025_W10-A1_C22-20pT', '02252025_W10-A1_C17-20pT']:
            surface_profile_radius_adjust = 20
            surface_profile_subset = 'right_half' # 'right_half'  # 'right_half' 'left_half'
            tck_smoothing = 1
    elif wid == 11:
        if test_config in ['01122025_W11-B1_C9-0pT', '03072025_W11-A1_C19-30pT_20+10nmAu']:
            surface_profile_radius_adjust = 30
            surface_profile_subset = 'left_half'  # 'right_half' 'left_half' 'average_right_half'
            tck_smoothing = 0
    elif wid == 12:
        if test_config in ['01122025_W12-D1_C9-0pT']:
            surface_profile_radius_adjust = 15
            surface_profile_subset = 'average_right_half'
            tck_smoothing = 5  # 5: gives scallop-dependent deflection; 12 (old): too smooth for other discussions
        elif test_config in ['01282025_W12-D1_C7-20pT']:
            surface_profile_radius_adjust = 0
            surface_profile_subset = 'average_right_half'
            tck_smoothing = 10
        elif test_config in ['03072025_W12-D1_C19-30pT_20+10nmAu']:
            surface_profile_radius_adjust = 40  # 30
            surface_profile_subset = 'average_right_half'
            tck_smoothing = 0.75  # 0.1
    elif wid == 13:
        if test_config in ['01102025_W13-D1_C9-0pT']:
            surface_profile_radius_adjust = 35
            surface_profile_subset = 'left_half'  # 'right_half' 'left_half'
            tck_smoothing = 8.5
        elif test_config in ['02242025_W13-D1_C15-15pT_iter2', '02242025_W13-D1_C17-20pT',
                             '03052025_W13-D1_C19-30pT_20+10nmAu', '03122025_W13-D1_C15-15pT_25nmAu']:
            surface_profile_radius_adjust = 30
            surface_profile_subset = 'right_half'
            tck_smoothing = 15
    elif wid == 14:
        if use_radial_displacement_settings:
            if test_config in ['01132025_W14-F1_C9-0pT']:
                surface_profile_radius_adjust = 25
            elif test_config in ['01272025_W14-F1_C7-20pT']:
                surface_profile_radius_adjust = 25
            else:
                raise ValueError()
            surface_profile_subset = 'left_half'
            tck_smoothing = 50.0
        elif test_config in ['01132025_W14-F1_C9-0pT']:
            surface_profile_radius_adjust = 10
            surface_profile_subset = 'left_half'  # 'right_half' 'left_half'
            tck_smoothing = 25
        elif test_config in ['01272025_W14-F1_C7-20pT']:
            surface_profile_radius_adjust = 10
            surface_profile_subset = 'right_half'
            tck_smoothing = 0
    elif wid == 101:
        surface_profile_radius_adjust = 0
        surface_profile_subset = 'right_half'
        tck_smoothing = 0
    else:
        pass

    if surface_profile_subset is None:
        raise ValueError('wid not in settings.')

    dict_surface = {
        'shape': shape,
        'surface_profile_radius_adjust': surface_profile_radius_adjust,
        'surface_profile_subset': surface_profile_subset,
        'tck_smoothing': tck_smoothing,
        'num_segments': num_segments,
        'sd_eps_r': surface_dielectric_eps_r,
        'sd_t': surface_dielectric_thickness,
        'sd_sr': surface_dielectric_surface_roughness,
    }

    if dict_override is not None:
        dict_surface.update(dict_override)

    return dict_surface

def sweep_formatter(key, values):
    if key == 'memb_E':
        units = 'MPa'
        scale_units = 1e6
    elif key == 'memb_J':
        units = ''
        scale_units = 1
    elif key == 'met_E':
        units = 'GPa'
        scale_units = 1e9
    elif key == 'met_ps':
        units = ''
        scale_units = 1
    elif key == 'comp_E':
        units = 'MPa'
        scale_units = 1e6
    elif key == 'comp_ps':
        units = ''
        scale_units = 1
    elif key == 'memb_t':
        units = 'um'
        scale_units = 1e-6
    elif key == 'memb_ps':
        units = ''
        scale_units = 1
    elif key == 'sd_t':
        units = 'um'
        scale_units = 1e-6
    elif key == 'sd_sr':
        units = 'nm'
        scale_units = 1e-9
    elif key == 'sd_eps_r':
        units = ''
        scale_units = 1
    else:
        raise ValueError('Key not recognized.')

    labels = values
    values = np.array(values) * scale_units

    return values, labels, units

def recommended_sweep(memb_id, sweep_key):
    values, measured_idx = None, None
    if sweep_key == 'comp_E':
        if memb_id in ['C9-0pT-20nmAu', 'C7-20pT-20nmAu', 'C22-20pT-20nmAu']:
            values = [2.5, 3.7, 5, 7.5]
            measured_idx = 1
        elif memb_id in ['C17-20pT-25nmAu', 'C15-15pT-25nmAu']:
            values = [4.6, 6, 7.5]
            measured_idx = 0
        elif memb_id in ['C21-15pT-30nmAu', 'C19-30pT-20+10nmAu']:
            values = [5.5, 6.5, 7.5]
            measured_idx = 0
        elif memb_id in ['C00-0pT-0nmAu']:
            values = [1, 2.5, 5, 10]
            measured_idx = 0
    elif sweep_key == 'memb_ps':
        if memb_id in ['C9-0pT-20nmAu']:
            values = [1.0, 1.006, 1.016, 1.026]
            measured_idx = 1
        elif memb_id in ['C7-20pT-20nmAu']:
            values = [1.04, 1.061, 1.09, 1.12]
            measured_idx = 1
        elif memb_id in ['C22-20pT-20nmAu']:
            values = [1.06, 1.08, 1.1, 1.12]
            measured_idx = 1
        elif memb_id in ['C00-0pT-0nmAu']:
            values = [1, 1.1, 1.2, 1.3]
            measured_idx = 0
    elif sweep_key == 'memb_J':
        values = [25, 50, 75, 100]
        measured_idx = 1
    elif sweep_key == 'sd_t':
        values = [1.8, 2, 2.2, 2.4]
        measured_idx = 1

    return values, measured_idx


if __name__ == '__main__':

    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation'
    TEST_CONFIG = '01132025_W14-F1_C9-0pTx'
    WID = 14

    MEMB_ID = 'C9-0pT-20nmAu' # 'C9-0pT-20nmAu' 'C7-20pT-20nmAu' 'C22-20pT-20nmAu' 'C15-15pT-25nmAu' 'C19-30pT-20+10nmAu'
    USE_MEMB_OR_COMP = 'comp'  # 'memb' or 'comp'

    # Set up configurations to iterate through
    sweep_key = 'comp_E'  # 'comp_E', 'memb_E', 'met_E', 'memb_t', 'memb_ps', 'sd_t', 'sd_eps_r', 'memb_J'
    sweep_values, measured_idx = recommended_sweep(MEMB_ID, sweep_key)
    # sweep_values = [1, 5, 10, 25]
    print(sweep_values)
    if sweep_values is None:
        raise ValueError("No recommended values found.")
    plot_sweep_idx = [measured_idx]  # plot these sweep values (by index)
    use_radial_displacement_settings = False
    SAVE_SUB_DIR = USE_MEMB_OR_COMP + '_sweep_' + sweep_key + '_energy_landscape'
    dict_override = None #  {'tck_smoothing': 6.5}

    # --- --- SOLVER SEQUENCE

    save_id, save_dir, dict_model_solve = set_up_model(TEST_CONFIG, WID, MEMB_ID, USE_MEMB_OR_COMP,
                                                       ROOT_DIR, SAVE_SUB_DIR, dict_override=dict_override,
                                                       use_radial_displacement_settings=use_radial_displacement_settings)
    sweep_values, sweep_labels, sweep_units = sweep_formatter(sweep_key, sweep_values)
    save_sweep = save_id
    save_sweep_value = save_id
    z_clip = 2.5
    voltages = np.arange(5, 131, 1)
    ignore_dZ_below_v = (20e-6, 50)  # ignore dZ > dZ.max() - VAR1, if voltage < VAR2
    assign_z = 'z_comp'  # options: 'z_memb', 'z_mm', 'z_comp'
    use_neo_hookean = False
    # export intermediate analyses (i.e., energy parts per volume)
    export_elastic_energy = True  # True False
    export_all_energy = True
    export_total_energy = True
    # plotting (if None, then do not plot. Otherwise, plot energy by depth for specified voltages)
    plot_e_by_z_overlay_v = [[60, 70, 80, 90, 100, 110],]  # [[40, 80, 120, 160, 200], [150, 160, 170, 180], [200, 220, 240]]
    animate_e_by_z_by_v = None  # None or a list of voltages to plot total energy vs depth
    plot_minima_for_vs = [60, 70, 80, 90, 100, 110]  # [60, 75, 100] # None or a list of voltages to plot total energy vs depth and first minima
    animate_minima_by_z_by_v = None # np.arange(60, 256, 5)  # None or a list of voltages to plot total energy vs depth and first minima
    # -
    if WID == 5:
        FIT_SURFACE_POLY_DEG = 5
    elif WID == 10:
        FIT_SURFACE_POLY_DEG = 5
    elif WID == 12:
        FIT_SURFACE_POLY_DEG = 15
    elif WID in [14, 13]:
        FIT_SURFACE_POLY_DEG = 10
    else:
        raise ValueError('WID not recognized or appropriate settings not yet determined.')
    # -

    save_volume = join(save_dir, 'volume')
    save_correction = join(save_dir, 'correction')
    save_elastic = join(save_dir, 'elastic')
    save_electrostatic = join(save_dir, 'electrostatic')
    save_total_energy = join(save_dir, 'total_energy')
    save_energy_minima = join(save_dir, 'energy_minima')
    save_specific = join(save_dir, 'specific')
    pths = [save_volume, save_correction]
    if export_elastic_energy:
        pths.append(save_elastic)
    if plot_e_by_z_overlay_v is not None:
        pths.append(save_electrostatic)
    if plot_e_by_z_overlay_v is not None or animate_e_by_z_by_v is not None:
        pths.append(save_total_energy)
    if plot_minima_for_vs is not None or animate_minima_by_z_by_v is not None:
        pths.append(save_energy_minima)
    if np.max(plot_sweep_idx) >= 0:
        pths.append(save_specific)
    for pth in pths:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # columns needed for each sub-analysis
    columns_id = ['step', 'dZ']
    inputs_elastic = [
        'vol_i', 'stretch_i',
        'vol_i_metal', 'stretch_i_metal',
        'vol_flat', 'stretch_flat',
        'vol_flat_metal', 'stretch_flat_metal',
    ]
    outputs_elastic = [
        'Em_seg_i_memb', 'Em_seg_i_metal',
        'Em_seg_i_mm', 'Em_seg_i_comp',
        'Em_seg_sum_i_memb', 'Em_seg_sum_i_metal', 'Em_seg_sum_i_mm', 'Em_seg_sum_i_comp',
        'Em_flat_i_memb', 'Em_flat_i_metal', 'Em_flat_i_mm', 'Em_flat_i_comp',
        'Em_tot_i_memb', 'Em_tot_i_metal', 'Em_tot_i_mm', 'Em_tot_i_comp',
    ]
    inputs_electrostatic = ['sa_i']
    outputs_total_energy = ['U', 'E_tot_i_memb', 'E_tot_i_mm', 'E_tot_i_comp']

    # ---

    # --- Volumes
    #if sweep_key not in ['memb_t', 'memb_ps', 'met_t', 'met_ps']:
    df_volume = model_zipping.calculate_zipped_segment_volumes_by_depth(dict_model_solve)
    df_volume.to_excel(join(save_volume, save_id + '_model_volumes.xlsx'), index=False)

    df_strain = model_zipping.calculate_radial_displacement_by_depth(df_volume, FIT_SURFACE_POLY_DEG, path_save=save_correction, save_id=save_id)
    df_strain.to_excel(join(save_dir, save_id + '_model_strain-by-z.xlsx'), index=False)
    model_zipping.plot_deformation_by_depth(df_strain, save_dir, save_id + '_model_strain-by-z.png', z_clip)

    # -
    if use_radial_displacement_settings:
        raise ValueError("No need to run further.")

    # ------------------- # ------------------- ITERATE CONFIGURATIONS AND SOLVE

    df_dZ_by_v_by_E = []
    ii = np.arange(len(sweep_values))
    for i, sweep_value, sweep_label in zip(ii, sweep_values, sweep_labels):
        save_id = save_sweep_value + '_{}_{}{}'.format(sweep_key, sweep_label, sweep_units)
        dict_model_solve.update({sweep_key: sweep_value})

        # -

        # ------------------- START SOLVE ONE CONFIGURATION

        # --- Only re-solve volumes if sweeping membrane thickness
        if sweep_key in ['memb_t', 'memb_ps', 'met_t', 'met_ps']:
            df_volume = model_zipping.calculate_zipped_segment_volumes_by_depth(dict_model_solve)
            df_strain = model_zipping.calculate_radial_displacement_by_depth(df_volume, FIT_SURFACE_POLY_DEG, path_save=save_correction, save_id=save_id)
            if i in plot_sweep_idx:
                df_volume.to_excel(join(save_volume, save_id + '_model_volumes.xlsx'), index=False)
                df_strain.to_excel(join(save_dir, save_id + '_model_strain-by-z.xlsx'), index=False)
                model_zipping.plot_deformation_by_depth(df_strain, save_dir, save_id + '_model_strain-by-z.png', z_clip)

        # -

        # Elastic energy
        df_elastic = model_zipping.calculate_elastic_energy_by_depth(df_volume[columns_id + inputs_elastic], dict_model_solve,
                                                       use_neo_hookean=use_neo_hookean)

        if i in plot_sweep_idx and export_elastic_energy is True:
            df_elastic.to_excel(join(save_elastic, save_id + '_model_elastic_energy.xlsx'), index=False)
            model_zipping.plot_elastic_energy_by_depth(df_elastic, save_elastic, save_id + '_model_elastic_energy.png', z_clip)
            model_zipping.plot_elastic_energy_by_depth_by_component(df_elastic, save_elastic, save_id + '_model_elastic_energy_parts.png', z_clip)
        # -
        # --- Calculate electrostatic energy + total energy
        df_all_energy_by_voltage = model_zipping.wrapper_calculate_total_energy_by_depth_by_voltage(
            df_elastic[columns_id + outputs_elastic],
            df_volume[columns_id + inputs_electrostatic],
            dict_model_solve,
            voltages,
        )
        # plot electrostatic energy
        if i in plot_sweep_idx and plot_e_by_z_overlay_v is not None:
            for k, vo in enumerate(plot_e_by_z_overlay_v):
                # This is the only plot that needs the MASSIVE dataframe
                model_zipping.plot_electrostatic_energy_by_depth_overlay_by_voltage(
                    df_all_energy_by_voltage,
                    vo,
                    save_electrostatic,
                    save_id + f'_overlay_electrostatic_energy_by_V_group{k}.png',
                    z_clip,
                )
        # -
        # Export MASSIVE dataframe with all energy components
        if i in plot_sweep_idx and export_all_energy:
            df_all_energy_by_voltage.to_excel(join(save_dir, save_specific, save_id + '_model_all_energy_by_voltage.xlsx'), index=False)
        # -
        # Keep only necessary columns
        df_total_energy_by_voltage = df_all_energy_by_voltage[columns_id + outputs_total_energy]
        if i in plot_sweep_idx and export_total_energy:
            df_total_energy_by_voltage.to_excel(join(save_dir, save_specific, save_id + '_model_total_energy_by_voltage.xlsx'), index=False)
        # -
        # plot total energy
        if i in plot_sweep_idx and plot_e_by_z_overlay_v is not None:
            for k, vo in enumerate(plot_e_by_z_overlay_v):
                model_zipping.plot_total_energy_by_depth_overlay_by_voltage(
                    df_total_energy_by_voltage,
                    vo,
                    save_total_energy,
                    save_id + f'_{k}',
                    z_clip,
                )
                model_zipping.plot_normalized_total_energy_by_depth_by_voltage_overlay_by_model(
                    df_total_energy_by_voltage,
                    vo,
                    save_total_energy,
                    save_id + f'_norm{k}',
                    z_clip,
                )
        # -
        if i in plot_sweep_idx and animate_e_by_z_by_v is not None:
            model_zipping.animate_normalized_total_energy_by_depth_by_voltage_overlay_by_model(
                df_total_energy_by_voltage,
                animate_e_by_z_by_v,
                save_total_energy,
                save_id,
                z_clip,
            )
        # -
        # --- Find first energy minima for each voltage
        df_dZ_by_v = model_zipping.wrapper_find_first_energy_minima_by_voltage(
            df=df_total_energy_by_voltage,
            ignore_dZ_below_v=ignore_dZ_below_v,
            assign_z=assign_z,
        )
        if i in plot_sweep_idx:
            df_dZ_by_v.to_excel(join(save_dir, save_specific, save_id + '_model_z-by-v.xlsx'), index=False)
            model_zipping.plot_z_by_voltage_by_model(df_dZ_by_v, join(save_dir, save_specific), save_id)
        # -
        # plot identified first minima on total energy vs depth
        if i in plot_sweep_idx and plot_minima_for_vs is not None:
            model_zipping.plot_total_energy_and_minima_by_depth(
                plot_minima_for_vs,
                df_total_energy_by_voltage,
                df_dZ_by_v,
                save_energy_minima,
                save_id,
                z_clip,
            )
        # -
        if i in plot_sweep_idx and animate_minima_by_z_by_v is not None:
            model_zipping.animate_total_energy_and_minima_by_depth(
                animate_minima_by_z_by_v,
                df_total_energy_by_voltage,
                df_dZ_by_v,
                save_energy_minima,
                save_id,
                z_clip,
            )

        # -

        # ------------------- END SOLVE ONE CONFIGURATION

        df_dZ_by_v[sweep_key] = sweep_value
        df_dZ_by_v_by_E.append(df_dZ_by_v)

    # ------------------- # ------------------- END ITERATE CONFIGURATIONS AND SOLVE

    # ------------------- COMPARE (PLOT) CONFIGURATIONS

    df_dZ_by_v_by_E = pd.concat(df_dZ_by_v_by_E)
    df_dZ_by_v_by_E.to_excel(join(save_dir, save_sweep + '_model_z-by-v.xlsx'), index=False)
    model_zipping.plot_sweep_z_by_voltage_by_model(df_dZ_by_v_by_E, sweep_key, sweep_labels, sweep_units, save_dir, save_sweep)
    model_zipping.plot_sweep_z_by_voltage_comp_E(df_dZ_by_v_by_E, sweep_key, sweep_labels, sweep_units, save_dir, save_sweep)





