import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dpt
from utils import plotting, empirical, fit


def export_net_d0zr_per_pid_per_tid(df, xym, tid, dict_settings, dict_test, path_results, average_max_n_positions):
    # assign simple column labels
    # xym = 'g'  # options: 'g': sub-pixel localization using 2D Gaussian; 'm': discrete localization using cross-corr
    px, py, pr, pdx, pdy, pdr, pd0x, pd0y, pd0r = [k + xym for k in ['x', 'y', 'r', 'dx', 'dy', 'dr', 'd0x', 'd0y', 'd0r']]
    # -
    path_save = join(path_results, 'net-d0zr_per_pid', 'xy' + xym)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    # calculate net displacement
    # (maximum displacement relative to initial coords (i.e., d0f))
    _ = dpt.calculate_net_total_displacement(
        df=df,
        pxyr=(px, py, pr),
        start_frame=dict_test['dpt_start_frame'],
        end_frames=None,  # dict_test['dpt_end_frames'],
        path_results=path_save,
        average_max_n_positions=average_max_n_positions,
        save_id=f'tid{tid}',
    )


def get_surface_profile_dict(dict_settings):
    if 'fid_process_profile' in dict_settings.keys():
        surf_fid_override = dict_settings['fid_process_profile']
    else:
        surf_fid_override = None
    surf_profile_subset = 'right_half'  # 'full', 'left_half', 'right_half'
    surf_shift_r, surf_shift_z, surf_scale_z = 0, 0, 1
    include_hole = True
    if surf_profile_subset == 'full':
        include_hole = False

    df_surface = empirical.read_surface_profile(
        dict_settings,
        subset=surf_profile_subset,
        hole=include_hole,
        fid_override=surf_fid_override,
    )
    dict_surface_profilometry = {'r': df_surface['r'].to_numpy(), 'z': df_surface['z'].to_numpy(),
                                 'dr': surf_shift_r, 'dz': surf_shift_z, 'scale_z': surf_scale_z,
                                 'subset': surf_profile_subset}
    return dict_surface_profilometry


def second_pass(df, xym, tid, dict_settings, dict_test, path_results, animate_frames=None, animate_rzdr=None):
    # assign simple column labels
    # xym = 'g'  # options: 'g': sub-pixel localization using 2D Gaussian; 'm': discrete localization using cross-corr
    px, py, pr, pdx, pdy, pdr, pd0x, pd0y, pd0r = [k + xym for k in ['x', 'y', 'r', 'dx', 'dy', 'dr', 'd0x', 'd0y', 'd0r']]
    pz, pdz, pd0z = 'z', 'dz', 'd0z'
    pdz_lock_in, pd0z_lock_in = 'dz_lock_in', 'd0z_lock_in'
    # -
    # compare with model
    dz_compare_with_model = pd0z  # pd0z or pdz
    dz_lock_in_compare_with_model = dz_compare_with_model + '_lock_in'
    # for plotting 3D DPT coords overlay surface profile
    przdr = (pr, pd0z, pdr)
    # -
    # -
    # initialize
    plot_scatter_xy_by_frame = False
    plot_depth_dependent_in_plane_stretch_w_rotation_correction = False
    rotation_correction_poly_deg = [7, 9]  # [3, 5]
    df_model_VdZ = None
    df_model_strain = None
    if 'model_mkey' in dict_settings.keys():
        model_mkey, model_mval = dict_settings['model_mkey'], dict_settings['model_mval']
    else:
        model_mkey, model_mval = None, None
    # -
    if dict_test['samples_per_voltage_level'] > 1.5 and pd0z_lock_in in df.columns:
        plot_all_dz_lock_ins = True
        plot_pids_by_synchronous_time_voltage_dz_lock_in = plot_all_dz_lock_ins
    else:
        plot_all_dz_lock_ins = False
        plot_pids_by_synchronous_time_voltage_dz_lock_in = plot_all_dz_lock_ins
    # -
    if animate_frames is not None:
        pass
    elif 'animate_frames' in dict_test.keys():
        animate_frames = np.arange(dict_test['animate_frames'][0], dict_test['animate_frames'][1])
    elif 'dpt_start_frame' in dict_test.keys() and 'dpt_end_frames' in dict_test.keys():
        animate_frames = np.arange(dict_test['dpt_start_frame'][1], dict_test['dpt_end_frames'][0])
    else:
        animate_frames = np.arange(5, 125)
    frames_drdz = animate_frames
    # -
    if animate_rzdr is not None:
        przdr = animate_rzdr
    # -
    if 'smu_test_type' in dict_test.keys():
        if dict_test['smu_test_type'] in [1, 2, '1', '2']:
            compare_pull_in_voltage_with_model = True
            plot_depth_dependent_in_plane_stretch = True
            plot_pids_dr_by_dz = False
            plot_pids_dz_by_voltage_hysteresis = False
            plot_normalized_membrane_profile, frois_norm_profile = True, [10, 70, 90, 105]
            plot_1D_dz_by_r_by_frois_with_surface_profile, frois_overlay = True, [10, 70, 90, 105]
        else:
            compare_pull_in_voltage_with_model = False
            plot_depth_dependent_in_plane_stretch = False
            plot_pids_dr_by_dz = False
            plot_pids_dz_by_voltage_hysteresis = False
            plot_normalized_membrane_profile, frois_norm_profile = False, []
            plot_1D_dz_by_r_by_frois_with_surface_profile, frois_overlay = False, []
    elif 'test_type' in dict_test.keys():
        if dict_test['test_type'] in ['STD1', 'STD2']:
            compare_pull_in_voltage_with_model = True
            plot_depth_dependent_in_plane_stretch = True
            plot_pids_dr_by_dz = False
            plot_pids_dz_by_voltage_hysteresis = False
            plot_normalized_membrane_profile, frois_norm_profile = False, [40, 50, 70, 101]
            plot_1D_dz_by_r_by_frois_with_surface_profile, frois_overlay = False, [40, 50, 75, 101]
        else:
            compare_pull_in_voltage_with_model = False
            plot_depth_dependent_in_plane_stretch = False
            plot_pids_dr_by_dz = False
            plot_pids_dz_by_voltage_hysteresis = False
            plot_normalized_membrane_profile, frois_norm_profile = False, []
            plot_1D_dz_by_r_by_frois_with_surface_profile, frois_overlay = False, []
    else:
        # pass
        compare_pull_in_voltage_with_model = False
        plot_depth_dependent_in_plane_stretch = False
        plot_pids_dr_by_dz = False
        plot_pids_dz_by_voltage_hysteresis = False
        plot_normalized_membrane_profile, frois_norm_profile = False, []
        plot_1D_dz_by_r_by_frois_with_surface_profile, frois_overlay = False, []

    # -
    only_pids = None  # if [1, 2...], then remove these pids from df, dfd, and dfd0 before plotting; None: keep all pids
    not_pids = []  # if [], then keep all pids in df, dfd, and dfd0 for plotting; [1, 2...]: remove before plotting
    only_per_pid_plot_these_pids = None  # None; if [1, 2...], then plot only these pids for per-pid trajectory plots
    # -
    # modifiers (True False)
    eval_pids_drz = True  # True: calculate/export net-displacement per-particle in r- and z-directions
    average_max_n_positions = 70
    # -
    plot_all_pids_by_X, Xs = False, ['frame', 't_sync', 'STEP', 'VOLT']
    plot_heatmaps = True  # True: plot 2D heat map (requires eval_pids_dz having been run)
    # --- --- PIDS
    plot_single_pids_by_frame = False  # If you have voltage data, generally False. Can be useful to align by "frame" (not time)
    plot_pids_by_synchronous_time_voltage = False
    plot_pids_by_synchronous_time_voltage_monitor = False  # AC only
    # -
    # --- --- ANIMATIONS (True, False)
    plot_quiver_xy_by_frame = True
    plot_1D_dz_by_r_by_frame_with_surface_profile = True
    show_zipping_interface = True
    dr_ampl = 10
    fit_spline_to_memb = False
    test_description = dict_test['filename'].split('.xlsx')[0].split('_data')[0].replace('_', ' ')
    if (plot_quiver_xy_by_frame or plot_1D_dz_by_r_by_frame_with_surface_profile or
            plot_normalized_membrane_profile or plot_1D_dz_by_r_by_frois_with_surface_profile):
        dict_surface_profilometry = get_surface_profile_dict(dict_settings)
    if fit_spline_to_memb:
        # get radial position of n-th inner-most particle
        faux_is_average_of_n = 5
        faux_r_zero = df[df['frame'] == animate_frames[0]].sort_values(pr, ascending=True).iloc[faux_is_average_of_n][pr] + 1.5
        dict_fit_memb = {
            's': 350,  # straight-angled sidewalls: s ~ 1500; erf: s ~ 1000; multi-stable: s ~ 250
            'faux_r_zero': faux_r_zero,  # dict_settings['radius_hole_microns'] * 1.5,  # None: do not use faux(r=0)
            'faux_r_edge': dict_settings['radius_microns'] + 0,  # None: do not use faux(r=A, dz=0) particle for fitting
            'show_faux_particles': True,  # True: show all particles used for fitting, False: show only tracked FP's
        }
    else:
        dict_fit_memb = None
    # -
    # ---
    plot_1D_z_by_r_by_frame = False
    plot_1D_dz_by_r_by_frame = False
    plot_2D_z_by_frame = False
    plot_2D_dz_by_frame = True
    plot_2D_dr_by_frame = False
    # -
    # for contourf plots
    levels_z = 15
    levels_r = 10
    if xym == 'm':
        plot_pids_by_synchronous_time_voltage = False
        plot_pids_dz_by_voltage_ascending = False
        plot_pids_dz_by_voltage_hysteresis = False
        plot_1D_z_by_r_by_frame = False
        plot_1D_dz_by_r_by_frame = False
        plot_2D_z_by_frame = False
        plot_2D_dz_by_frame = False
        plot_1D_dz_by_r_by_frame_with_surface_profile = False
    # ---
    if plot_pids_by_synchronous_time_voltage_monitor and 'MONITOR_VALUE' not in df.columns:
        plot_pids_by_synchronous_time_voltage_monitor = False

    # read model data
    if compare_pull_in_voltage_with_model or plot_depth_dependent_in_plane_stretch:
        #if 'path_model_dZ_by_V' or 'path_model_strain' in dict_test.keys():
        #    print("TEST SETTINGS OVERRIDING SETTINGS FOR 'path_model_dZ_by_V'")
        #    raise ValueError("This is a special case. Need to double-check this instance.")
        #    df_model_VdZ = pd.read_excel(dict_test['path_model_dZ_by_V'])
        if 'path_model' in dict_settings.keys():
            if dict_settings['path_model'] == 'nan':
                compare_pull_in_voltage_with_model = False
                plot_depth_dependent_in_plane_stretch = False
            else:
                mfiles = [x for x in os.listdir(dict_settings['path_model'].strip("'")) if x.endswith('.xlsx')]
                mfile_z_by_v = [x for x in mfiles if x.endswith('z-by-v.xlsx')][0]
                mfile_strain_by_z = [x for x in mfiles if x.endswith('strain-by-z.xlsx')][0]

                df_model_VdZ = pd.read_excel(join(dict_settings['path_model'], mfile_z_by_v))
                df_model_strain = pd.read_excel(join(dict_settings['path_model'], mfile_strain_by_z))
        elif 'path_model_dZ_by_V' or 'path_model_strain' in dict_settings.keys():
            df_model_VdZ = pd.read_excel(dict_settings['path_model_dZ_by_V'].strip("'"))
            df_model_strain = pd.read_excel(dict_settings['path_model_strain'].strip("'"))
        else:
            compare_pull_in_voltage_with_model = False
            plot_depth_dependent_in_plane_stretch = False
        if plot_depth_dependent_in_plane_stretch:
            plot_depth_dependent_in_plane_stretch_w_rotation_correction = True

    # -
    path_results_rep = join(path_results.format(tid), 'xy' + xym)
    path_results_per_pid = join(path_results_rep, 'per_pid')
    path_results_slice = join(path_results_rep, 'slice')
    path_results_xy = join(path_results_rep, 'field-of-view')
    path_results_rz = join(path_results_rep, 'cross-section')
    # data slice
    path_results_compare_dZ_by_V = join(path_results_slice, 'compare_dZ_by_V_with_model')
    path_results_depth_dependent_in_plane_stretch = join(path_results_slice, 'depth-dependent-in-plane-stretch')
    path_results_depth_dependent_in_plane_stretch_w_rotation_correction = join(path_results_slice, 'w_rotation_correction')
    path_results_pids_by_X = join(path_results_slice, 'pids_by_X')
    # particle trajectories
    path_results_pids_by_frame = join(path_results_per_pid, 'pids_by_frame')
    path_results_pids_dr_by_dz = join(path_results_per_pid, 'pids_dr_by_dz')
    path_results_pids_by_synchronous_time_voltage = join(path_results_per_pid, 'pids_by_sync-time-volts')
    path_results_pids_by_synchronous_time_voltage_dz_lock_in = join(path_results_per_pid, 'pids_by_sync-time-volts_dz-lock-in')
    path_results_pids_by_synchronous_time_voltage_monitor = join(path_results_per_pid, 'pids_by_sync-time-volts-monitor')
    path_results_pids_dz_by_voltage_hysteresis = join(path_results_per_pid, 'pids_dz_by_voltage_hysteresis')
    path_results_pids_dzdr_by_voltage_hysteresis = join(path_results_per_pid, 'pids_dzdr_by_voltage_hysteresis')
    # field-of-view
    path_results_scatter_xy_by_frame = join(path_results_xy, 'scatter_xy_by_frame')
    path_results_quiver_xy_by_frame = join(path_results_xy, 'quiver_dxdy_by_frame')
    path_results_2D_z_by_frame = join(path_results_xy, '2D_z_by_frame')
    path_results_2D_dz_by_frame = join(path_results_xy, '2D_dz_by_frame')
    path_results_2D_dr_by_frame = join(path_results_xy, '2D_dr_by_frame')
    # cross-section
    path_results_1D_z_by_r_by_frame = join(path_results_rz, '1D_z-r_by_frame')
    path_results_1D_dz_by_r_by_frame = join(path_results_rz, '1D_dz-r_by_frame')
    path_results_1D_dznorm_by_r_by_frois = join(path_results_rz, f'1D_{przdr[1]}{przdr[2]}-r_by_frois_norm-memb-profile')
    path_results_1D_dz_by_r_by_frame_w_surf = join(path_results_rz, f'1D_{przdr[1]}{przdr[2]}-r_by_frame_w-surf+dr{dr_ampl}X')
    path_results_1D_dz_by_r_by_frois_w_surf = join(path_results_rz, f'1D_{przdr[1]}{przdr[2]}-r_by_frois_w-surf')
    # -

    # make directories
    pths = [path_results_rep, path_results_per_pid, path_results_xy, path_results_rz,
            path_results_compare_dZ_by_V, path_results_depth_dependent_in_plane_stretch,
            path_results_depth_dependent_in_plane_stretch_w_rotation_correction, path_results_pids_by_X,
            path_results_pids_by_frame, path_results_pids_dr_by_dz,
            path_results_pids_by_synchronous_time_voltage, path_results_pids_by_synchronous_time_voltage_dz_lock_in, path_results_pids_by_synchronous_time_voltage_monitor,
            path_results_pids_dz_by_voltage_hysteresis, path_results_pids_dzdr_by_voltage_hysteresis,
            path_results_scatter_xy_by_frame, path_results_quiver_xy_by_frame,
            path_results_1D_z_by_r_by_frame, path_results_1D_dz_by_r_by_frame,
            path_results_2D_z_by_frame, path_results_2D_dz_by_frame, path_results_2D_dr_by_frame,
            path_results_1D_dznorm_by_r_by_frois,
            path_results_1D_dz_by_r_by_frame_w_surf, path_results_1D_dz_by_r_by_frois_w_surf]
    mods = [True, True, True, True,
            compare_pull_in_voltage_with_model, plot_depth_dependent_in_plane_stretch,
            plot_depth_dependent_in_plane_stretch_w_rotation_correction, plot_all_pids_by_X,
            plot_single_pids_by_frame, plot_pids_dr_by_dz,
            plot_pids_by_synchronous_time_voltage, plot_pids_by_synchronous_time_voltage_dz_lock_in, plot_pids_by_synchronous_time_voltage_monitor,
            plot_pids_dz_by_voltage_hysteresis, plot_pids_dz_by_voltage_hysteresis,
            plot_scatter_xy_by_frame, plot_quiver_xy_by_frame,
            plot_1D_z_by_r_by_frame, plot_1D_dz_by_r_by_frame,
            plot_2D_z_by_frame, plot_2D_dz_by_frame, plot_2D_dr_by_frame,
            plot_normalized_membrane_profile,
            plot_1D_dz_by_r_by_frame_with_surface_profile, plot_1D_dz_by_r_by_frois_with_surface_profile]
    pths = [x for x, y in zip(pths, mods) if y is True]
    for pth in pths:
        if not os.path.exists(pth):
            os.makedirs(pth)

    # --- evaluate net displacement per-pid
    # always evaluate all particles
    if eval_pids_drz:
        # calculate displacement "snapshot"
        # (position during end frames relative to start frames)
        dfd = dpt.calculate_net_displacement(
            df=df,
            pxyr=(px, py, pr),
            start_frame=dict_test['dpt_start_frame'],
            end_frames=dict_test['dpt_end_frames'],
            path_results=path_results_rep,
            average_max_n_positions=None,
        )
        # calculate net displacement
        # (maximum displacement relative to initial coords (i.e., d0f))
        dfd0 = dpt.calculate_net_total_displacement(
            df=df,
            pxyr=(px, py, pr),
            start_frame=dict_test['dpt_start_frame'],
            end_frames=None, # dict_test['dpt_end_frames'],
            path_results=path_results_rep,
            average_max_n_positions=average_max_n_positions,
        )
    else:
        dfd = pd.read_excel(join(path_results_rep, 'net-dzr_per_pid.xlsx'))
        dfd0 = pd.read_excel(join(path_results_rep, 'net-d0zr_per_pid.xlsx'))

    # --- plot only a subset of particles
    # removes "bad" particles from visualizations, or
    # includes only "good" particles
    if only_pids:
        print("Keeping only pids: {}".format(only_pids))
        df = df[df['id'].isin(only_pids)]
    elif not_pids:
        print("Removing pids: {}".format(not_pids))
        df = df[~df['id'].isin(not_pids)]
    only_pids = df['id'].unique()
    dfd = dfd[dfd['id'].isin(only_pids)]
    dfd0 = dfd0[dfd0['id'].isin(only_pids)]
    if only_per_pid_plot_these_pids is not None:
        only_pids = only_per_pid_plot_these_pids

    # --- --- SINGLE PLOTS

    # --- SLICE
    if (compare_pull_in_voltage_with_model or plot_depth_dependent_in_plane_stretch or plot_all_pids_by_X):
        if compare_pull_in_voltage_with_model:
            pids_to_compare_with_model = dfd0[dfd0['dz_mean'] < dfd0['dz_mean'].quantile(0.1)]['id'].unique()
            for pid in pids_to_compare_with_model:
                df_pid = df[df['id'] == pid].reset_index(drop=True)
                plotting.compare_dZ_by_V_with_model(
                    df=df_pid,
                    dfm=df_model_VdZ,
                    path_results=path_results_compare_dZ_by_V,
                    save_id=f'pid{int(pid)}',
                    mkey=model_mkey,  # column name
                    mval=model_mval,  # column value
                    dz=dz_compare_with_model,
                )
                plotting.compare_dZmin_by_V_with_model(
                    df=df_pid,
                    dfm=df_model_VdZ,
                    path_results=path_results_compare_dZ_by_V,
                    save_id=f'pid{int(pid)}',
                    mkey=model_mkey,  # column name
                    mval=model_mval,  # column value
                    dz=dz_compare_with_model,
                )
                if plot_all_dz_lock_ins and dz_lock_in_compare_with_model in df_pid.columns:
                    plotting.compare_dZ_by_V_with_model(
                        df=df_pid,
                        dfm=df_model_VdZ,
                        path_results=path_results_compare_dZ_by_V,
                        save_id=f'pid{int(pid)}',
                        mkey=model_mkey,  # column name
                        mval=model_mval,  # column value
                        dz=dz_lock_in_compare_with_model,
                    )

        if plot_depth_dependent_in_plane_stretch:
            plotting.plot_depth_dependent_in_plane_stretch_from_dfd(
                dfd=dfd0,
                path_results=path_results_depth_dependent_in_plane_stretch,
                save_id='dfd0',
            )
            plotting.plot_depth_dependent_in_plane_stretch_from_dfd(
                dfd=dfd,
                path_results=path_results_depth_dependent_in_plane_stretch,
                save_id='dfd',
            )

            if df_model_strain is not None:
                plotting.compare_depth_dependent_in_plane_stretch_with_model(
                    dfd=dfd0,
                    dfm=df_model_strain,
                    path_results=path_results_depth_dependent_in_plane_stretch,
                    save_id='dfd0',
                )
                plotting.compare_depth_dependent_in_plane_stretch_with_model(
                    dfd=dfd,
                    dfm=df_model_strain,
                    path_results=path_results_depth_dependent_in_plane_stretch,
                    save_id='dfd',
                )

                if plot_depth_dependent_in_plane_stretch_w_rotation_correction:
                    dict_surface_profilometry = get_surface_profile_dict(dict_settings)
                    surf_r, surf_z = dict_surface_profilometry['r'], dict_surface_profilometry['z']
                    for poly_deg in rotation_correction_poly_deg:
                        func_apparent_r_displacement = empirical.calculate_apparent_radial_displacement_due_to_rotation(
                            surf_r=surf_r,
                            surf_z=surf_z,
                            poly_deg=poly_deg,
                            membrane_thickness=dict_settings['membrane_thickness'],
                            z_clip=-0.125,  # -0.125 (for W11: -0.8)
                            path_save=path_results_depth_dependent_in_plane_stretch_w_rotation_correction,
                        )


                        plotting.compare_corrected_depth_dependent_in_plane_stretch_with_model(
                            dfd=dfd0,
                            dfm=df_model_strain,
                            correction_function=func_apparent_r_displacement,
                            pr=pr,
                            path_results=path_results_depth_dependent_in_plane_stretch_w_rotation_correction,
                            save_id='poly-deg={}_dfd0_corr-dr'.format(poly_deg),
                            dfd_id='dfd0',
                            poly_deg_id=poly_deg,
                        )

                        plotting.compare_corrected_depth_dependent_in_plane_stretch_with_model(
                            dfd=dfd,
                            dfm=df_model_strain,
                            correction_function=func_apparent_r_displacement,
                            pr=pr,
                            path_results=path_results_depth_dependent_in_plane_stretch_w_rotation_correction,
                            save_id='poly-deg={}_dfd_corr-dr'.format(poly_deg),
                            dfd_id='dfd',
                            poly_deg_id=poly_deg,
                        )

        if plot_all_pids_by_X:
            for X in Xs:
                plotting.plot_all_pids_displacement_trajectory(
                    df=df,
                    px=X,
                    pdzdr=(pd0z, pd0r),
                    path_results=path_results_pids_by_X,
                )

    # --- --- FIELD OF VIEW
    if plot_scatter_xy_by_frame:
        plotting.plot_scatter_xy(df=df,
                                 pxy=(px, py),
                                 pcolor='dz',
                                 savepath=join(path_results_scatter_xy_by_frame, 'scatter_xy.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta z \: (\mu m)$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

    if plot_quiver_xy_by_frame:
        plotting.plot_quiver_xy_dxdy(
            df=df,
            pxydxdy=(px, py, pdx, pdy),
            pcolor_dxdy=pdr,
            frames=animate_frames,
            scale=5,  # arrow lengths are 5X larger than data units
            savepath=join(path_results_quiver_xy_by_frame, 'quiver_xy_dxdy_clr=dr_fr{}.png'),
            field=(0, dict_settings['field_of_view']),
            units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta z \: (\mu m)$'),
            overlay_circles=True,
            dict_settings=dict_settings,
            show_interface=show_zipping_interface,
            dict_surf=dict_surface_profilometry,
            prz=(pr, pd0z),
            temporal_display_units='frame',
            title=test_description,
        )

    if plot_heatmaps:
        # ---
        plotting.plot_2D_heatmap(df=dfd0,
                                 pxyz=(px, py, 'dz_mean'),
                                 savepath=join(path_results_xy, 'dz-mean_per_pid_2D__d0z.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_z,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta_{o} z \: (\mu m)$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

        plotting.plot_2D_heatmap(df=dfd0,
                                 pxyz=(px, py, 'dr_mean'),
                                 savepath=join(path_results_xy, 'dr-mean_per_pid_2D__d0r.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_r,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta_{o} r \: (\mu m)$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

        plotting.plot_2D_heatmap(df=dfd0,
                                 pxyz=(px, py, 'r_strain'),
                                 savepath=join(path_results_xy, 'dr-strain_per_pid_2D__d0r.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_r,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta_{o} r / r$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

        # ---
        # ---
        plotting.plot_2D_heatmap(df=dfd,
                                 pxyz=(px, py, 'dz_mean'),
                                 savepath=join(path_results_xy, 'dz-mean_per_pid_2D__dz.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_z,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta z \: (\mu m)$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

        plotting.plot_2D_heatmap(df=dfd,
                                 pxyz=(px, py, 'dr_mean'),
                                 savepath=join(path_results_xy, 'dr-mean_per_pid_2D__dr.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_r,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r \: (\mu m)$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

        plotting.plot_2D_heatmap(df=dfd,
                                 pxyz=(px, py, 'r_strain'),
                                 savepath=join(path_results_xy, 'dr-strain_per_pid_2D__dr.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_r,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r / r$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )


    # --- --- ONE PLOT PER PID
    if (
            plot_single_pids_by_frame or
            plot_pids_dr_by_dz or
            plot_pids_by_synchronous_time_voltage or
            plot_pids_by_synchronous_time_voltage_dz_lock_in or
            plot_pids_by_synchronous_time_voltage_monitor or
            plot_pids_dz_by_voltage_hysteresis
    ):
        for pid in only_pids:
            dfpid = df[df['id'] == pid]
            dz_mean = dfd[dfd['id'] == pid]['dz_mean'].item()
            dr_mean = dfd[dfd['id'] == pid]['dr_mean'].item()

            if plot_single_pids_by_frame:
                plotting.plot_single_pid_displacement_trajectory(
                    df=dfpid,
                    pdzdr=(pdz, pdr),
                    pid=pid,
                    dzr_mean=(dz_mean, dr_mean),
                    path_results=path_results_pids_by_frame,
                )

            if plot_pids_dr_by_dz:
                plotting.plot_single_pid_dr_by_dz(
                    df=dfpid,
                    pdzdr=(pdz, pdr),
                    pid=pid,
                    path_results=path_results_pids_dr_by_dz,
                    only_frames=frames_drdz,
                )

            if plot_pids_by_synchronous_time_voltage:
                plotting.plot_pids_by_synchronous_time_voltage(
                    df=dfpid,
                    pdzdr=(pdz, pdr),
                    pid=pid,
                    path_results=path_results_pids_by_synchronous_time_voltage,
                )

            if plot_pids_by_synchronous_time_voltage_dz_lock_in:
                plotting.plot_pids_by_synchronous_time_voltage_lock_in(
                    df=dfpid,
                    pdz='d0z',
                    dict_test=dict_test,
                    pid=pid,
                    path_results=path_results_pids_by_synchronous_time_voltage_dz_lock_in,
                )
                plotting.plot_pids_by_synchronous_time_voltage_lock_in(
                    df=dfpid,
                    pdz='dz',
                    dict_test=dict_test,
                    pid=pid,
                    path_results=path_results_pids_by_synchronous_time_voltage_dz_lock_in,
                )

            if plot_pids_by_synchronous_time_voltage_monitor:
                plotting.plot_pids_by_synchronous_time_voltage_monitor(
                    df=dfpid,
                    pz=pdz,
                    pid=pid,
                    test_settings=dict_test,
                    path_results=path_results_pids_by_synchronous_time_voltage_monitor,
                )

            if plot_pids_dz_by_voltage_hysteresis:
                plotting.plot_pids_dzdr_by_voltage_hysteresis(
                    df=dfpid,
                    pdzdr=(pdz, pdr),
                    dict_test=dict_test,
                    pid=pid,
                    path_results=path_results_pids_dzdr_by_voltage_hysteresis)
                plotting.plot_pids_dzdr_by_voltage_hysteresis(
                    df=dfpid,
                    pdzdr=(pd0z, pd0r),
                    dict_test=dict_test,
                    pid=pid,
                    path_results=path_results_pids_dzdr_by_voltage_hysteresis)

                if plot_all_dz_lock_ins:
                    plotting.plot_pids_dz_by_voltage_hysteresis(
                        df=dfpid,
                        pdz=pd0z_lock_in,
                        dict_test=dict_test,
                        pid=pid,
                        path_results=path_results_pids_dz_by_voltage_hysteresis)

            # ---

    # ---

    # --- --- ONE PLOT PER FRAME


    # --- plot 2D heat maps
    if plot_1D_z_by_r_by_frame or plot_1D_dz_by_r_by_frame or plot_2D_z_by_frame or plot_2D_dz_by_frame or plot_2D_dr_by_frame:
        rmin, rmax = 0, df[pr].max()
        zmin, zmax = df[pz].min(), df[pz].max()
        dzmin, dzmax = df[pdz].min(), df[pdz].max()
        drmin, drmax = df[pdr].min(), df[pdr].max()
        # ---
        for frame in animate_frames:
            df_frame = df[df['frame'] == frame]
            # -
            if plot_1D_z_by_r_by_frame:
                fig, ax = plt.subplots(figsize=(5, 2.75))
                ax.plot(df_frame[pr], df_frame[pz], 'o', ms=1.5)
                ax.set_xlabel(r'$r \: (\mu m)$')
                ax.set_xlim(rmin, rmax + 5)
                ax.set_ylabel(r'$z \: (\mu m)$')
                ax.set_ylim(zmin - 2.5, zmax + 2.5)
                ax.grid(alpha=0.25)
                ax.set_title('frame: {:03d}'.format(frame))
                plt.tight_layout()
                plt.savefig(join(path_results_1D_z_by_r_by_frame, 'z-r_by_fr{:03d}.png'.format(frame)),
                            dpi=300, facecolor='w')
                plt.close()
            # ---
            if plot_1D_dz_by_r_by_frame:
                fig, ax = plt.subplots(figsize=(5, 2.75))
                ax.plot(df_frame[pr], df_frame[pdz], 'o', ms=1.5)
                ax.set_xlabel(r'$r \: (\mu m)$')
                ax.set_xlim(rmin, rmax + 5)
                ax.set_ylabel(r'$\Delta z \: (\mu m)$')
                ax.set_ylim(dzmin - 2.5, dzmax + 2.5)
                ax.grid(alpha=0.25)
                ax.set_title('frame: {:03d}'.format(frame))
                plt.tight_layout()
                plt.savefig(join(path_results_1D_dz_by_r_by_frame, 'dz-r_by_fr{:03d}.png'.format(frame)),
                            dpi=300, facecolor='w')
                plt.close()
            # ---
            if plot_2D_z_by_frame:
                plotting.plot_2D_heatmap(df=df_frame,
                                         pxyz=(px, py, pz),
                                         savepath=join(path_results_2D_z_by_frame, 'xy-z_fr{:03d}.png'.format(frame)),
                                         field=(0, dict_settings['field_of_view']),
                                         interpolate='linear',
                                         levels=np.round(np.linspace(zmin, zmax, levels_z)),
                                         units=(r'$(\mu m)$', r'$(\mu m)$', r'$z \: (\mu m)$'),
                                         title='frame: {:03d}'.format(frame),
                                         )
            # ---
            if plot_2D_dz_by_frame:
                plotting.plot_2D_heatmap(df=df_frame,
                                         pxyz=(px, py, pdz),
                                         savepath=join(path_results_2D_dz_by_frame, 'xy-dz_fr{:03d}.png'.format(frame)),
                                         field=(0, dict_settings['field_of_view']),
                                         interpolate='linear',
                                         levels=np.round(np.linspace(dzmin, dzmax, levels_z)),
                                         units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta z \: (\mu m)$'),
                                         title='frame: {:03d}'.format(frame),
                                         )
            # ---
            if plot_2D_dr_by_frame:
                plotting.plot_2D_heatmap(df=df_frame,
                                         pxyz=(px, py, pdr),
                                         savepath=join(path_results_2D_dr_by_frame, 'xy-dr_fr{:03d}.png'.format(frame)),
                                         field=(0, dict_settings['field_of_view']),
                                         interpolate='linear',
                                         levels=np.round(np.linspace(drmin, drmax, levels_r), 1),
                                         units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r \: (\mu m)$'),
                                         title='frame: {:03d}'.format(frame),
                                         )

    # ---

    # --- --- ONE PLOT PER MULTIPLE: PIDS, FRAMES, ETC.

    # plotting.plot_multi_single_pid_displacement_trajectory(df_, px=X, py=pdz, path_results=path_results_pids_by_X)

    if (
            plot_1D_dz_by_r_by_frame_with_surface_profile or
            plot_1D_dz_by_r_by_frois_with_surface_profile or
            plot_normalized_membrane_profile
    ):
        # --- plot surface plot: 1 figure = several profiles
        if plot_1D_dz_by_r_by_frois_with_surface_profile:
            df_frois = df[df['frame'].isin(frois_overlay)]
            plotting.plot_dz_by_r_by_frois_with_surface_profile(
                df=df_frois,
                przdr=przdr,
                dict_surf=dict_surface_profilometry,
                frames=frois_overlay,
                path_save=path_results_1D_dz_by_r_by_frois_w_surf,
                dict_fit=dict_fit_memb,
            )

        # --- plot normalized membrane profile
        if plot_normalized_membrane_profile:
            df_frois = df[(df['frame'].isin(frois_norm_profile))]
            plotting.plot_dz_by_r_by_frois_normalize_membrane_profile(
                df=df_frois,
                przdr=przdr,
                dict_surf=dict_surface_profilometry,
                frames=frois_norm_profile,
                path_save=path_results_1D_dznorm_by_r_by_frois,
            )

        if plot_1D_dz_by_r_by_frame_with_surface_profile:
            # --- plot surface profile w/ 3D DPT coords
            plotting.plot_dz_by_r_by_frame_with_surface_profile(
                df,
                przdr=przdr,
                dict_surf=dict_surface_profilometry,
                frames=animate_frames,
                path_save=path_results_1D_dz_by_r_by_frame_w_surf,
                dict_fit=dict_fit_memb,
                dr_ampl=dr_ampl,
                show_interface=show_zipping_interface,
                temporal_display_units='frame',
                title=test_description,
            )


