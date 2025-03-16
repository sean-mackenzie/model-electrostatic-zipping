import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dpt
from utils import plotting, empirical, fit


def second_pass(df, xym, tid, dict_settings, dict_test, path_results):
    # assign simple column labels
    # xym = 'g'  # options: 'g': sub-pixel localization using 2D Gaussian; 'm': discrete localization using cross-corr
    px, py, pr, pdx, pdy, pdr, pd0x, pd0y, pd0r = [k + xym for k in ['x', 'y', 'r', 'dx', 'dy', 'dr', 'd0x', 'd0y', 'd0r']]
    pz, pdz, pd0z = 'z', 'dz', 'd0z'
    przdr = (pr, pd0z, pdr)  # for plotting 3D DPT coords overlay surface profile
    # -
    if 'animate_frames' in dict_test.keys():
        frames_with_surface_profile = np.arange(dict_test['animate_frames'][0], dict_test['animate_frames'][1])
    elif 'dpt_start_frame' in dict_test.keys() and 'dpt_end_frames' in dict_test.keys():
        frames_with_surface_profile = np.arange(dict_test['dpt_start_frame'][1], dict_test['dpt_end_frames'][0])
    else:
        frames_with_surface_profile = np.arange(5, 125)
    frames_quiver = frames_with_surface_profile
    frames_drdz = frames_with_surface_profile
    # -
    only_pids = []  # if None, then plot all pids
    not_pids = [18]  # [], then plot all pids
    # -
    # modifiers (True False)
    eval_pids_drz = False  # True: calculate/export net-displacement per-particle in r- and z-directions
    average_max_n_positions = 20
    plot_depth_dependent_in_plane_stretch = True
    plot_all_pids_by_X, Xs = False, ['frame', 't_sync', 'STEP', 'VOLT']
    plot_heatmaps = False  # True: plot 2D heat map (requires eval_pids_dz having been run)
    plot_single_pids_by_frame = False  # If you have voltage data, generally False. Can be useful to align by "frame" (not time)
    plot_pids_dr_by_dz = False
    plot_pids_by_synchronous_time_voltage = False
    plot_pids_by_synchronous_time_voltage_monitor = False  # AC only
    plot_pids_dz_by_voltage_hysteresis = False
    # ---
    # --- --- PLOT X-Y DISPLACEMENTS: FIELD OF VIEW
    plot_scatter_xy_by_frame = False
    plot_quiver_xy_by_frame = False
    # -
    # --- --- PLOT R-Z DISPLACEMENTS: CROSS-SECTION VIEW
    # plot normalized profile to visualize bending of suspended membrane
    plot_normalized_membrane_profile, frois_norm_profile = False, [8, 50, 70]
    # plot frames of interest overlay
    plot_1D_dz_by_r_by_frois_with_surface_profile, frois_overlay = False, [8, 83, 84, 104, 179]
    # plot every frame
    plot_1D_dz_by_r_by_frame_with_surface_profile = False
    fit_spline_to_memb = False
    dr_ampl = 1
    if plot_1D_dz_by_r_by_frame_with_surface_profile or plot_quiver_xy_by_frame:
        surf_profile_subset = 'left_half'  # 'full', 'left_half', 'right_half'
        surf_fid_override = None
        surf_shift_r, surf_shift_z, surf_scale_z = 0, 0, 1

        df_surface = empirical.read_surface_profile(
            dict_settings,
            subset=surf_profile_subset,
            hole=True,
            fid_override=surf_fid_override,
        )
        dict_surface_profilometry = {'r': df_surface['r'].to_numpy(), 'z': df_surface['z'].to_numpy(),
                                     'dr': surf_shift_r, 'dz': surf_shift_z, 'scale_z': surf_scale_z}
    if fit_spline_to_memb:
        # get radial position of n-th inner-most particle
        faux_is_average_of_n = 5
        faux_r_zero = df[df['frame'] == frames_with_surface_profile[0]].sort_values(pr, ascending=True).iloc[faux_is_average_of_n][pr] + 1.5
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
    plot_2D_dz_by_frame = False
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
    if 'path_model_strain' in dict_settings.keys() and plot_depth_dependent_in_plane_stretch is True:
        df_model = pd.read_excel(dict_settings['path_model_strain'])
        # df_model = df_model[df_model['dZ'] < df_model['dZ'].max() - 0.75e-6]
    else:
        df_model = None
    # -
    path_results_rep = join(path_results.format(tid), 'xy' + xym)
    path_results_per_pid = join(path_results_rep, 'per_pid')
    path_results_slice = join(path_results_rep, 'slice')
    path_results_xy = join(path_results_rep, 'field-of-view')
    path_results_rz = join(path_results_rep, 'cross-section')
    # data slice
    path_results_depth_dependent_in_plane_stretch = join(path_results_slice, 'depth-dependent-in-plane-stretch')
    path_results_pids_by_X = join(path_results_slice, 'pids_by_X')
    # particle trajectories
    path_results_pids_by_frame = join(path_results_per_pid, 'pids_by_frame')
    path_results_pids_dr_by_dz = join(path_results_per_pid, 'pids_dr_by_dz')
    path_results_pids_by_synchronous_time_voltage = join(path_results_per_pid, 'pids_by_sync-time-volts')
    path_results_pids_by_synchronous_time_voltage_monitor = join(path_results_per_pid, 'pids_by_sync-time-volts-monitor')
    path_results_pids_dz_by_voltage_hysteresis = join(path_results_per_pid, 'pids_dz_by_voltage_hysteresis')
    # field-of-view
    path_results_scatter_xy_by_frame = join(path_results_xy, 'scatter_xy_by_frame')
    path_results_quiver_xy_by_frame = join(path_results_xy, 'quiver_xy_by_frame')
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
            path_results_depth_dependent_in_plane_stretch, path_results_pids_by_X,
            path_results_pids_by_frame, path_results_pids_dr_by_dz,
            path_results_pids_by_synchronous_time_voltage, path_results_pids_by_synchronous_time_voltage_monitor,
            path_results_pids_dz_by_voltage_hysteresis,
            path_results_scatter_xy_by_frame, path_results_quiver_xy_by_frame,
            path_results_1D_z_by_r_by_frame, path_results_1D_dz_by_r_by_frame,
            path_results_2D_z_by_frame, path_results_2D_dz_by_frame, path_results_2D_dr_by_frame,
            path_results_1D_dznorm_by_r_by_frois,
            path_results_1D_dz_by_r_by_frame_w_surf, path_results_1D_dz_by_r_by_frois_w_surf]
    mods = [True, True, True, True,
            plot_depth_dependent_in_plane_stretch, plot_all_pids_by_X,
            plot_single_pids_by_frame, plot_pids_dr_by_dz,
            plot_pids_by_synchronous_time_voltage, plot_pids_by_synchronous_time_voltage_monitor,
            plot_pids_dz_by_voltage_hysteresis,
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

    # --- plot only a subset of particles
    # removes "bad" particles from visualizations, or
    # includes only "good" particles
    if only_pids:
        df = df[df['id'].isin(only_pids)]
    elif not_pids:
        df = df[~df['id'].isin(not_pids)]
    only_pids = df['id'].unique()
    if eval_pids_drz is False:
        dfd = pd.read_excel(join(path_results_rep, 'net-dzr_per_pid.xlsx'))
        dfd0 = pd.read_excel(join(path_results_rep, 'net-d0zr_per_pid.xlsx'))
    dfd = dfd[dfd['id'].isin(only_pids)]
    dfd0 = dfd0[dfd0['id'].isin(only_pids)]

    # --- --- SINGLE PLOTS

    # --- SLICE
    if (plot_depth_dependent_in_plane_stretch or plot_all_pids_by_X):
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

            if df_model is not None:
                plotting.compare_depth_dependent_in_plane_stretch_with_model(
                    dfd=dfd0,
                    dfm=df_model,
                    path_results=path_results_depth_dependent_in_plane_stretch,
                    save_id='dfd0',
                )
                plotting.compare_depth_dependent_in_plane_stretch_with_model(
                    dfd=dfd,
                    dfm=df_model,
                    path_results=path_results_depth_dependent_in_plane_stretch,
                    save_id='dfd',
                )


        if plot_all_pids_by_X:
            for X in Xs:
                plotting.plot_all_pids_displacement_trajectory(
                    df=df,
                    px=X,
                    pdzdr=(pdz, pdr),
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
            pxydxdy=(px, py, pd0x, pd0y),
            pcolor_dxdy=pd0r,
            frames=frames_quiver,
            scale=5,  # arrow lengths are 5X larger than data units
            savepath=join(path_results_quiver_xy_by_frame, 'quiver_xyd0xd0y_fr{}.png'),
            field=(0, dict_settings['field_of_view']),
            units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta z \: (\mu m)$'),
            overlay_circles=True,
            dict_settings=dict_settings,
            show_interface=True,
            dict_surf=dict_surface_profilometry,
            prz=(pr, pd0z),
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

            if plot_pids_by_synchronous_time_voltage_monitor:
                plotting.plot_pids_by_synchronous_time_voltage_monitor(
                    df=dfpid,
                    pz=pdz,
                    pid=pid,
                    test_settings=dict_test,
                    path_results=path_results_pids_by_synchronous_time_voltage_monitor,
                )

            if plot_pids_dz_by_voltage_hysteresis:
                plotting.plot_pids_dz_by_voltage_hysteresis(
                    df=dfpid,
                    pdzdr=(pdz, pdr),
                    dict_test=dict_test,
                    pid=pid,
                    path_results=path_results_pids_dz_by_voltage_hysteresis)
                plotting.plot_pids_dz_by_voltage_hysteresis(
                    df=dfpid,
                    pdzdr=(pd0z, pd0r),
                    dict_test=dict_test,
                    pid=pid,
                    path_results=path_results_pids_dz_by_voltage_hysteresis)

            # ---

    # ---

    # --- --- ONE PLOT PER FRAME
    if plot_1D_dz_by_r_by_frame_with_surface_profile:
        # --- plot surface profile w/ 3D DPT coords
        plotting.plot_dz_by_r_by_frame_with_surface_profile(
            df,
            przdr=przdr,
            dict_surf=dict_surface_profilometry,
            frames=frames_with_surface_profile,
            path_save=path_results_1D_dz_by_r_by_frame_w_surf,
            dict_fit=dict_fit_memb,
            dr_ampl=dr_ampl,
            show_interface=True
        )

        # --- plot 2D heat maps
        plot_2d_heatmaps_per_frame = False
        if plot_2d_heatmaps_per_frame:
            if plot_1D_z_by_r_by_frame or plot_1D_dz_by_r_by_frame or plot_2D_z_by_frame or plot_2D_dz_by_frame or plot_2D_dr_by_frame:
                rmin, rmax = 0, df[pr].max()
                zmin, zmax = df[pz].min(), df[pz].max()
                dzmin, dzmax = df[pdz].min(), df[pdz].max()
                drmin, drmax = df[pdr].min(), df[pdr].max()
                # ---
                for frame in np.arange(dict_test['dpt_start_frame'][1], dict_test['dpt_end_frames'][0] + 1):
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
                frames=frois_norm_profile,
                path_save=path_results_1D_dznorm_by_r_by_frois,
            )


