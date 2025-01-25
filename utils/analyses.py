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
    px, py, pr, pdx, pdy, pdr = [k + xym for k in ['x', 'y', 'r', 'dx', 'dy', 'dr']]
    pz, pdz = 'z', 'dz'
    # -
    # modifiers
    eval_pids_drz = False  # True: calculate/export net-displacement per-particle in r- and z-directions
    plot_heatmaps = False  # True: plot 2D heat map (requires eval_pids_dz having been run)
    plot_pids_by_frame = False  # If you have voltage data, False
    plot_pids_by_synchronous_time_voltage = False
    plot_pids_dz_by_voltage_ascending = False  # Should probably always be False
    plot_pids_dz_by_voltage_hysteresis = False
    # -
    plot_1D_z_by_r_by_frame = False
    plot_1D_dz_by_r_by_frame = False
    plot_2D_z_by_frame = False
    plot_2D_dz_by_frame = False
    plot_2D_dr_by_frame = False
    # -
    plot_1D_dz_by_r_by_frame_with_surface_profile = True
    # for contourf plots
    levels_z = 15
    levels_r = 10
    # -
    if xym == 'm':
        plot_pids_by_synchronous_time_voltage = False
        plot_pids_dz_by_voltage_ascending = False
        plot_pids_dz_by_voltage_hysteresis = False
        plot_1D_z_by_r_by_frame = False
        plot_1D_dz_by_r_by_frame = False
        plot_2D_z_by_frame = False
        plot_2D_dz_by_frame = False
        plot_1D_dz_by_r_by_frame_with_surface_profile = False
    """
    ANALYSES TO ADD: 
        1. Plot circle to show through-hole in 2D heat maps to understand its effects
        2. Plot concentric rings around r = 0 in 2D heat maps to show radial coordinate
        3. Start writing that paper!
            * Need to focus on what is going into the paper (with the data I have now)
            * And what other experiments would be useful/interesting (e.g., AC voltage)
            * I think one of the biggest contributions would be solving the "collapse" issue
                ** Does an AC voltage "solve" (or at least help) the collapse issue?
                ** What about very short DC pulses that alternate polarity (i.e., bipolar)?
    """

    # -
    path_results_rep = join(path_results.format(tid), 'xy' + xym)
    path_results_pids_by_frame = join(path_results_rep, 'pids_by_frame')
    path_results_pids_by_synchronous_time_voltage = join(path_results_rep, 'pids_by_sync-time-volts')
    path_results_pids_dz_by_voltage_ascending = join(path_results_rep, 'pids_dz_by_voltage_ascending')
    path_results_pids_dz_by_voltage_hysteresis = join(path_results_rep, 'pids_dz_by_voltage_hysteresis')
    path_results_1D_z_by_r_by_frame = join(path_results_rep, '1D_z-r_by_frame')
    path_results_1D_dz_by_r_by_frame = join(path_results_rep, '1D_dz-r_by_frame')
    path_results_2D_z_by_frame = join(path_results_rep, '2D_z_by_frame')
    path_results_2D_dz_by_frame = join(path_results_rep, '2D_dz_by_frame')
    path_results_2D_dr_by_frame = join(path_results_rep, '2D_dr_by_frame')
    # path_results_1D_dz_by_r_by_frame_w_surf = join(path_results_rep, '1D_dz-r_by_frame_w-surface')
    path_results_1D_dz_by_r_by_frame_w_surf = join(path_results_rep, '1D_dzdr-r_by_frame_w-surf+drX+raytracing')
    # make directories
    pths = [path_results_rep,
            path_results_pids_by_frame, path_results_pids_by_synchronous_time_voltage,
            path_results_pids_dz_by_voltage_ascending, path_results_pids_dz_by_voltage_hysteresis,
            path_results_1D_z_by_r_by_frame, path_results_1D_dz_by_r_by_frame,
            path_results_2D_z_by_frame, path_results_2D_dz_by_frame, path_results_2D_dr_by_frame,
            path_results_1D_dz_by_r_by_frame_w_surf]
    mods = [True, plot_pids_by_frame, plot_pids_by_synchronous_time_voltage,
            plot_pids_dz_by_voltage_ascending, plot_pids_dz_by_voltage_hysteresis,
            plot_1D_z_by_r_by_frame, plot_1D_dz_by_r_by_frame,
            plot_2D_z_by_frame, plot_2D_dz_by_frame, plot_2D_dr_by_frame,
            plot_1D_dz_by_r_by_frame_with_surface_profile]
    pths = [x for x, y in zip(pths, mods) if y is True]
    for pth in pths:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # ---
    # initialize some variables
    dfd = None

    # --- evaluate net displacement per-pid
    if eval_pids_drz:
        dfd = dpt.calculate_net_displacement(
            df=df,
            pxyrz=(px, py, pr, pz),
            start_frame=dict_test['dpt_start_frame'],
            end_frames=dict_test['dpt_end_frames'],
            path_results=path_results_rep,
        )

    # --- plot 2D heat map
    if plot_heatmaps:
        if dfd is None:
            dfd = pd.read_excel(join(path_results_rep, 'net-dzr_per_pid.xlsx'))
        # ---
        plotting.plot_2D_heatmap(df=dfd,
                                 pxyz=(px, py, 'dz_mean'),
                                 savepath=join(path_results_rep, 'dz-mean_per_pid_2D.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_z,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta z \: (\mu m)$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

        plotting.plot_2D_heatmap(df=dfd,
                                 pxyz=(px, py, 'dr_mean'),
                                 savepath=join(path_results_rep, 'dr-mean_per_pid_2D.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_r,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r \: (\mu m)$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

        plotting.plot_2D_heatmap(df=dfd,
                                 pxyz=(px, py, 'r_strain'),
                                 savepath=join(path_results_rep, 'dr-strain_per_pid_2D.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_r,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r / r$'),
                                 overlay_circles=True,
                                 dict_settings=dict_settings,
                                 )

    # --- plot displacement trajectories
    if plot_pids_by_frame:
        if dfd is None:
            dfd = pd.read_excel(join(path_results_rep, 'net-dzr_per_pid.xlsx'))
        for pid in df.sort_values(by=pr, ascending=True)['id'].unique():
            dfpid = df[df['id'] == pid]
            dz_mean = dfd[dfd['id'] == pid]['dz_mean'].item()
            dr_mean = dfd[dfd['id'] == pid]['dr_mean'].item()
            plotting.plot_single_pid_displacement_trajectory(
                df=dfpid,
                pdzdr=(pdz, pdr),
                pid=pid,
                dzr_mean=(dz_mean, dr_mean),
                path_results=path_results_pids_by_frame,
            )

    # --- plot synchronous coords + voltage
    if plot_pids_by_synchronous_time_voltage or plot_pids_dz_by_voltage_ascending or plot_pids_dz_by_voltage_hysteresis:
        for pid in df.sort_values(by=pr, ascending=True)['id'].unique():
            dfpid = df[df['id'] == pid]
            if plot_pids_by_synchronous_time_voltage:
                plotting.plot_pids_by_synchronous_time_voltage(
                    df=dfpid,
                    pdzdr=(pdz, pdr),
                    pid=pid,
                    path_results=path_results_pids_by_synchronous_time_voltage,
                )
            if plot_pids_dz_by_voltage_ascending:
                plotting.plot_pids_dz_by_voltage_ascending(
                    df=dfpid,
                    pdzdr=(pdz, pdr),
                    dict_test=dict_test,
                    pid=pid,
                    path_results=path_results_pids_dz_by_voltage_ascending,
                )
            if plot_pids_dz_by_voltage_hysteresis:
                plotting.plot_pids_dz_by_voltage_hysteresis(
                    df=dfpid,
                    pdzdr=(pdz, pdr),
                    dict_test=dict_test,
                    pid=pid,
                    path_results=path_results_pids_dz_by_voltage_hysteresis)

            # ---

    # ---

    # --- -- FOR EACH FRAME

    # --- plot 2D heat maps
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

    # --- -- PLOT SURFACE PROFILE WITH 3D DPT COORDS, FOR EACH FRAME

    # --- plot 2D heat maps
    if plot_1D_dz_by_r_by_frame_with_surface_profile:
        # read surface profile
        df_surface = empirical.read_surface_profile(dict_settings, subset='right_half', hole=True, fid_override=None)
        sr, sz = 'r', 'z'
        surfr, surfz = df_surface[sr].to_numpy(), df_surface[sz].to_numpy()
        # -
        # get 3D DPT limits
        rmin, rmax = 0, df[pr].max()
        zmin, zmax = df[pz].min(), df[pz].max()
        dzmin, dzmax = df[pdz].min(), df[pdz].max()
        drmin, drmax = df[pdr].min(), df[pdr].max()
        # -
        # make "amplified radial displacement" column
        # and set color scale
        df['r_dr'] = df[pr] + df[pdr] * 10
        vmin, vmax = df[pdr].min(), df[pdr].max()
        # ---
        frames = np.arange(60, 71)  # dict_test['dpt_start_frame'][1], dict_test['dpt_end_frames'][0] + 1):
        for frame in frames:
            df_frame = df[df['frame'] == frame].sort_values(pr)
            # -
            if plot_1D_dz_by_r_by_frame_with_surface_profile:
                x = df_frame[pr].to_numpy()
                y = df_frame[pdz].to_numpy()
                xnew, ynew, xf, yf = fit.wrapper_fit_radial_membrane_profile(x=x, y=y, s=3000,
                    dict_settings=dict_settings, faux_r_zero=True, faux_r_edge=False,
                )

                x = df_frame['r_dr'].to_numpy()

                fig, ax = plt.subplots(figsize=(5, 2.75))

                # ---
                # ---
                if frame > frames[1]:
                    df_last = df[df['frame'] == frame - 1]
                    df_two = pd.concat([df_last, df_frame])
                    df_two = df_two.sort_values('frame')
                    for pid in df_two['id'].unique():
                        dfpid = df_two[df_two['id'] == pid]
                        ax.plot(dfpid['r_dr'], dfpid[pdz], '-', color='gray', lw=0.25, alpha=0.25, zorder=3.1)
                    if frame > frames[2]:
                        df_last2 = df[df['frame'] == frame - 2]
                        df_three = pd.concat([df_two, df_last2])
                        df_three = df_three.sort_values('frame')
                        for pid in df_three['id'].unique():
                            dfpid = df_three[df_three['id'] == pid]
                            ax.plot(dfpid['r_dr'], dfpid[pdz], '-', color='gray', lw=0.25, alpha=0.25, zorder=3.1)
                # ---

                ax.plot(surfr, surfz, '-', color='gray', lw=0.5, label='SP', zorder=3.1)
                ax.scatter(x, y, c=df_frame[pdr], s=5, cmap='coolwarm', vmin=vmin, vmax=vmax, label='FPs', zorder=3.3)
                ax.plot(xnew, ynew, 'r-', lw=0.5, label='Fit', zorder=3.2)
                ax.set_xlabel(r'$r \: (\mu m)$')
                ax.set_xlim(rmin, rmax + 5)
                ax.set_ylabel(r'$\Delta z \: (\mu m)$')
                ax.set_ylim(dzmin - 2.5, dzmax + 2.5)
                ax.grid(alpha=0.0625)
                ax.set_title('frame: {:03d}'.format(frame))
                plt.tight_layout()
                plt.savefig(join(path_results_1D_dz_by_r_by_frame_w_surf, 'dz-r_by_fr{:03d}_+dr10X.png'.format(frame)),
                            dpi=300, facecolor='w')
                plt.close()
            # ---