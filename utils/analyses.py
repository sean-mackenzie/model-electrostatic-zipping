import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import smu, dpt
from utils import plotting

"""
df = DF
xyms = ['g', 'm']
tid = TID
PLOT_COORDS = True
path_root = PATH_REPRESENTATIVE
dict_test
"""

def second_pass(df, xym, tid, dict_settings, dict_test, path_results):
    # assign simple column labels
    # xym = 'g'  # options: 'g': sub-pixel localization using 2D Gaussian; 'm': discrete localization using cross-corr
    px, py, pr, pdx, pdy, pdr = [k + xym for k in ['x', 'y', 'r', 'dx', 'dy', 'dr']]
    pz, pdz = 'z', 'dz'
    # -
    # modifiers
    eval_pids_drz = True  # True: calculate/export net-displacement per-particle in r- and z-directions
    plot_pids_by_frame = True  # True: plot particle z-trajectories
    plot_pids_by_synchronous_time_voltage = True
    plot_pids_dz_by_voltage_ascending = True
    plot_pids_dz_by_voltage_hysteresis = True
    plot_heatmaps = True  # True: plot 2D heat map (requires eval_pids_dz having been run)
    # -
    plot_1D_z_by_r_by_frame = True
    plot_1D_dz_by_r_by_frame = True
    plot_2D_z_by_frame = True
    plot_2D_dz_by_frame = True
    plot_2D_dr_by_frame = True
    if xym == 'm':
        plot_1D_z_by_r_by_frame = False
        plot_1D_dz_by_r_by_frame = False
        plot_2D_z_by_frame = False
        plot_2D_dz_by_frame = False
    # for contourf plots
    levels_z = 40
    levels_r = 25

    """
    ANALYSES TO ADD:
        1. Identifying the zipping interface
            * How? See next. 
        2. It would be interesting to add a plot showing: VELOCITY (i.e., change in position wrt time)
            * Why? Because, it should reveal the zipping interface!
                ** When a particle is on the suspended membrane, it is moving (i.e., VELOCITY != 0)
                ** When a particle zips against the sidewall, it effectively becomes stationary (i.e., V = 0!)
                ** So, a 1D plot of velocity vs. radius, or, 2D contour plot of velocity vs. (x, y),
                    should reveal the zipping interface. 
            * But here is what may be really, really interesting:
                ** Knowing the exact position of the zipping interface, we can evaluate SLIPPAGE ALONG THE SIDEWALL! 
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
    # make directories
    pths = [path_results_rep,
            path_results_pids_by_frame, path_results_pids_by_synchronous_time_voltage,
            path_results_pids_dz_by_voltage_ascending, path_results_pids_dz_by_voltage_hysteresis,
            path_results_1D_z_by_r_by_frame, path_results_1D_dz_by_r_by_frame,
            path_results_2D_z_by_frame, path_results_2D_dz_by_frame, path_results_2D_dr_by_frame]
    mods = [True, plot_pids_by_frame, plot_pids_by_synchronous_time_voltage,
            plot_pids_dz_by_voltage_ascending, plot_pids_dz_by_voltage_hysteresis,
            plot_1D_z_by_r_by_frame, plot_1D_dz_by_r_by_frame,
            plot_2D_z_by_frame, plot_2D_dz_by_frame, plot_2D_dr_by_frame]
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
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta z \: (\mu m)$')
                                 )
        plotting.plot_2D_heatmap(df=dfd,
                                 pxyz=(px, py, 'dr_mean'),
                                 savepath=join(path_results_rep, 'dr-mean_per_pid_2D.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_r,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r \: (\mu m)$'),
                                 )

        plotting.plot_2D_heatmap(df=dfd,
                                 pxyz=(px, py, 'r_strain'),
                                 savepath=join(path_results_rep, 'dr-strain_per_pid_2D.png'),
                                 field=(0, dict_settings['field_of_view']),
                                 interpolate='linear',
                                 levels=levels_r,
                                 units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r / r$'),
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

    # --- plot 2D heat maps for each frame
    if plot_1D_z_by_r_by_frame or plot_1D_dz_by_r_by_frame or plot_2D_z_by_frame or plot_2D_dz_by_frame or plot_2D_dr_by_frame:
        rmin, rmax = 0, df[pr].max()
        zmin, zmax = df[pz].min(), df[pz].max()
        dzmin, dzmax = df[pdz].min(), df[pdz].max()
        drmin, drmax = df[pdr].min(), df[pdr].max()
        # ---
        for frame in np.arange(dict_test['dpt_start_frame'], dict_test['dpt_end_frames'][1] + 1):
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