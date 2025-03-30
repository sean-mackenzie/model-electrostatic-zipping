# tests/test_model_sweep.py

# imports
import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from tifffile import imread
from skimage.transform import rescale

import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from utils.shapes import complete_erf_profile
from utils.empirical import read_surface_profile, get_zipping_interface_rz
from utils.fit import wrapper_fit_radial_membrane_profile


# -

# ----------------------------------------------------------------------------------------------------------------------
# BELOW: PLOT SETTINGS VERIFICATION
# ----------------------------------------------------------------------------------------------------------------------


def plot_scatter_with_pid_labels(df, pxy, dict_settings, savepath):
    px, py = pxy
    x, y, id_ = df[px].to_numpy(), df[py].to_numpy(), df['id'].to_numpy()
    mpp = dict_settings['microns_per_pixel']

    # plot
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    # plot 1: units = microns
    ax1.scatter(x, y, c=id_, s=2)
    for xi, yi, pid in zip(x, y, id_):
        ax1.text(xi, yi, str(pid), color='black', fontsize=8)

    # make circular patches for inner and outer radii
    circle_edge = Circle(dict_settings['xyc_microns'], dict_settings['radius_microns'])
    circle_hole = Circle(dict_settings['xyc_microns'], dict_settings['radius_hole_microns'])
    patches = [circle_edge, circle_hole]
    pc = PatchCollection(patches, fc='none', ec='k', lw=0.5, ls='--', alpha=0.5)
    ax1.add_collection(pc)

    ax1.set_xlim([0, dict_settings['field_of_view']])
    ax1.set_xticks([0, dict_settings['field_of_view']])
    ax1.set_xlabel(r'$x \: (\mu m)$')
    ax1.set_ylim([0, dict_settings['field_of_view']])
    ax1.set_yticks([0, dict_settings['field_of_view']])
    ax1.set_ylabel(r'$y \: (\mu m)$')
    ax1.invert_yaxis()
    ax1.set_aspect('equal')

    # plot 2: units = pixels
    x, y = x / mpp, y / mpp
    ax2.scatter(x, y, c=id_, s=2)
    for xi, yi, pid in zip(x, y, id_):
        ax2.text(xi, yi, str(pid), color='black', fontsize=8)

    # make circular patches for inner and outer radii
    circle_edge = Circle(dict_settings['xyc_pixels'], dict_settings['radius_pixels'])
    circle_hole = Circle(dict_settings['xyc_pixels'], dict_settings['radius_hole_pixels'])
    patches = [circle_edge, circle_hole]
    pc = PatchCollection(patches, fc='none', ec='k', lw=0.5, ls='--', alpha=0.5)
    ax2.add_collection(pc)

    ax2.set_xlim([0, dict_settings['field_of_view'] / mpp])
    ax2.set_xticks([0, dict_settings['field_of_view'] / mpp])
    ax2.set_xlabel(r'$x \: (pix)$')
    ax2.set_ylim([0, dict_settings['field_of_view'] / mpp])
    ax2.set_yticks([0, dict_settings['field_of_view'] / mpp])
    ax2.set_ylabel(r'$y \: (pix)$')
    ax2.invert_yaxis()
    ax2.set_aspect('equal')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor='white', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_scatter_on_image(df, pxy, dict_settings, savepath):
    px, py = pxy
    x, y, id_ = df[px].to_numpy(), df[py].to_numpy(), df['id'].to_numpy()
    mpp = dict_settings['microns_per_pixel']

    # image
    img = imread(dict_settings['path_image_overlay'])
    if len(img.shape) == 3:
        img = np.mean(img, axis=0)

    # plot
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    # plot 1: units = microns
    ax1.scatter(x, y, s=2, color='r')
    ax1.imshow(rescale(img, mpp), cmap='gray', alpha=0.9)
    # make circular patches for inner and outer radii
    circle_edge = Circle(dict_settings['xyc_microns'], dict_settings['radius_microns'])
    circle_hole = Circle(dict_settings['xyc_microns'], dict_settings['radius_hole_microns'])
    patches = [circle_edge, circle_hole]
    pc = PatchCollection(patches, fc='none', ec='yellow', lw=0.5, ls='--', alpha=0.5)
    ax1.add_collection(pc)

    ax1.set_xlim([0, dict_settings['field_of_view']])
    ax1.set_xticks([0, dict_settings['field_of_view']])
    ax1.set_xlabel(r'$x \: (\mu m)$')
    ax1.set_ylim([0, dict_settings['field_of_view']])
    ax1.set_yticks([0, dict_settings['field_of_view']])
    ax1.set_ylabel(r'$y \: (\mu m)$')
    ax1.invert_yaxis()
    ax1.set_aspect('equal')

    # plot 2: units = pixels
    x, y = x / mpp, y / mpp
    ax2.scatter(x, y, s=2, color='r')
    ax2.imshow(img, cmap='gray', alpha=0.9)
    # make circular patches for inner and outer radii
    circle_edge = Circle(dict_settings['xyc_pixels'], dict_settings['radius_pixels'])
    circle_hole = Circle(dict_settings['xyc_pixels'], dict_settings['radius_hole_pixels'])
    patches = [circle_edge, circle_hole]
    pc = PatchCollection(patches, fc='none', ec='yellow', lw=0.5, ls='--', alpha=0.5)
    ax2.add_collection(pc)

    ax2.set_xlim([0, dict_settings['field_of_view'] / mpp])
    ax2.set_xticks([0, dict_settings['field_of_view'] / mpp])
    ax2.set_xlabel(r'$x \: (pix)$')
    ax2.set_ylim([0, dict_settings['field_of_view'] / mpp])
    ax2.set_yticks([0, dict_settings['field_of_view'] / mpp])
    ax2.set_ylabel(r'$y \: (pix)$')
    ax2.invert_yaxis()
    ax2.set_aspect('equal')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor='white', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_surface_profilometry(dict_settings, savepath):
    sr, sz = 'r', 'z'
    if 'fid_process_profile' in dict_settings.keys():
        fid = dict_settings['fid_process_profile']
    else:
        fid = dict_settings['fid']

    df1 = read_surface_profile(dict_settings, subset='full', hole=False, fid_override=fid)
    df2 = read_surface_profile(dict_settings, subset='full', hole=True, fid_override=fid)
    # -
    # original depth
    dz1 = np.round(df1[sz].max() - df1[sz].min(), 1)
    # surface-to-hole depth
    dz2 = np.round(df2[sz].max() - df2[sz].min(), 1)
    # -
    # plot
    fig, ax = plt.subplots(figsize=(5, 2.75))

    ax.plot(df1[sr], df1[sz], '-', color='r', lw=1, label='Before Via Etch')
    ax.plot(df2[sr], df2[sz], 'o', color='k', ms=1, label='Estimated Profile')
    ax.plot(df2[sr], df2[sz], '--', color='k', lw=0.5, alpha=0.25)

    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_title('fid{}: '.format(fid) + r'$\Delta z, \Delta z_{Hole}=$' +
                 ' ({}, {}) '.format(dz1, dz2) + r'$\mu m$')
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()



# ----------------------------------------------------------------------------------------------------------------------
# BELOW: PLOT EXPERIMENTAL DATA
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------
# 1D PLOTS
# --------------------------------------------------------

def pre_process_model_dZ_by_V_for_compare(dfm, mkey=None, mval=None, extend_max_voltage=0):
    # max_voltage = df[dx].abs().max()
    # data
    dx = 'VOLT'
    # model
    mx, my = 'U', 'z'
    if mkey is not None:
        dfm = dfm[dfm[mkey] == mval]
    # data pre-processing
    x = dfm[mx].to_numpy()
    y = dfm[my].to_numpy() * -1e6
    x = x[np.argmax(y):]
    y = y[np.argmax(y):]
    x0 = np.array([0])
    y0 = np.array([0])
    xf = np.array([np.max([extend_max_voltage, x.max()])])
    yf = np.array([np.min(y)])
    x = np.concatenate((x0, x, xf))
    y = np.concatenate((y0, y, yf))
    return x, y

def compare_dZ_by_V_with_model(df, dfm, path_results, save_id, mkey=None, mval=None, dz='d0z'):
    # -- setup
    # data
    dx, dy = 'VOLT', dz
    df = df[df['STEP'] <= df['STEP'].iloc[df['VOLT'].abs().idxmax()]]
    # model
    mx, my = 'U', 'z'
    if mkey is not None:
        dfm = dfm[dfm[mkey] == mval]
    # data pre-processing
    # dfm[my] = dfm[my] * -1e6
    x = dfm[mx].to_numpy()
    y = dfm[my].to_numpy() * -1e6
    x = x[np.argmax(y):]
    y = y[np.argmax(y):]
    x0 = np.array([0])
    y0 = np.array([0])
    xf = np.array([np.max([df[dx].abs().max(), x.max()])])
    yf = np.array([np.min(y)])
    x = np.concatenate((x0, x, xf))
    y = np.concatenate((y0, y, yf))
    # -
    # plot
    fig, ax = plt.subplots(figsize=(3.75, 2.5))
    ax.plot(x, y, 'r-', label='Model')
    ax.plot(df[dx].abs(), df[dy], 'ko', ms=1.5, label='Data')
    ax.set_xlabel(r'$V_{app} \: (V)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.grid(alpha=0.25)
    ax.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(
        join(path_results, f'compare_data_to_model_{save_id}_{dz}.png'),
        dpi=300,
        facecolor='w',
        bbox_inches='tight',
    )
    plt.close()


def compare_depth_dependent_in_plane_stretch_with_model(dfd, dfm, path_results, save_id):
    # --- 3D DPT data
    pdz, pdr, pr_strain = 'dz_mean', 'dr_mean', 'r_strain'
    x, y2, y3 = dfd[pdz].abs(), dfd[pr_strain], dfd[pdr]
    # --- Model data
    # for radial displacement, we want all data
    mx3 = dfm['dZ'].to_numpy() * 1e6
    my3 = dfm['disp_r_microns'].to_numpy()
    # because strain goes highly non-linear near max deflection, we must limit
    dfm = dfm[dfm['dZ'] < dfm['dZ'].max() - 2e-6]
    mx = dfm['dZ'] * 1e6
    my1 = dfm['t_f'] * 1e6
    my2a, my2b = dfm['strain_xy'], dfm['strain_z_inv']

    # --- plot

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(5, 5))

    ax1.plot(mx, my1, 'k-', label='Model')
    ax1.set_ylabel(r'$t_{memb} \: (\mu m)$')
    ax1.grid(alpha=0.15)
    ax1.legend(fontsize='small')

    # ax2.plot(mx, my2b, '--', label='Model: out-of-plane')
    ax2.plot(mx, my2a, 'k-', label='Model')
    ax2.plot(x, y2, 'ro', ms=1, label='Experiment')
    ax2.set_ylabel(r'$Strain_{in-plane}$')
    if np.min(y2) < 0.99:
        ax2.set_ylim(bottom=0.99)
    ax2.grid(alpha=0.15)
    ax2.legend(fontsize='small')

    ax3.plot(mx3, my3, 'k-', label='Model')
    ax3.plot(x, y3, 'ro', ms=1, label='Experiment')
    ax3.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax3.set_xlabel(r'$ z \: (\mu m)$')
    ax3.grid(alpha=0.15)
    ax3.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(join(path_results, f'compare_depth_dependent_in_plane_stretch_with_model_{save_id}.png'),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_depth_dependent_in_plane_stretch_from_dfd(dfd, path_results, save_id):
    pdz, pdr, pr_strain = 'dz_mean', 'dr_mean', 'r_strain'
    x, y1, y2 = dfd[pdz].abs(), dfd[pr_strain], dfd[pdr]

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(4.85, 3.3))

    ax1.plot(x, y1, 'o', ms=1, label=pdz)
    ax2.plot(x, y2, 'o', ms=1, label=pdr)

    ax1.set_ylabel('Strain (in-plane)')
    ax1.grid(alpha=0.15)

    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax2.set_xlabel(r'$ z \: (\mu m)$')
    ax2.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(join(path_results, f'depth_dependent_in_plane_stretch_{save_id}.png'),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()

def plot_all_pids_displacement_trajectory(df, px, pdzdr, path_results):
    pdz, pdr = pdzdr
    pcm = 'cm'
    pids = df.sort_values('d0z', ascending=True)['id'].unique()

    # -
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 8), nrows=3, sharex=True,
                                   gridspec_kw={'height_ratios': [1.5, 1.0, 0.5]})
    ms, lw, cs, elw = 0.75, 0.5, 1, 0.5

    for pid in pids:
        df_pid = df[df['id'] == pid]
        if px in ['VOLT', 'STEP']:
            dfstd = df_pid.groupby('STEP').std().reset_index()
            df_pid = df_pid.groupby('STEP').mean().reset_index()
            ax1.errorbar(df_pid[px], df_pid[pdz], yerr=dfstd[pdz], fmt='-o', ms=ms, lw=lw, capsize=cs, elinewidth=elw)
            ax2.errorbar(df_pid[px], df_pid[pdr], yerr=dfstd[pdr], fmt='-o', ms=ms, lw=lw, capsize=cs, elinewidth=elw)
            ax3.errorbar(df_pid[px], df_pid[pcm], yerr=dfstd[pcm], fmt='-o', ms=ms, lw=lw, capsize=cs, elinewidth=elw)
        else:
            ax1.plot(df_pid[px], df_pid[pdz], '-o', ms=ms, lw=lw, label=pid)
            if len(pids) < 10:
                # ax1.plot(df_pid[px], df_pid[pdz], '-o', ms=ms, lw=lw, label=pid)
                ax1.legend(fontsize='small')
            #else:
            #    ax1.plot(df_pid[px], df_pid[pdz], '-o', ms=ms, lw=lw)
            ax2.plot(df_pid[px], df_pid[pdr], '-o', ms=ms, lw=lw)
            ax3.plot(df_pid[px], df_pid[pcm], '-o', ms=ms, lw=lw)

    # plot dz by X
    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax1.grid(alpha=0.125)

    # plot dr by X
    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax2.grid(alpha=0.125)

    # plot cm by X
    ax3.set_ylabel(r'$c_{m}$')
    ax3.set_ylim([0, 1])
    ax3.set_yticks([0, 1])
    ax3.grid(alpha=0.125)
    ax3.set_xlabel(px)

    plt.tight_layout()
    plt.savefig(join(path_results, 'pids_by_{}.png'.format(px)),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_multi_single_pid_displacement_trajectory(df, px, py, path_results):
    pids = df.sort_values(py, ascending=False)['id'].unique()
    # -
    fig, axs = plt.subplots(figsize=(6, 10), nrows=len(pids), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, 1]})
    for ax, pid in zip(axs, pids):
        df_pid = df[df['id'] == pid]
        if px in ['VOLT', 'STEP']:
            dfstd = df_pid.groupby('STEP').std().reset_index()
            df_pid = df_pid.groupby('STEP').mean().reset_index()
            ax.errorbar(df_pid[px], df_pid[py], yerr=dfstd[py], fmt='-o', ms=1, lw=1, capsize=2, elinewidth=1)
        else:
            ax.plot(df_pid[px], df_pid[py], '-o', ms=1, lw=1, label=pid)
            ax.legend(fontsize='small', title=r'$p_{ID}$')
        ax.set_ylabel(r'$\Delta z \: (\mu m)$')
        ax.grid(alpha=0.125)
    axs[-1].set_xlabel(px)
    plt.tight_layout()
    plt.savefig(join(path_results, 'pids_by_{}_{}-pids{}.png'.format(px, py, pids)),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_single_pid_displacement_trajectory(df, pdzdr, pid, dzr_mean, path_results):
    """

    :param df:
    :param pdzdr:
    :param pid:
    :param dzr_mean:
    :param path_results:
    :return:
    """
    pdz, pdr = pdzdr
    dz_mean, dr_mean = dzr_mean
    px = 'frame'
    pcm = 'cm'
    # -
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 6), nrows=2, sharex=True,
                                   gridspec_kw={'height_ratios': [1.25, 1]})

    # plot dz by frames
    ax1.plot(df[px], df[pdz], '-o', ms=0.75, lw=0.5, label=pid)
    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax1.legend(title=r'$p_{ID}$')
    ax1.grid(alpha=0.125)
    ax1.set_title(r'$p_{ID}$' + f' {pid}:  ' + r'$\Delta z_{net}=$' + ' {} '.format(dz_mean) + r'$\mu m$')

    ax1r = ax1.twinx()
    ax1r.plot(df[px], df[pcm], '-', lw=0.5, color='gray', alpha=0.5)
    ax1r.set_ylabel(r'$c_{m}$', labelpad=-4, color='gray', alpha=0.75)
    ax1r.set_ylim([0, 1])
    ax1r.set_yticks([0, 1])

    # plot dr by frames
    ax2.plot(df[px], df[pdr], '-o', ms=0.75, lw=0.5)
    ax2.set_xlabel('Frame')
    ax2.set_xticks(np.arange(0, df[px].max() + 15, 25))
    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax2.grid(alpha=0.125)
    ax2.set_title(r'$\Delta r_{net}=$' + ' {} '.format(dr_mean) + r'$\mu m$')

    plt.tight_layout()
    plt.savefig(join(path_results, 'pid{}.png'.format(pid)), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_single_pid_dr_by_dz(df, pdzdr, pid, path_results, only_frames=None):
    if only_frames is not None:
        if isinstance(only_frames, (list, np.ndarray)):
            only_frames = (only_frames[0], only_frames[-1] + 1)
        df = df[(df['frame'] > only_frames[0]) & (df['frame'] < only_frames[1])]
    pdz, pdr = pdzdr
    # -

    # plot setup
    x_lim = df[pdr].abs().max() * 1.1

    # split into positive and negative radial displacement
    dfp = df[df[pdr] > 0]
    dfn = df[df[pdr] < 0]

    # -
    fig, ax = plt.subplots()

    # plot dz by frames
    ax.plot(df[pdr], df[pdz], '-', color='k', lw=0.5, label=r'$p_{ID}=$' + str(pid))

    ax.plot(dfp[pdr], dfp[pdz], 'ro', ms=3, label=r'$+ \Delta r$')
    ax.plot(dfn[pdr], dfn[pdz], 'bo', ms=3, label=r'$- \Delta r$')

    ax.axvline(0, color='gray', ls='--', lw=0.5)

    ax.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax.set_xlabel(r'$\Delta r \: (\mu m)$')
    ax.set_xlim([-x_lim, x_lim])
    ax.grid(alpha=0.125)
    ax.legend()

    plt.tight_layout()
    plt.savefig(join(path_results, 'pid{}.png'.format(pid)), dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()




def plot_pids_by_synchronous_time_voltage(df, pdzdr, pid, path_results):
    """

    :param df:
    :param pdzdr:
    :param pid:
    :param path_results:
    :return:
    """
    # hard-coded
    px1, px2, py2 = 't_sync', 'SOURCE_TIME_MIDPOINT', 'VOLT'
    # inputs
    pdz, pdr = pdzdr
    # plot
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True,
                                   gridspec_kw={'height_ratios': [1.4, 1]})
    # plot z/dz by frames
    ax1.plot(df[px1], df[pdz], '-o', ms=1.5, lw=0.75, label=pdz)
    ax1.plot(df[px1], df['d0z'], '-o', ms=1.5, lw=0.75, label='d0z')
    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax1.legend(title=r'$p_{ID}=$' + str(pid), fontsize='small')
    ax1.grid(alpha=0.1)
    # plot V(t)
    ax1r = ax1.twinx()
    ax1r.plot(df[px1], df[py2], '-', color='gray', lw=0.75, alpha=0.5)
    ax1r.set_ylabel(r'$V_{app} \: (V)$', color='gray', alpha=0.5)
    # plot r/dr by frames
    ax2.plot(df[px1], df[pdr], '-o', ms=1.5, lw=0.75, label=pdr)
    ax2.plot(df[px1], df['d0rg'], '-o', ms=1.5, lw=0.75, label='d0rg')
    ax2.set_xlabel(r'$t \: (s)$')
    ax2.set_ylabel(r'$\Delta r \: (\mu m)$', fontsize='small')
    ax2.grid(alpha=0.1)
    ax2.legend()
    # -
    plt.tight_layout()
    plt.savefig(join(path_results, 'pid{}.png'.format(pid)),
                dpi=300, facecolor='w')
    plt.close()


def plot_pids_by_synchronous_time_voltage_monitor(df, pz, pid, test_settings, path_results):
    """

    :param df:
    :param pdzdr:
    :param pid:
    :param path_results:
    :return:
    """
    # hard-coded
    px, py1, py2 = 't_sync', 'VOLT', 'MONITOR_VALUE'
    # plot
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), nrows=2, sharex=True,
                                   gridspec_kw={'height_ratios': [1.4, 1]})
    # plot z/dz by frames
    ax1.plot(df[px], df[pz], 'k-', ms=1, lw=1, label=pz)
    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax1.legend(title=r'$p_{ID}=$' + str(pid), fontsize='small')
    ax1.grid(alpha=0.2)
    # plot V(t)
    ax1r = ax1.twinx()
    if test_settings['keithley_monitor'] == 'VOLT':
        ax1r.plot(df[px], df[py2], '-', color='tab:blue', ms=1, lw=0.75, label=r'$V_{app} \: (V)$')
        ax1r.set_ylabel('OUTPUT: {}Hz {} (V)'.format(test_settings['awg_freq'], test_settings['awg_wave']),
                        color='tab:blue')

        y1max = df[pz].abs().max()
        y2max = df[py2].abs().max()
        ax1.set_ylim([y1max * -1.05, y1max * 1.05])
        ax1r.set_ylim([y2max * -1.05, y2max * 1.05])
        ax1.legend(title=r'$p_{ID}=$' + str(pid), fontsize='small')
        ax1.grid(alpha=0.2)
    else:
        ax1r.plot(df[px], df[py1], '-', color='gray', lw=0.75, alpha=0.5)
        ax1r.set_ylabel(r'$V_{app} \: (V)$', color='gray', alpha=0.5)
    # plot monitor values
    ax2.plot(df[px], df[py2], '-o', ms=1.5, lw=0.75,
             label='OUTPUT: {}Hz {}'.format(test_settings['awg_freq'], test_settings['awg_wave']))
    ax2.set_xlabel(r'$t \: (s)$')
    ax2.set_xticks(np.arange(0, df[px].max() + 0.1, 1))
    ax2.grid(alpha=0.2)
    # -
    if test_settings['keithley_monitor'] == 'VOLT':
        ax2.plot(df[px], df[py1], '-', color='gray', lw=0.75, alpha=0.5, label='INPUT: AMPL.')
        ax2.plot(df[px], df[py1] * -1, '-', color='gray', lw=0.75, alpha=0.5)
        ax2.set_ylabel(r'$V_{app} \: (V)$', color='gray', alpha=0.5)
        ax2.legend()
    else:
        ax2.set_ylabel(f'MONITOR: {test_settings['keithley_monitor']} ({test_settings['keithley_measure_units']})',
                       fontsize='small')
        ax2r = ax2.twinx()
        ax2r.plot(df[px], df[py1], '-', color='gray', lw=0.75, alpha=0.5)
        ax2r.plot(df[px], df[py1] * -1, '-', color='gray', lw=0.75, alpha=0.5)
        ax2r.set_ylabel(r'$V_{app} \: (V)$', color='gray', alpha=0.5)
    ax2.legend()
    # -
    plt.tight_layout()
    plt.savefig(join(path_results, 'pid{}.png'.format(pid)),
                dpi=300, facecolor='w')
    plt.close()


def plot_pids_dz_by_voltage_ascending(df, pdzdr, dict_test, pid, path_results):
    """

    :param df:
    :param pdzdr:
    :param dict_test:
    :param pid:
    :param path_results:
    :return:
    """
    # hard-coded
    px1 = 'VOLT'
    # inputs
    pdz, pdr = pdzdr
    # pre-processing
    df_ascending = df[df['STEP'] <= dict_test['smu_step_max']]
    dfg = df_ascending.groupby(px1).mean().reset_index()

    # plot
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True,
                                   gridspec_kw={'height_ratios': [1.4, 1]})

    # plot z/dz by frames
    ax1.plot(df_ascending[px1], df_ascending[pdz], 'o', ms=1.5, label='Measured', zorder=3.2)
    ax1.plot(dfg[px1], dfg[pdz], '--', color='gray', lw=0.75, label='Mean', zorder=3.1)
    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax1.legend(title=r'$p_{ID}=$' + str(pid))
    ax1.grid(alpha=0.25)
    # plot r/dr by frames
    ax2.plot(df_ascending[px1], df_ascending[pdr], 'o', ms=1.5, label='Measured', zorder=3.2)
    ax2.plot(dfg[px1], dfg[pdr], '--', color='gray', lw=0.75, label='Mean', zorder=3.1)
    ax2.set_xlabel(r'$V_{app} \: (V)$')
    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(join(path_results, 'pid{}.png'.format(pid)),
                dpi=300, facecolor='w')
    plt.close()


def plot_pids_dz_by_voltage_hysteresis(df, pdz, dict_test, pid, path_results):
    """

    :param df:
    :param pdzdr:
    :param dict_test:
    :param pid:
    :param path_results:
    :return:
    """
    # hard-coded
    px1 = 'VOLT'
    # step_max depends on AC or DC
    if 'awg_mod_ampl_num_steps' in dict_test.keys():
        step_max = dict_test['awg_mod_ampl_num_steps'] - 1
    else:
        step_max = dict_test['smu_step_max']
    # pre-processing
    dfpid_ascending = df[df['STEP'] <= step_max]
    dfpid_descending = df[df['STEP'] > step_max]

    dfg_asc = dfpid_ascending.groupby(px1).mean().reset_index()
    dfg_desc = dfpid_descending.groupby(px1).mean().reset_index()

    fig, ax1 = plt.subplots()

    # plot z/dz by frames
    ax1.plot(dfpid_ascending[px1], dfpid_ascending[pdz], 'ro', ms=1.5, label='Ascending', zorder=3.3)
    ax1.plot(dfg_asc[px1], dfg_asc[pdz], '--', color='tab:red', lw=0.75, zorder=3.1)
    ax1.plot(dfpid_descending[px1], dfpid_descending[pdz], 'bo', ms=1.5, label='Descending', zorder=3.2)
    ax1.plot(dfg_desc[px1], dfg_desc[pdz], '--', color='tab:blue', lw=0.75, zorder=3.0)
    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax1.legend(title=r'$p_{ID}=$' + str(pid))
    ax1.grid(alpha=0.25)
    ax1.set_xlabel(r'$V_{app} \: (V)$')

    plt.tight_layout()
    plt.savefig(join(path_results, 'pid{}_{}.png'.format(pid, pdz)), dpi=300, facecolor='w')
    plt.close()


def plot_pids_dzdr_by_voltage_hysteresis(df, pdzdr, dict_test, pid, path_results):
    """

    :param df:
    :param pdzdr:
    :param dict_test:
    :param pid:
    :param path_results:
    :return:
    """
    # hard-coded
    px1 = 'VOLT'
    # inputs
    pdz, pdr = pdzdr
    # step_max depends on AC or DC
    if 'awg_mod_ampl_num_steps' in dict_test.keys():
        step_max = dict_test['awg_mod_ampl_num_steps'] - 1
    else:
        step_max = dict_test['smu_step_max']
    # pre-processing
    dfpid_ascending = df[df['STEP'] <= step_max]
    dfpid_descending = df[df['STEP'] > step_max]

    dfg_asc = dfpid_ascending.groupby(px1).mean().reset_index()
    dfg_desc = dfpid_descending.groupby(px1).mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True,
                                   gridspec_kw={'height_ratios': [1.4, 1]})

    # plot z/dz by frames
    ax1.plot(dfpid_ascending[px1], dfpid_ascending[pdz], 'ro', ms=1.5, label='Ascending', zorder=3.3)
    ax1.plot(dfg_asc[px1], dfg_asc[pdz], '--', color='tab:red', lw=0.75, zorder=3.1)
    ax1.plot(dfpid_descending[px1], dfpid_descending[pdz], 'bo', ms=1.5, label='Descending', zorder=3.2)
    ax1.plot(dfg_desc[px1], dfg_desc[pdz], '--', color='tab:blue', lw=0.75, zorder=3.0)
    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax1.legend(title=r'$p_{ID}=$' + str(pid))
    ax1.grid(alpha=0.25)
    # plot r/dr by frames
    ax2.plot(dfpid_ascending[px1], dfpid_ascending[pdr], 'ro', ms=1.5, label='Ascending', zorder=3.3)
    ax2.plot(dfg_asc[px1], dfg_asc[pdr], '--', color='tab:red', lw=0.75, zorder=3.1)
    ax2.plot(dfpid_descending[px1], dfpid_descending[pdr], 'bo', ms=1.5, label='Descending', zorder=3.2)
    ax2.plot(dfg_desc[px1], dfg_desc[pdr], '--', color='tab:blue', lw=0.75, zorder=3.0)
    ax2.set_xlabel(r'$V_{app} \: (V)$')
    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(join(path_results, 'pid{}_{}{}.png'.format(pid, pdz, pdr)),
                dpi=300, facecolor='w')
    plt.close()


def plot_dz_by_r_by_frame_with_surface_profile(df, przdr, dict_surf, frames, path_save, dict_fit, dr_ampl=1,
                                               show_interface=True, temporal_display_units='frame', title=None):
    """

    :param df:
    :param przdr: ('rg', 'd0z', 'drg')
    :param dict_surf: {'r': arr_r, 'z': arr_z, 'dr': arr_r = arr_r + dr, 'dz': arr_z = arr_z + dz}
    :param frames: np.arange(1, 100)
    :param path_save:
    :param dict_fit: None or {'s': smoothing param, 'faux_r_zero': mean(r < r_zero), 'faux_r_edge': z(r=r_edge) = 0}
    :param dr_ampl: amplification factor to visualize radial displacement.
    :return:
    """
    # initialize variables
    pr, pz, pdr = przdr
    pdr_ampl = 'r_dr'
    surf_r, surf_z = dict_surf['r'] + dict_surf['dr'], dict_surf['z'] + dict_surf['dz']
    # -
    # get 3D DPT limits
    rmin = 0
    rmax = np.max([df[pr].max(), surf_r[surf_z < dict_surf['dz'] - 0.5].max() + 100])
    zmin = np.min([df[pz].min(), surf_z.min()])
    zmax = np.max([df[pz].quantile(0.985), surf_z.max()])  # np.max([df[pz].max(), surf_z.max()])
    # -
    if dict_surf['subset'] == 'full':
        rmin = rmax * -1
    # -
    x_lim = rmin, rmax
    y_lim = zmin - 5, zmax + 5
    # ---
    # make "amplified radial displacement" column
    # and set color scale
    df[pdr_ampl] = df[pr] + df[pdr] * dr_ampl
    vmax_abs = df[pdr].abs().quantile(0.975)
    vmin, vmax = -vmax_abs, vmax_abs
    cmap = 'coolwarm'
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # ---
    for frame in frames:
        df_frame = df[df['frame'] == frame].sort_values(pr)
        x = df_frame[pdr_ampl].to_numpy()
        y = df_frame[pz].to_numpy()
        # -
        # ---
        # plot
        fig, ax = plt.subplots(figsize=(5.25, 2.75))
        # -
        # fit spline
        if dict_fit is not None:
            xf = df_frame[pr].to_numpy()
            xnew, ynew, xf, yf = wrapper_fit_radial_membrane_profile(
                x=xf,
                y=y,
                s=dict_fit['s'],
                faux_r_zero=dict_fit['faux_r_zero'],
                faux_r_edge=dict_fit['faux_r_edge'],
            )
            if dict_fit['show_faux_particles']:
                ax.scatter(xf, yf, s=20, color='k', marker='*', alpha=0.95)  # show faux particles used for fitting
            ax.plot(xnew, ynew, 'r-', lw=0.5, label='Fit', zorder=3.2)
        # -
        # ray tracing
        if frame > frames[1]:
            df_last = df[df['frame'] == frame - 1]
            df_two = pd.concat([df_last, df_frame])
            df_two = df_two.sort_values('frame')
            for pid in df_two['id'].unique():
                dfpid = df_two[df_two['id'] == pid]
                ax.plot(dfpid[pdr_ampl], dfpid[pz], '-', color='gray', lw=0.25, alpha=0.25, zorder=3.1)
            if frame > frames[2]:
                df_last2 = df[df['frame'] == frame - 2]
                df_three = pd.concat([df_two, df_last2])
                df_three = df_three.sort_values('frame')
                for pid in df_three['id'].unique():
                    dfpid = df_three[df_three['id'] == pid]
                    ax.plot(dfpid[pdr_ampl], dfpid[pz], '-', color='gray', lw=0.25, alpha=0.25, zorder=3.1)
        # -
        # plot surface
        ax.plot(surf_r, surf_z, '-', color='gray', lw=0.5, label='SP', zorder=3.1)
        # plot particles w/ color bar
        ax.scatter(x, y, c=df_frame[pdr], s=5, cmap=cmap, vmin=vmin, vmax=vmax, label='FPs', zorder=3.3)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label=r'$\Delta r \: (\mu m)$', pad=0.025, )
        # -
        if show_interface:
            # --- calculate position of zipping interface
            zipping_interface_r, zipping_interface_z = get_zipping_interface_rz(
                r=df_frame[pr].to_numpy(),
                z=df_frame[pz].to_numpy(),
                surf_r=surf_r,
                surf_z=surf_z,
            )
            ax.axhline(zipping_interface_z, 0, 1, ls='--', color='gray', lw=0.5, alpha=0.25, zorder=3.0)
            ax.axvline(zipping_interface_r, 0, 1, ls='--', color='gray', lw=0.5, alpha=0.25, zorder=3.0)
        # -
        if temporal_display_units == 't_sync':
            time_display = '{:.1f} ms'.format(df_frame['t_sync'].mean() * 1e3)
        else:
            time_display = 'fr: {:03d}'.format(frame)
        ax.text(0.0125, 1.0125, time_display, color='black', fontsize=8,
                horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        # ax.set_title('frame: {:03d}'.format(frame))
        # -
        if title is not None:
            ax.set_title(title, fontsize='small')
        # -
        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_xlim(x_lim)
        ax.set_ylabel(r'$\Delta z \: (\mu m)$')
        ax.set_ylim(y_lim)
        ax.grid(alpha=0.0625)

        plt.tight_layout()
        plt.savefig(join(path_save, 'dz-r_by_fr{:03d}_+dr{}X.png'.format(frame, dr_ampl)),
                    dpi=300, facecolor='w')
        plt.close()
    # ---


def plot_dz_by_r_by_frois_with_surface_profile(df, przdr, dict_surf, frames, path_save, dict_fit):
    """

    :param df:
    :param przdr: ('rg', 'd0z', 'drg')
    :param dict_surf: {'r': arr_r, 'z': arr_z, 'dr': arr_r = arr_r + dr, 'dz': arr_z = arr_z + dz}
    :param frames: np.arange(1, 100)
    :param path_save:
    :param dict_fit: None or {'s': smoothing param, 'faux_r_zero': mean(r < r_zero), 'faux_r_edge': z(r=r_edge) = 0}
    :return:
    """
    # initialize variables
    pr, pz, pdr = przdr
    surf_r = dict_surf['r'] + dict_surf['dr']
    surf_z = (dict_surf['z'] + dict_surf['dz']) * dict_surf['scale_z']
    # -
    # get 3D DPT limits
    rmin = 0
    rmax = np.max([df[pr].max(), surf_r.max()])
    zmin = np.min([df[pz].min(), surf_z.min()])
    zmax = np.max([df[pz].max(), surf_z.max()])
    # -
    # spline smoothing parameter
    if dict_fit is not None:
        spline_smooths = dict_fit['s']
        if isinstance(spline_smooths, (int, float)):
            spline_smooths = [spline_smooths] * len(frames)
        # constrain x_lim_max using radius of feature instead of range of surface profile
        rmax = np.max([df[pr].max(), dict_fit['faux_r_edge'] * 1.075])
    else:
        spline_smooths = [0] * len(frames)
    # -
    x_lim = rmin, rmax
    y_lim = zmin - 5, zmax + 5
    # ---
    # set color scale
    vmin, vmax = df[pz].min(), df[pz].max()
    cmap = 'coolwarm'
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    clrs = [mpl.cm.coolwarm(norm(df[df['frame'] == frame][pz].min())) for frame in frames]
    # ---
    # plot
    fig, ax = plt.subplots(figsize=(5.25, 2.75))

    # plot surface
    ax.plot(surf_r, surf_z, '-', color='gray', lw=0.5, zorder=3.1)  # , label='SP'
    ax.fill_between(surf_r, y_lim[0], surf_z, fc='gray', alpha=0.125, ec='gray')

    for frame, clr, s in zip(frames, clrs, spline_smooths):
        df_frame = df[df['frame'] == frame].sort_values(pr)
        x = df_frame[pr].to_numpy()  # radial position
        y = df_frame[pz].to_numpy()  # axial position or axial displacement
        # -
        # get voltage corresponding to this frame
        if frame == frames[0]:
            voltage_frame = 0
        else:
            voltage_frame = df_frame['VOLT'].values.tolist()[0]
        # ---
        # -
        # fit spline
        if dict_fit is not None:
            xf = df_frame[pr].to_numpy()
            xnew, ynew, xf, yf = wrapper_fit_radial_membrane_profile(
                x=xf,
                y=y,
                s=s,
                faux_r_zero=dict_fit['faux_r_zero'],
                faux_r_edge=dict_fit['faux_r_edge'] + dict_surf['dr'],
            )
            if dict_fit['show_faux_particles']:
                ax.scatter(xf, yf, s=20, color='k', marker='*', alpha=0.95)  # show faux particles used for fitting
            ax.plot(xnew, ynew, '--', color=clr, lw=0.5, zorder=3.2)  # , label='Fit'
        # -
        # plot particles w/ color bar
        ax.scatter(x, y, s=5, color=clr, zorder=3.3, label=int(np.round(voltage_frame)))  # , label='FPs'
        # plot particle ID's for reference
        plot_particle_ids = False
        if plot_particle_ids:
            id_ = df_frame['id'].to_numpy()
            for xi, yi, pid in zip(x, y, id_):
                ax.text(xi, yi, str(pid), color='black', fontsize=8)

    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$\Delta z \: (\mu m)$', pad=0.025, )
    # -
    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_xlim(x_lim)
    ax.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax.set_ylim(y_lim)
    ax.grid(alpha=0.0625)
    # ax.set_title('frames: {}'.format(frames))
    ax.legend(title=r'$V_{app} \: (V)$', loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig(join(path_save, 'dz-r_by_frois{}_legend-voltage.png'.format(frames)),
                dpi=300, facecolor='w')
    plt.close()
    # ---


def plot_dz_by_r_by_frois_normalize_membrane_profile(df, przdr, dict_surf, frames, path_save):
    """

    :param df:
    :param przdr: ('rg', 'd0z', 'drg')
    :param dict_surf: {'r': arr_r, 'z': arr_z, 'dr': arr_r = arr_r + dr, 'dz': arr_z = arr_z + dz}
    :param frames: np.arange(1, 100)
    :param path_save:
    :param dict_fit: None or {'s': smoothing param, 'faux_r_zero': mean(r < r_zero), 'faux_r_edge': z(r=r_edge) = 0}
    :return:
    """
    # initialize variables
    pr, pz, pdr = przdr
    pz_norm = pz + '_norm'
    # get surface profile
    surf_r, surf_z = dict_surf['r'] + dict_surf['dr'], dict_surf['z'] + dict_surf['dz']
    # -
    # normalize pz
    df_frames, dz_frames = [], []
    for frame in frames:
        df_frame = df[df['frame'] == frame].sort_values(pr)

        dz_frame = df_frame[pz].mean()
        dz_frames.append(dz_frame)

        df_frame[pz_norm] = df_frame[pz] - dz_frame
        df_frames.append(df_frame)

    df = pd.concat(df_frames)
    # get 3D DPT limits
    y_lim = [df[pz_norm].abs().max() * -1.5, df[pz_norm].abs().max() * 1.5]
    # ---
    # set color scale
    vmin, vmax = np.min(dz_frames), np.max(dz_frames)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # clrs = [mpl.cm.coolwarm(norm(df[df['frame'] == frame][pz].min())) for frame in frames]
    clrs = [mpl.cm.coolwarm(norm(x)) for x in dz_frames]
    # ---
    # plot
    fig, ax = plt.subplots(figsize=(5.5, 2.75))

    for frame, clr, dz_frame in zip(frames, clrs, dz_frames):
        df_frame = df[df['frame'] == frame].sort_values(pr)
        # -
        # get voltage corresponding to this frame
        voltage_frame = df_frame['VOLT'].values.tolist()[0]
        # -
        # get position of zipping interface
        zipping_interface_r, zipping_interface_z = get_zipping_interface_rz(
            r=df_frame[pr].to_numpy(),
            z=df_frame[pz].to_numpy(),
            surf_r=surf_r,
            surf_z=surf_z,
        )
        # -
        # get data
        x = df_frame[df_frame[pr] < zipping_interface_r][pr].to_numpy()  # radial position
        y = df_frame[df_frame[pr] < zipping_interface_r][pz_norm].to_numpy()  #  - df_frame[pz].mean()  # axial position or axial displacement

        # -
        # plot particles w/ color bar
        ax.scatter(x, y, s=5, color=clr, zorder=3.3,
                   label='{}V: {} um'.format(int(np.round(voltage_frame)), np.round(dz_frame, 1)))  # , label='FPs'
        # plot particle ID's for reference
        plot_particle_ids = False
        if plot_particle_ids:
            id_ = df_frame['id'].to_numpy()
            for xi, yi, pid in zip(x, y, id_):
                ax.text(xi, yi, str(pid), color='black', fontsize=8)
        # -
        # fit parabola
        def fit_parabola(x, a, b, c):
            return a * x ** 2 + b * x + c

        if len(x) > 3:
            popt, pcov = curve_fit(fit_parabola, x, y)
            ax.plot(x, fit_parabola(x, *popt), '-', color=clr)

    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$\Delta z \: (\mu m)$', pad=0.025, )
    # -
    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel(r'$\Delta z_{norm} \: (\mu m)$')
    ax.set_ylim(y_lim)
    ax.grid(alpha=0.0625)
    # ax.set_title('frames: {}'.format(frames))
    ax.legend(title=r'$V_{app}: \Delta z$', loc='upper left', fontsize='xx-small', ncols=len(frames))
    plt.tight_layout()
    plt.savefig(join(path_save, 'dz-norm-r_by_frois{}_legend-voltage.png'.format(frames)),
                dpi=300, facecolor='w')
    plt.close()
    # ---


# --------------------------------------------------------
# 2D PLOTS
# --------------------------------------------------------

def plot_scatter_xy(df, pxy, pcolor, savepath=None, field=None,
                         units=None, title=None, overlay_circles=False, dict_settings=None):
    # get data
    x, y, c = df[pxy[0]].to_numpy(), df[pxy[1]].to_numpy(), df[pcolor].to_numpy()

    # plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=c, s=1)

    # --- format figure
    # if no field is passed, use x-y limits.
    if field is None:
        field = (np.min([x.min(), y.min()]), np.max([x.max(), y.max()]))
    # if no units, don't assume any
    if units is None:
        units = ('', '', '')
    # overlay circles to show diameter of features
    if overlay_circles:
        # make circular patches for inner and outer radii
        circle_edge = Circle(dict_settings['xyc_microns'], dict_settings['radius_microns'])
        patches = [circle_edge]
        if 'radius_hole_microns' in dict_settings.keys():
            circle_hole = Circle(dict_settings['xyc_microns'], dict_settings['radius_hole_microns'])
            patches.append(circle_hole)
        pc = PatchCollection(patches, fc='none', ec='k', lw=0.5, ls='--', alpha=0.5)
        ax.add_collection(pc)
    ax.set(xlim=(field[0], field[1]), xticks=(field[0], field[1]),
           ylim=(field[0], field[1]), yticks=(field[0], field[1]),
           )
    ax.invert_yaxis()
    ax.set_xlabel(r'$x$ ' + units[0])
    ax.set_ylabel(r'$y$ ' + units[1])
    if title is not None:
        ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    if savepath is not None:
            plt.savefig(savepath, dpi=300, facecolor='white', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_quiver_xy_dxdy(df, pxydxdy, pcolor_dxdy, frames, scale=5, savepath=None, field=None,
                         units=None, overlay_circles=False, dict_settings=None,
                        show_interface=False, dict_surf=None, prz=None,
                        temporal_display_units='frame', title=None):
    # --- setup
    # colorscale of quiver plot (dx, dy)
    max_dxdy = df[pcolor_dxdy].abs().quantile(0.96)  # .max()
    cmap_quiver = 'coolwarm'
    norm_quiver = mpl.colors.Normalize(vmin=-max_dxdy, vmax=max_dxdy)

    for fr in frames:
        df_fr = df[df['frame'] == fr]
        # get data
        x, y = df_fr[pxydxdy[0]].to_numpy(), df_fr[pxydxdy[1]].to_numpy()
        dx, dy = df_fr[pxydxdy[2]].to_numpy(), df_fr[pxydxdy[3]].to_numpy()
        c_quiver = df_fr[pcolor_dxdy].to_numpy()

        # plot
        fig, ax = plt.subplots()

        q = ax.quiver(x, y, dx, dy, c_quiver, cmap=cmap_quiver, norm=norm_quiver,
                  angles='xy',  # 'xy': arrow points from (x, y) to (x+u, y+v)
                  pivot='tail',  # the arrow is anchored around its tail
                  scale_units='xy',  # arrow length uses same units as (x, y)
                  scale=1/scale,  # scale arrow length relative to data values (e.g., 0.2: arrow is 5X data units)
                  units='width',  # arrow units depend on width of axes
                  width=0.0035,  # shaft width in arrow units (typical starting value is 0.005 * width of axes)
                  headwidth=3,
                  headlength=2.5,
                  headaxislength=2.5,
                  minshaft=1,
                  minlength=2,  # minimum length as multiple of shaft width, below this: plot a dot
        )
        ax.quiverkey(q, X=0.85, Y=1.025, U=scale, label=r'$1 \: (\mu m)$', labelpos='E', labelsep=0.05)
        ax.scatter(x, y, s=0.5, color='k')  # c=c_scatter, s=1, cmap=cmap_scatter, norm=norm_scatter, )

        # --- format figure
        # if no field is passed, use x-y limits.
        if field is None:
            field = (np.min([x.min(), y.min()]), np.max([x.max(), y.max()]))
        # if no units, don't assume any
        if units is None:
            units = ('', '', '')
        # overlay zipping interface
        if show_interface:
            # --- calculate position of zipping interface
            zipping_interface_r, zipping_interface_z = get_zipping_interface_rz(
                r=df_fr[prz[0]].to_numpy(),
                z=df_fr[prz[1]].to_numpy(),
                surf_r=dict_surf['r'],
                surf_z=dict_surf['z'],
            )
            # make circular patch
            circle_interface = Circle(dict_settings['xyc_microns'], zipping_interface_r)
            patches1 = [circle_interface]
            pc1 = PatchCollection(patches1, fc='none', ec='gray', lw=0.65, ls='--', alpha=0.5)
            ax.add_collection(pc1)
        # overlay circles to show diameter of features
        if overlay_circles:
            # make circular patches for inner and outer radii
            circle_edge = Circle(dict_settings['xyc_microns'], dict_settings['radius_microns'])
            patches = [circle_edge]
            if 'radius_hole_microns' in dict_settings.keys():
                circle_hole = Circle(dict_settings['xyc_microns'], dict_settings['radius_hole_microns'])
                patches.append(circle_hole)
            pc = PatchCollection(patches, fc='none', ec='k', lw=0.5, ls='-', alpha=0.65)
            ax.add_collection(pc)
        ax.set(xlim=(field[0], field[1]), xticks=(field[0], field[1]),
               ylim=(field[0], field[1]), yticks=(field[0], field[1]),
               )
        # -
        if temporal_display_units == 't_sync':
            time_display = '{:.1f} ms'.format(df_fr['t_sync'].mean() * 1e3)
        else:
            time_display = 'fr: {:03d}'.format(fr)
        ax.text(7.5, -7.5, time_display, color='black', fontsize=8,
                horizontalalignment='left', verticalalignment='bottom') # , transform=ax.transAxes
        # ax.set_title('frame: {:03d}'.format(frame)) # ax.set_title(f'Frame: {fr}')
        # -
        if title is not None:
            ax.set_title(title, fontsize='x-small')
        # -
        ax.invert_yaxis()
        ax.set_xlabel(r'$x$ ' + units[0])
        ax.set_ylabel(r'$y$ ' + units[1])
        ax.set_aspect('equal')
        plt.tight_layout()
        if savepath is not None:
                plt.savefig(savepath.format(fr), dpi=300, facecolor='white', bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def plot_2D_heatmap(df, pxyz, savepath=None, field=None, interpolate='linear',
                    levels=15, units=None, title=None, overlay_circles=False, dict_settings=None):
    """

    :param df:
    :param pxyz:
    :param savepath:
    :param field:
    :param interpolate:
    :param levels:
    :param units:
    :param title:
    :param overlay_circles:
    :param dict_settings:
    :param vminmax:
    :return:
    """

    # get data
    x, y, z = df[pxyz[0]].to_numpy(), df[pxyz[1]].to_numpy(), df[pxyz[2]].to_numpy()

    # if no field is passed, use x-y limits.
    if field is None:
        field = (np.min([x.min(), y.min()]), np.max([x.max(), y.max()]))
    # if no units, don't assume any
    if units is None:
        units = ('', '', '')

    # Create grid values.
    xi = np.linspace(field[0], field[1], len(df))
    yi = np.linspace(field[0], field[1], len(df))

    # interpolate
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method=interpolate)

    # plot
    fig, ax = plt.subplots()

    # contour
    # ax.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
    cntr1 = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax, label=units[2])

    # scatter
    ax.plot(x, y, 'ko', ms=3)

    # overlay circles to show diameter of features
    if overlay_circles:
        # make circular patches for inner and outer radii
        circle_edge = Circle(dict_settings['xyc_microns'], dict_settings['radius_microns'])
        patches = [circle_edge]
        if 'radius_hole_microns' in dict_settings.keys():
            circle_hole = Circle(dict_settings['xyc_microns'], dict_settings['radius_hole_microns'])
            patches.append(circle_hole)
        pc = PatchCollection(patches, fc='none', ec='k', lw=0.5, ls='--', alpha=0.5)
        ax.add_collection(pc)

    ax.set(xlim=(field[0], field[1]), xticks=(field[0], field[1]),
           ylim=(field[0], field[1]), yticks=(field[0], field[1]),
           )
    ax.invert_yaxis()
    ax.set_xlabel(r'$x$ ' + units[0])
    ax.set_ylabel(r'$y$ ' + units[1])

    if title is not None:
        ax.set_title(title)

    ax.set_aspect('equal')
    plt.tight_layout()
    if savepath is not None:
            plt.savefig(savepath, dpi=300, facecolor='white', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# BELOW: PLOTS FOR TESTX
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------
# BELOW: PLOTS FOR TESTX: MERGED COORDS VOLTS
# ----------------------------------

def plot_merged_coords_volt_per_pid_by_volt_freq(df, path_save, threshold_pids_by_d0z=0.0, only_pids=None,
                                                 only_test_types=None, arr_model_VdZ=None):
    pz, pc = 'd0z', 'awg_freq'
    dx, dy = 'VOLT', pz
    # filter test types
    if only_test_types is None:
        only_test_types = ['STD1', 'STD2', 'STD3', 'VAR3', '1', '2', '3']
    df = df[df['test_type'].isin(only_test_types)]

    # setup frequency color bar
    cmap_name = 'plasma'
    cmap = mpl.colormaps['plasma']
    vmin, vmax = df[pc].min(), df[pc].max()
    # if vmin < 0.25: vmin = 0.25
    # if vmax > 10e3: vmax = 10e3
    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # filter z-displacement
    if only_pids is None:
        only_pids = df[df[pz] < threshold_pids_by_d0z]['id'].unique()
    df = df[df['id'].isin(only_pids)]

    for pid in only_pids:
        df_pid = df[df['id'] == pid].reset_index(drop=True)
        tids = df_pid['tid'].unique()

        # initialize figure
        fig, ax = plt.subplots()
        # plot model
        if arr_model_VdZ is not None:
            ax.plot(arr_model_VdZ[0], arr_model_VdZ[1], 'k-', label='Model')

        for tid in tids:
            df_pid_tid = df_pid[df_pid['tid'] == tid].reset_index(drop=True)
            df_pid_tid = df_pid_tid[df_pid_tid['STEP'] <= df_pid_tid['STEP'].iloc[df_pid_tid['VOLT'].abs().idxmax()]]
            ax.plot(df_pid_tid[dx].abs(), df_pid_tid[dy], 'o', ms=1.5, label=tid,
                    color=cmap(norm(df_pid_tid[pc].iloc[0])))

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$f \: (Hz)$', pad=0.025, )
        ax.set_xlabel(r'$V \: (V)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.grid(alpha=0.25)
        # ax.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(join(path_save, f'compare_all_pid{pid}_data_to_model.png'),
                    dpi=300, facecolor='w', bbox_inches='tight')
        plt.close(fig)


def plot_parametric_sweeps_per_pid_trajectory_by_tid(df_mcviv, combinations, only_pids, threshold_pids_by_d0z, path_save):
    px, py1, py2 = 't_sync', 'd0z', 'd0rg'

    if only_pids is None:
        only_pids = df_mcviv[df_mcviv[py1] < threshold_pids_by_d0z]['id'].unique()

    df_mcviv = df_mcviv[df_mcviv['id'].isin(only_pids)]

    for result in combinations:
        # parse group definition into a nice standard name
        # and make directory to save figures into
        gd = result['group_definition']
        unique_id = ''
        if 'test_type' in gd.keys():
            unique_id += str(gd['test_type']) + '_'
        if 'output_volt' in gd.keys():
            unique_id += str(int(gd['output_volt'])) + 'V_'
        if 'awg_freq' in gd.keys():
            if gd['awg_freq'] >= 1000:
                unique_id += str(gd['awg_freq'] / 1000) + 'kHz_'
            elif gd['awg_freq'] >= 10:
                unique_id += str(int(gd['awg_freq'])) + 'Hz_'
            else:
                unique_id += str(gd['awg_freq']) + 'Hz_'
        if 'awg_wave' in gd.keys():
            unique_id += str(gd['awg_wave'])
        unique_id = unique_id.rstrip('_')
        path_save_group = join(path_save, unique_id)
        if not os.path.exists(path_save_group):
            os.makedirs(path_save_group)

        # get tids in this group
        tids = result['group_tids']
        df = df_mcviv[df_mcviv['tid'].isin(tids)]

        # get independent variable
        col_label = result['remaining_column']
        ncols = int(np.ceil(len(tids) / 16))

        # iterate and plot
        for pid in only_pids:
            df_pid = df[df['id'] == pid]
            tids.sort()

            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 7), gridspec_kw={'height_ratios': [1, 0.5]})

            for tid in tids:
                df_pid_tid = df_pid[df_pid['tid'] == tid]
                try:
                    label = df_pid_tid[col_label].iloc[0]
                except IndexError:
                    print("IndexError: {} - tid {} - pid {}".format(unique_id, tid, pid))
                    label = ''

                ax1.plot(df_pid_tid[px], df_pid_tid[py1], '-', lw=0.5, label=f'{tid}: {label}')
                ax2.plot(df_pid_tid[px], df_pid_tid[py2], '-', lw=0.5, label=f'{tid}: {label}')

            ax1.set_ylabel(r'$\Delta_{o} z \: (\mu m)$')
            ax1.grid(alpha=0.2)
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title=col_label, fontsize='x-small', ncol=ncols)

            ax2.set_ylabel(r'$\Delta_{o} r \: (\mu m)$')
            ax2.grid(alpha=0.2)
            ax2.set_xlabel(r'$t \: (s)$')

            plt.suptitle(unique_id + ': ' + r'$p_{ID}=$' + '{}'.format(pid))
            plt.tight_layout()
            plt.savefig(join(path_save_group, 'pid{}_{}_vs_{}.png'.format(pid, col_label, unique_id)),
                        dpi=300, facecolor='w', bbox_inches='tight')
            plt.close(fig)


def plot_heatmap_merged_coords_volt_all_pids_by_volt_freq(df, path_save, threshold_pids_by_d0z=0.0,
                                                          only_test_types=None, save_id=None):

    pv, pf, pz = 'VOLT', 'awg_freq', 'd0z'
    # Check that required columns exist in the dataframe
    required_columns = {pv, pf, pz}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The dataframe must contain columns: {required_columns}")

    if only_test_types is None:
        only_test_types = ['STD1', 'STD2', 'STD3', 'VAR3', '1', '2', '3']
    df = df[df['test_type'].isin(only_test_types)]

    if save_id is None:
        save_id = ''

    dz_threshold_pids = df[df[pz] < threshold_pids_by_d0z]['id'].unique()
    df = df[df['id'].isin(dz_threshold_pids)]

    df = df.round({'VOLT': -1})

    # Grouping data by 'output_volt' and 'awg_freq', and averaging 'dz_mean'
    grouped_data = df.groupby([pf, pv])[pz].mean().reset_index()

    # Pivot the grouped data to create a 2D grid for heatmap plotting
    heatmap_data = grouped_data.pivot(index=pf, columns=pv, values=pz)

    # Ensure sorted order for proper heatmap representation
    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)

    # Plot the heatmap using matplotlib
    cmap = 'plasma'
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(label='D0Z (um)')
    plt.xlabel('Volt (V)')
    plt.ylabel('Frequency (Hz)')
    plt.title('2D Heatmap: Frequency vs Volt (D0Z, DZ_thresh={})'.format(threshold_pids_by_d0z))
    plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index)
    plt.tight_layout()
    plt.savefig(join(path_save, f'{save_id}_heatmap_merged_coords_volt_by_volt-freq_all_pids.png'),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()


# ----------------------------------
# BELOW: PLOTS FOR TESTX: NET DISPLACEMENT
# ----------------------------------


def plot_net_d0zr_per_pid_by_volt_freq(df, path_save, only_test_types=None, arr_model_VdZ=None):
    px, py1, py2, pc = 'output_volt', 'dz_mean', 'dr_mean', 'awg_freq'
    cmap = 'coolwarm'

    if only_test_types is None:
        only_test_types = ['STD1', 'STD2', 'STD3', 'VAR3', '1', '2', '3']
    df = df[df['test_type'].isin(only_test_types)]

    pids = df['id'].unique()

    for pid in pids:
        df_pid = df[df['id'] == pid].sort_values('awg_freq', ascending=True)
        x, y, c = df_pid[px].abs(), df_pid[py1], df_pid[pc]
        vmin, vmax = c.min(), c.max()
        if vmin < 0.25: vmin = 0.25
        if vmax > 10e3: vmax = 10e3
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # plot dz by voltage w/ frequency color bar
        fig, ax = plt.subplots()
        if arr_model_VdZ is not None:
            ax.plot(arr_model_VdZ[0], arr_model_VdZ[1], '-', color='black', lw=0.5, label='model')
        ax.scatter(x, y, c=c, s=5, cmap=cmap)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$f \: (Hz)$', pad=0.025, )
        ax.set_xlabel(r'$V \: (V)$')
        ax.set_ylabel(r'$\Delta_{o} z \: (\mu m)$')
        ax.set_title(r'$p_{ID}$' + ': {}'.format(pid))
        plt.tight_layout()
        plt.savefig(join(path_save, 'net-d0zr_by_volt-freq_pid{}.png'.format(pid)),
                    dpi=300, facecolor='w', bbox_inches='tight')
        plt.close(fig)

def plot_net_d0zr_all_pids_by_volt_freq(df, path_save, threshold_pids_by_d0z=0.0, only_test_types=None,
                                        arr_model_VdZ=None):
    px, py1, py2, pc = 'output_volt', 'dz_mean', 'dr_mean', 'awg_freq'
    cmap = 'plasma'

    if only_test_types is None:
        only_test_types = ['STD1', 'STD2', 'STD3', 'VAR3', '1', '2', '3', 1, 2, 3]
    df = df[df['test_type'].isin(only_test_types)]

    dz_threshold_pids = df[df['dz_mean'] < threshold_pids_by_d0z]['id'].unique()
    df = df[df['id'].isin(dz_threshold_pids)]

    df = df.sort_values('awg_freq', ascending=True)

    # plot dz by voltage w/ frequency color bar
    x, y, c = df[px].abs(), df[py1], df[pc]
    vmin, vmax = c.min(), c.max()
    #if vmin < 0.25: vmin = 0.25
    #if vmax > 10e3: vmax = 10e3
    #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()
    if arr_model_VdZ is not None:
        ax.plot(arr_model_VdZ[0], arr_model_VdZ[1], '-', color='black', lw=0.5, label='model')
    ax.scatter(x, y, c=c, s=5, cmap=cmap)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$f \: (Hz)$', pad=0.025, )
    ax.set_xlabel(r'$V \: (V)$')
    ax.set_ylabel(r'$\Delta_{o} z \: (\mu m)$')
    ax.set_title(r'$\Delta_{o} z_{threshold} =$' + str(threshold_pids_by_d0z) + r'$\mu m$' + ' (n={})'.format(len(dz_threshold_pids)))
    plt.tight_layout()
    plt.savefig(join(path_save, 'net-d0zr_by_volt-freq_all_pids.png'),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.close(fig)


def plot_net_d0zr_all_pids_by_volt_freq_errorbars_per_tid(df, path_save, threshold_pids_by_d0z=0.0,
                                                          only_test_types=None, shift_V_by_X_log_freq=0,
                                                          arr_model_VdZ=None):
    px, py1, py2, pc = 'output_volt', 'dz_mean', 'dr_mean', 'awg_freq'
    cmap_name = 'plasma'
    cmap = mpl.colormaps['plasma']

    if only_test_types is None:
        only_test_types = ['STD1', 'STD2', 'STD3', 'VAR3', '1', '2', '3']
    df = df[df['test_type'].isin(only_test_types)]

    dz_threshold_pids = df[df['dz_mean'] < threshold_pids_by_d0z]['id'].unique()
    df = df[df['id'].isin(dz_threshold_pids)].sort_values('awg_freq', ascending=True)

    dfm = df[['tid', px, py1, py2, pc]].groupby('tid').mean().reset_index()
    dfstd = df[['tid', px, py1, py2, pc]].groupby('tid').std().reset_index()

    # plot dz by voltage w/ frequency color bar
    x, y, c = df[px].abs(), df[py1], df[pc]
    vmin, vmax = c.min(), c.max()
    #if vmin < 0.25: vmin = 0.25
    #if vmax > 10e3: vmax = 10e3
    #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # figure
    fig, ax = plt.subplots()

    if arr_model_VdZ is not None:
        ax.plot(arr_model_VdZ[0], arr_model_VdZ[1], '-', color='black', lw=0.5, label='model')

    for tid in dfm['tid'].unique():
        dfm_tid = dfm[dfm['tid'] == tid]
        dfstd_tid = dfstd[dfstd['tid'] == tid]
        dB = np.log10(dfm_tid[pc]) * shift_V_by_X_log_freq
        ax.errorbar(dfm_tid[px].abs() + dB, dfm_tid[py1], yerr=dfstd_tid[py1],
                    color=cmap(norm(dfm_tid[pc])), fmt='o', capsize=2, elinewidth=1, label=tid)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$f \: (Hz)$', pad=0.025, )
    ax.set_xlabel(r'$V \pm $' + '{} dB '.format(shift_V_by_X_log_freq) + r'$ (V)$')
    ax.set_ylabel(r'$\Delta_{o} z \: (\mu m)$')
    ax.set_title(r'$\Delta_{o} z_{threshold} =$' + str(threshold_pids_by_d0z) + r'$\mu m$' + ' (n={})'.format(len(dz_threshold_pids)))
    plt.tight_layout()
    plt.savefig(join(path_save, 'errorbars_net-d0zr_by_volt-freq_all_pids__pm{}dB.png'.format(shift_V_by_X_log_freq)),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.close(fig)

def plot_heatmap_net_d0zr_all_pids_by_volt_freq(df, path_save, threshold_pids_by_d0z=0.0, only_test_types=None):

    # Check that required columns exist in the dataframe
    required_columns = {'output_volt', 'awg_freq', 'dz_mean'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The dataframe must contain columns: {required_columns}")

    if only_test_types is None:
        only_test_types = ['STD1', 'STD2', 'STD3', 'VAR3', '1', '2', '3']
    df = df[df['test_type'].isin(only_test_types)]

    dz_threshold_pids = df[df['dz_mean'] < threshold_pids_by_d0z]['id'].unique()
    df = df[df['id'].isin(dz_threshold_pids)]

    # Grouping data by 'output_volt' and 'awg_freq', and averaging 'dz_mean'
    grouped_data = df.groupby(['awg_freq', 'output_volt'])['dz_mean'].mean().reset_index()

    # Pivot the grouped data to create a 2D grid for heatmap plotting
    heatmap_data = grouped_data.pivot(index='awg_freq', columns='output_volt', values='dz_mean')

    # Ensure sorted order for proper heatmap representation
    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)

    # Plot the heatmap using matplotlib
    cmap = 'plasma'
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(label='Mean DZ (dz_mean)')
    plt.xlabel('Output Volt')
    plt.ylabel('AWG Frequency')
    plt.title('2D Heatmap: AWG Frequency vs Output Volt (Mean DZ, DZ_thresh={})'.format(threshold_pids_by_d0z))
    plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index)
    plt.tight_layout()
    plt.savefig(join(path_save, 'heatmap_net-d0zr_by_volt-freq_all_pids.png'),
                dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()




# ----------------------------------------------------------------------------------------------------------------------
# BELOW: PLOT ANALYTICAL MODELING
# ----------------------------------------------------------------------------------------------------------------------

def plot_sweep_z_by_v(df_roots, path_save, save_id):
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for j in range(len(df_roots)):
        ax.plot(df_roots[j].U, df_roots[j].z * 1e6, mrkrs[j], label=vs[j])

    ax.set_xlabel('V (V)', fontsize=14)
    ax.set_ylabel('z (um)', fontsize=14)
    ax.legend(title=k)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_sweep_z-by-v.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()


def plot_profile_fullwidth(full, use, shape_function, width, depth, num_segments, shape_x0, dia_flat,
                           ax=None, ylbl=True, ls=None):
    """ ax = plot_profile_fullwidth(full, use, shape_function, width, depth, num_segments, shape_x0, dia_flat, ax) """
    # px, py = get_erf_profile(X, Z, num_segments, shape_x0, dia_flat)
    px, py = shape_function(width, depth, num_segments, shape_x0, dia_flat)

    # plot

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if full is True:
        if use == 'erf':
            px_revolved, py_revolved = complete_erf_profile(px, py, width, dia_flat)
        elif use == 'flat':
            raise ValueError("Not implemented yet.")
            px_revolved, py_revolved = complete_flat_profile(px, py, width, theta=shape_x0)
        else:
            raise ValueError('must be: [erf, flat]')

        if ls is not None:
            ax.plot(px * 1e3, py * 1e6, ls)
            ax.plot(px_revolved * 1e3, py_revolved * 1e6, ls)
        else:
            ax.plot(px * 1e3, py * 1e6, ls='-', color='tab:blue')
            ax.plot(px_revolved * 1e3, py_revolved * 1e6, ls='-', color='tab:blue', label='Cross-section')
    else:
        ax.plot(px * 1e3, py * 1e6, ls='-', color='tab:blue', label='Zipping Path')
        ax.set_title('Dia.={}, Dia. Flat={}'.format(X, dia_flat))

    if ylbl:
        ax.set_ylabel('z (um)')

    ax.grid(alpha=0.25)
    ax.set_xlabel('r (mm)')

    if ls is None:
        ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        return ax


def plot_sweep_z_by_v_and_rz_profile():
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8.5, 3))

    for j in range(len(df_roots)):
        vval = vs[j]
        ls = mrkrs[j]

        ax2 = plot_profile_fullwidth(full=True,
                                     use='erf',
                                     shape_function=shape_function,
                                     width=vval,
                                     depth=Z,
                                     num_segments=num_segments,
                                     shape_x0=shape_x0,
                                     dia_flat=dia_flat,
                                     ax=ax2,
                                     ylbl=False,
                                     ls=ls,
                                     )

        ax1.plot(df_roots[j].U, df_roots[j].z * -1e6, ls, label=vs[j])

    ax2.set_xlabel('r (mm)', fontsize=14)
    ax2.set_title("x0={}, flat={}".format(shape_x0, dia_flat * 1e3))

    ax1.set_xlabel('V (V)', fontsize=14)
    ax1.set_ylabel('z (um)', fontsize=14)
    ax1.legend(title=k)
    ax1.grid(alpha=0.25)
    ax1.set_title('Pre-stretch={}'.format(pre_stretch))
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_sweep_z-by-v_and_rz-profile.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()


def plot_sweep_strain_by_z():
    # the below plotting method should be packaged into a function
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    # ---

    # plot

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6, 5))

    for j in range(len(dfs)):

        df = dfs[j][0]
        vval = vs[j]
        ls = mrkrs[j]

        df['strain_xy'] = df['stretch_i'] / df['stretch_i'].iloc[0]
        df['strain_z'] = df['t_i'] / df['t_i'].iloc[0]
        df['strain_z_inv'] = 1 / df['strain_z']

        if export_excel_strain:
            df.to_excel(join(path_save, save_id + '_strain-by-z_{}_{}.xlsx'.format(k, vval)))

        # plot

        ax1.plot(df['dZ'] * 1e6, df['t_i'] * 1e6, ls, label=vval)

        if ls == 'b-':
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-', label='in-plane')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--', label='out-of-plane')
        else:
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--')

        # -

    ax1.set_ylabel('Memb. Thickness', fontsize=14)
    ax1.grid(alpha=0.25)
    ax1.legend(title=k)

    ax2.set_ylabel('Strain', fontsize=14)
    ax2.set_xlabel('z (um)', fontsize=14)
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '_strain-by-z.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()