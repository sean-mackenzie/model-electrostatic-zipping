# tests/test_model_sweep.py

# imports
from os.path import join
import numpy as np
from scipy.interpolate import griddata

from tifffile import imread
from skimage.transform import rescale

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from utils.shapes import complete_erf_profile
from utils.empirical import read_surface_profile


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
    fid = dict_settings['fid']
    df1 = read_surface_profile(dict_settings, subset='full', hole=False, fid_override=None)
    df2 = read_surface_profile(dict_settings, subset='full', hole=True, fid_override=None)
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
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True,
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
    # ax2.set_xticks(np.arange(0, df[px].max() + 15, 25))
    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax2.grid(alpha=0.125)
    ax2.set_title(r'$\Delta r_{net}=$' + ' {} '.format(dr_mean) + r'$\mu m$')

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
    ax1.plot(df[px1], df[pdz], '-o', ms=1.5, lw=0.75, label=pid)
    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
    ax1.legend(title=r'$p_{ID}$')
    ax1.grid(alpha=0.25)
    # plot V(t)
    ax1r = ax1.twinx()
    ax1r.plot(df[px2], df[py2], '-', color='gray', lw=0.75, alpha=0.5)
    ax1r.set_ylabel(r'$V_{app} \: (V)$', color='gray', alpha=0.5)
    # plot r/dr by frames
    ax2.plot(df[px1], df[pdr], '-o', ms=1.5, lw=0.75)
    ax2.set_xlabel(r'$t \: (s)$')
    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
    ax2.grid(alpha=0.25)
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


def plot_pids_dz_by_voltage_hysteresis(df, pdzdr, dict_test, pid, path_results):
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
    dfpid_ascending = df[df['STEP'] <= dict_test['smu_step_max']]
    dfpid_descending = df[df['STEP'] > dict_test['smu_step_max']]

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
    plt.savefig(join(path_results, 'pid{}.png'.format(pid)),
                dpi=300, facecolor='w')
    plt.close()


def plot_2D_heatmap(df, pxyz, savepath=None, field=None, interpolate='linear',
                    levels=15, units=None, title=None, overlay_circles=False, dict_settings=None):
    """
    To-Do: add concentric circles around r = 0 to show radial coordinates

    :param df:
    :param pxyz:
    :param: savepath:
    :param field: (0, side-length of field-view (pixels or microns))
    :param interpolate:
    :param levels:
    :param units: two-tuple (x-y units, z units), like: ('pixels', r'$\Delta z \: (\mu m)$')
    :param title:
    :param overlay_circles:
    :param dict_settings:
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