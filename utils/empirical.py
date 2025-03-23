import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


def read_tck(filepath):
    df = pd.read_excel(filepath)
    t = df.t.to_numpy()
    c = df.c.to_numpy()
    k = int(filepath[-6])
    tck = (t, c, k)
    return tck


def profile_tck(tck, x):
    return BSpline(*tck)(x)


def profile_tck2solver(r, z):
    r = (r - np.max(r)) * -1
    z = z - np.max(z)

    r = np.flip(r)
    z = np.flip(z)

    r = r[:np.argmin(z)]
    z = z[:np.argmin(z)]

    return r, z


def dict_from_tck(wid, fid, depth, radius, units, num_segments, fp_tck, r_min=None):
    tck = read_tck(filepath=fp_tck)

    if r_min is None:
        r_min = 0
    r = np.linspace(r_min, radius, num_segments)
    z = profile_tck(tck, r)
    rnew, znew = profile_tck2solver(r, z)

    rnew = rnew * units
    znew = znew * units

    dict_fid = {
        'wid': wid,
        'fid': fid,
        'depth': depth,
        'radius': radius,
        'units': units,
        'tck': tck,
        'r_raw': r,
        'z_raw': z,
        'r': rnew,
        'z': znew,
    }

    return dict_fid


# --- SURFACE PROFILES

def read_surface_profile(dict_settings, subset=None, hole=True, fid_override=None):
    # read surface profile
    df = pd.read_excel(dict_settings['path_process_profiles'])
    # -
    # get feature profile
    if fid_override is not None:
        fid = fid_override
    else:
        fid = dict_settings['fid']
    if 'step_process_profile' in dict_settings.keys():
        step = dict_settings['step_process_profile']
    else:
        step = df['step'].max()
    df = df[(df['fid'] == fid) & (df['step'] == step)]
    # -
    # keep only necessary columns
    df = df[['x', 'z', 'r']]
    # filter out profile where through-hole has been etched
    if hole:
        df = df[df['r'].abs() > dict_settings['radius_hole_microns']]

    # return a subset of the profile
    if subset is None:
        pass  # return the full-width profile
    elif subset == 'full':
        pass  # alternate method to return the full-width profile
    elif subset == 'abs':
        df['r'] = df['r'].abs()
    elif subset == 'right_half':
        df = df[df['r'] > 0]
    elif subset == 'left_half':
        df = df[df['r'] < 0]
        df['r'] = df['r'].abs()
    else:
        raise ValueError("Options are: [None or full, abs, right_half, left_half]")

    return df


def get_zipping_interface_r_from_z(z0, surf_r, surf_z):
    # make sure the arrays are sorted
    # primary sort: r
    sorted_indices = np.argsort(surf_r)
    # Use these indices to sort both arrays
    sorted_r = surf_r[sorted_indices]
    sorted_z = surf_z[sorted_indices]
    # find smallest r that satisfies
    rmin_nearest = sorted_r[np.argmax(np.abs(sorted_z - z0) < 2)]
    return rmin_nearest


def get_zipping_interface_rz(r, z, surf_r, surf_z):
    # If you get problems, try using the code below to enforce limits on r, z
    """
    # make sure the arrays are sorted
    surf_z = surf_z[np.argsort(surf_r)]
    surf_r = surf_r[np.argsort(surf_r)]
    # define the maximum allowable zipping interface
    surf_r_max = surf_r[np.argmax(surf_z < -2)]
    surf_z_max = surf_z[np.argmax(surf_z < -2)]
    """

    # 0th-pass: estimate r, z as average of particles nearest the center
    zipping_interface_r = np.percentile(r, 7.5)
    zipping_interface_z = np.mean(z[r < zipping_interface_r])

    # 1st-pass: get zipping interface
    zipping_interface_r = get_zipping_interface_r_from_z(
        z0=zipping_interface_z,
        surf_r=surf_r,
        surf_z=surf_z,
    )
    try:
        # 2nd-pass: refine zipping interface
        r_fringing = np.percentile(r[r < zipping_interface_r], 75)
        # r_midpoint = (zipping_interface_r + np.min(r)) / 2
        zipping_interface_z = np.mean(z[(r < zipping_interface_r) & (r > r_fringing)])
        zipping_interface_r = get_zipping_interface_r_from_z(
            z0=zipping_interface_z,
            surf_r=surf_r,
            surf_z=surf_z,
        )
    except IndexError:
        pass
    return zipping_interface_r, zipping_interface_z

