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

def dict_from_tck(wid, fid, depth, radius, units, num_segments, fp_tck=None):
    if fp_tck is None:
        fp_tck = ('/Users/mackenzie/Desktop/zipper_paper/Fabrication/grayscale/'
                  'w{}/results/profiles_tck/fid{}_tc_k=3.xlsx'.format(wid, fid))

    tck = read_tck(filepath=fp_tck)
    r = np.linspace(0, radius, num_segments)
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