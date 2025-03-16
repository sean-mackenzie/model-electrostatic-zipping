# tests/test_model_sweep.py

# imports
from os.path import join
import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.pyplot as plt

from scipy.interpolate import splrep, BSpline


def smooth_array(x, y, smoothing, num_points, degree=3, return_tck=False):
    """
    Find the B-spline representation of a 1-D curve.
    Refer to : https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html

    :param x:
    :param y:
    :param smoothing: 0 = no smoothing, larger values = more smoothing
    :param num_points:
    :param degree:
    :return:
    """
    num_points = int(num_points)
    tck = splrep(x, y, s=smoothing, k=degree)
    x2 = np.linspace(x.min(), x.max(), num_points)
    y2 = BSpline(*tck)(x2)
    if return_tck:
        return x2, y2, tck
    return x2, y2


def manually_fit_tck(df, subset, radius, smoothing=50, num_points=500, degree=3, path_save=None):

    # create figure and plot at each step
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 10))

    # --- STEP 1: plot both halves, to evaluate symmetry and radius
    dfp = df[(df['r'] > 0)]
    dfn = df[(df['r'] < 0)]
    dfn.loc[:, 'r'] = dfn['r'] * -1  # dfn.loc[dfn['column'] == value, 'another_column'] = new_value
    ax1.plot(dfp.r, dfp.z, '-', label='right_half: r>0')
    ax1.plot(dfn.r, dfn.z, '-', label='left_half: r<0')
    ax1.axvline(radius, color='k', ls='--', lw=0.5)
    ax1.grid(alpha=0.25)
    ax1.legend(title='subset', loc='upper left')
    ax1.set_title('Evaluate symmetry and radius')

    # --- STEP 2: refine radius
    if subset == 'left_half':
        dfp = dfn
    dfpp = dfp[(dfp['r'] < radius + 25)]
    df = dfp[(dfp['r'] < radius)].reset_index(drop=True)
    ax2.plot(dfpp.r, dfpp.z, '-', color='gray', alpha=0.5, label='radius + 25')
    ax2.plot(df.r, df.z, '-', color='r', label='radius')
    ax2.grid(alpha=0.25)
    z_at_radius = df['z'].iloc[df['r'].idxmax()]
    ax2.legend(title='z(r=radius) = {}'.format(np.round(z_at_radius, 2)), loc='upper left')

    # --- STEP 3: smooth and plot smoothed
    rx = df.r.to_numpy()
    ry = df.z.to_numpy()
    px1, py1, tck = smooth_array(rx, ry, smoothing, num_points, degree=degree, return_tck=True)
    ax3.plot(rx, ry, 'k-o', ms=2, lw=2, label='raw')
    ax3.plot(px1, py1, 'r-o', ms=1, label='smooth')
    ax3.grid(alpha=0.125)
    ax3.legend(title='subset: {}'.format(subset), loc='upper left')
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, 'manually_fit_tck_to_surface_profile.png'), dpi=300,
                    facecolor='w', bbox_inches='tight')
    plt.show()

    return tck


if __name__ == '__main__':
    fn = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Fabrication/grayscale/w5/results/w5_merged_process_profiles.xlsx'
    dfo = pd.read_excel(fn)

    # get a single profile
    fid = 6
    step = np.max(dfo.step.unique())
    df1 = dfo[(dfo['step'] == step)]
    df2 = df1[(df1['fid'] == fid)]

    # --- plot raw
    fig, ax = plt.subplots()
    ax.plot(df2.r, df2.z, '-o', ms=1)
    plt.show()

    # --- plot both halves, to evaluate symmetry and radius
    # estimate radius at first
    radius = 685

    dfp = df2[(df2['r'] > 0)]

    dfn = df2[(df2['r'] < 0)]
    # supposedly the "correct" way: dfn.loc[dfn['column'] == value, 'another_column'] = new_value
    dfn.loc[:, 'r'] = dfn['r'] * -1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dfp.r, dfp.z, '-', label='r>0')
    ax.plot(dfn.r, dfn.z, '-', label='r<0')
    ax.axvline(radius, color='k', ls='--', lw=0.5)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.show()

    # --- refine radius
    final_radius = radius + 5
    dfpp = dfp[(dfp['r'] < radius + 25)]
    df = dfp[(dfp['r'] < final_radius)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dfpp.r, dfpp.z, '-', color='gray', alpha=0.5)
    ax.plot(df.r, df.z, '-', color='r')
    ax.grid(alpha=0.25)
    plt.show()

    # --- smooth and plot smoothed
    smoothing = 50
    num_points = 500
    degree = 3

    rx = df.r.to_numpy()
    ry = df.z.to_numpy()
    px1, py1 = smooth_array(rx, ry, smoothing, num_points, degree=degree)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rx, ry, 'k-o', ms=2, lw=2, label='raw')
    ax.plot(px1, py1, 'r-o', ms=1, label='smooth')
    ax.grid(alpha=0.125)
    ax.legend()
    plt.show()