
from os.path import join
import numpy as np
import pandas as pd


def pre_process_coords(df, settings):
    # system settings
    # 3D particle tracking
    scale_z = settings['scale_z']
    padding = settings['padding']
    drop_frame_zero = settings['drop_frame_zero']
    # Imaging and images
    xc_pixels, yc_pixels = settings['xyc_pixels']
    xc, yc = settings['xyc_microns']
    microns_per_pixel = settings['microns_per_pixel']
    frame_rate = settings['frame_rate']
    andor_keithley_delay = settings['andor_keithley_delay']

    # Format the coords
    keep_cols = ['frame', 'id', 'cm', 'xm', 'ym', 'xg', 'yg', 'z']
    rename_cols = {'xm': 'xm_pix', 'ym': 'ym_pix', 'xg': 'xg_pix', 'yg': 'yg_pix'}
    df = df[keep_cols].rename(columns=rename_cols)

    # Remove 3D particle tracking artifacts
    # scale z
    df['z'] = df['z'] * scale_z
    # remove faux baseline
    if drop_frame_zero:
        df = df[df['frame'] != 0]
    # Transform coordinates from image space (frames, pixels) to physical space (seconds, microns)
    # frames-to-seconds
    df['t_raw'] = df['frame'] / frame_rate  # (s)  where time only depends on image acquisition rate
    df['t_sync'] = df['frame'] / frame_rate + andor_keithley_delay  # (s) where time is synchronized to Keithley
    # NOTE: 't_sync' is supposed to represent the mid-exposure time point.
    df = df.round({'t_raw': 3, 't_sync': 3})

    for x, y, r in zip(['xm', 'xg'], ['ym', 'yg'], ['rm', 'rg']):
        # remove padding
        df[x + '_pix'] = df[x + '_pix'] - padding
        df[y + '_pix'] = df[y + '_pix'] - padding
        # pixels-to-microns
        df[x] = df[x + '_pix'] * microns_per_pixel  # (microns)
        df[y] = df[y + '_pix'] * microns_per_pixel  # (microns)
        # Relate image coordinates to image features
        # relative to r = 0 (i.e., center of diaphragm)
        df[r + '_pix'] = np.sqrt((df[x + '_pix'] - xc_pixels) ** 2 + (df[y + '_pix'] - yc_pixels) ** 2)
        df[r] = np.sqrt((df[x] - xc) ** 2 + (df[y] - yc) ** 2)
    # -
    return df

def calculate_relative_coords(df, test_settings, df0=None):
    # Define relative positions (i.e., displacement)
    # relative to: t < t_start
    # time
    df['dt'] = df['t_raw']
    # space
    if df0 is None:
        df0 = calculate_initial_coords(df, test_settings)
    dfpids = []
    for pid in df['id'].unique():
        dfpid = df[df['id'] == pid]
        for v in ['xg', 'yg', 'rg', 'z', 'xm', 'ym', 'rm',
                  'xg_pix', 'yg_pix', 'rg_pix', 'xm_pix', 'ym_pix', 'rm_pix']:
            dfpid['d' + v] = dfpid[v] - df0.loc[pid][v]
        dfpids.append(dfpid)
    df = pd.concat(dfpids)
    df = df.reset_index(drop=True)
    return df

def calculate_initial_coords(df, test_settings):
    # test-specific params
    start_frame = test_settings['dpt_start_frame']
    df0 = df[(df['frame'] > start_frame[0]) & (df['frame'] < start_frame[1])].groupby('id').mean()
    return df0

def merge_with_iv(df, dfv):
    """
    Note, the index of df and dfv is important.
    If df index is not [0, 1, 2, etc...], then I-V
    data will not be correctly matched with coords.

    :param df:
    :param dfv:
    :return:
    """
    dfv_mid_source = dfv.groupby('STEP').mean().reset_index()
    dfv_mid_source = dfv_mid_source.round({'TIME': 4})

    result = find_closest(
        original_values=df['t_sync'].to_numpy(),
        reference_values=dfv_mid_source['TIME'].to_numpy(),
    )

    df['SOURCE_TIME_MIDPOINT'] = pd.Series(result)
    df = df.merge(right=dfv_mid_source, left_on='SOURCE_TIME_MIDPOINT', right_on='TIME')
    df = df.drop(columns=['TIME'])
    return df


def find_closest(original_values, reference_values):
    # Function to find the closest value
    def closest(num, options):
        return min(options, key=lambda x: abs(x - num))

    # Replace each element in the original list with the closest from the reference list
    closest_values = [closest(value, reference_values) for value in original_values]
    return closest_values


def calculate_net_displacement(df, pxyrz, start_frame, end_frames, path_results):
    if isinstance(start_frame, int):
        start_frame = (0, start_frame)

    px = 'frame'
    x, y, r, z = pxyrz
    pids = df.sort_values(by=r, ascending=True)['id'].unique()

    y_means = []
    for pid in pids:
        dfpid = df[df['id'] == pid]

        # initial r-z position
        x_pos_mean_i = dfpid[(dfpid[px] > start_frame[0]) & (dfpid[px] < start_frame[1])][x].mean()
        y_pos_mean_i = dfpid[(dfpid[px] > start_frame[0]) & (dfpid[px] < start_frame[1])][y].mean()
        r_pos_mean_i = dfpid[(dfpid[px] > start_frame[0]) & (dfpid[px] < start_frame[1])][r].mean()
        z_pos_mean_i = dfpid[(dfpid[px] > start_frame[0]) & (dfpid[px] < start_frame[1])][z].mean()

        # dr analysis
        r_mean_i = r_pos_mean_i  # dfpid[(dfpid[px] > start_frame[0]) & (dfpid[px] < start_frame[1])][r].mean()
        r_mean_f = dfpid[(dfpid[px] > end_frames[0]) & (dfpid[px] < end_frames[1])][r].mean()
        dr_mean = np.round(r_mean_f - r_mean_i, 2)

        # dz analysis
        z_mean_i = z_pos_mean_i  # dfpid[(dfpid[px] > start_frame[0]) & (dfpid[px] < start_frame[1])][z].mean()
        z_mean_f = dfpid[(dfpid[px] > end_frames[0]) & (dfpid[px] < end_frames[1])][z].mean()
        dz_mean = np.round(z_mean_f - z_mean_i, 2)

        y_means.append([pid,
                        np.round(x_pos_mean_i, 1),
                        np.round(y_pos_mean_i, 1),
                        np.round(r_pos_mean_i, 1),
                        np.round(z_pos_mean_i, 1),
                        # np.round(r_mean_i, 2),
                        # np.round(z_mean_i, 1),
                        np.round(r_mean_f, 2),
                        np.round(z_mean_f, 1),
                        dr_mean,
                        dz_mean,
                        np.round((dr_mean + dfpid[r].mean()) / dfpid[r].mean(), 4),
                        np.round(dr_mean / dz_mean, 4),
                        start_frame[0],
                        start_frame[1],
                        end_frames[0],
                        end_frames[1],
                        ])

    df_rz_means = pd.DataFrame(np.array(y_means),
                               columns=['id', x, y, r, z,
                                        # r + '_mean_i', z + '_mean_i',
                                        r + '_mean_f', z + '_mean_f',
                                        'dr_mean', 'dz_mean',
                                        'r_strain', 'drdz',
                                        'start_frame_i', 'start_frame_f',
                                        'end_frame_i', 'end_frame_f',
                                        ])
    df_rz_means = df_rz_means.sort_values(by='dz_mean', ascending=True)
    df_rz_means.to_excel(join(path_results, 'net-dzr_per_pid.xlsx'), index=False)

    return df_rz_means