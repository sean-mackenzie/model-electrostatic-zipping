import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt

from utils import plotting, settings

def make_ivac_test_matrix(read_iv, filepath_save):
    files = [f for f in os.listdir(read_iv) if f.startswith('tid')]
    res = []
    for f in files:
        tid = int(f.split('_')[0].split('tid')[1])
        test_type = f.split('_')[1].split('test')[1]
        output_volt = f.split('_')[2].split('V')[0]
        awg_freq = f.split('_')[3].split('Hz')[0]
        awg_wave = f.split('_')[3].split('Hz')[1]
        res.append([tid, test_type, output_volt, awg_freq, awg_wave])
    df = pd.DataFrame(res, columns=['tid', 'test_type', 'output_volt', 'awg_freq', 'awg_wave'])
    df = df.astype({'tid': int, 'test_type': str, 'output_volt': int, 'awg_freq': float, 'awg_wave': str})
    df.to_excel(filepath_save, index=False)
    return df

def get_ivac_test_matrix(base_dir, save_dir):
    read_iv = join(base_dir, 'I-V', 'xlsx')
    fp_iv_matrix = join(save_dir, 'iv_test_matrix.xlsx')
    if not os.path.exists(fp_iv_matrix):
        df_iv_matrix = make_ivac_test_matrix(read_iv=read_iv, filepath_save=fp_iv_matrix)
    else:
        df_iv_matrix = pd.read_excel(fp_iv_matrix)
        df_iv_matrix = df_iv_matrix.astype(
            {'tid': int, 'test_type': str, 'output_volt': int, 'awg_freq': float, 'awg_wave': str})
    return df_iv_matrix

def get_tids_from_iv_matrix(df_iv_matrix, dict_test_group):
    # example dict_test_group: {'test_type': 'STD1', 'output_volt': 230, 'awg_wave': 'SQU'}
    df = df_iv_matrix.copy()
    for k, v in dict_test_group.items():
        df = df[df[k] == v]
    tids = df['tid'].unique()
    return tids

def make_all_merged_coords_volt(base_dir, save_dir, return_df=False):
    fn_endswith = '_merged-coords-volt.xlsx'
    read_dir = join(base_dir, 'analyses', 'coords')
    tids = [int(x.split(fn_endswith)[0].split('tid')[1]) for x in os.listdir(read_dir) if x.endswith(fn_endswith)]
    tids.sort()
    df = []
    for tid in tids:
        print("Reading tid: {} ...".format(tid))
        df_ = pd.read_excel(join(read_dir, 'tid{}{}'.format(tid, fn_endswith)))
        df_.insert(0, 'tid', tid)
        df.append(df_)
    df = pd.concat(df)
    df.to_excel(join(save_dir, 'all' + fn_endswith), index=False)
    if return_df:
        return df

def get_all_merged_coords_volt(base_dir, save_dir):
    filepath = join(save_dir, 'all_merged-coords-volt.xlsx')
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
    else:
        df = make_all_merged_coords_volt(base_dir=base_dir, save_dir=save_dir, return_df=True)
    return df

def get_joined_merged_coords_volt_and_iv_matrix(df_merged_coords_volt, df_iv_matrix, base_dir, save_dir):
    filepath = join(save_dir, 'joined_merged-coords-volt_and_iv_matrix.xlsx')
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
    else:
        if df_merged_coords_volt is None:
            df_merged_coords_volt = get_all_merged_coords_volt(base_dir, save_dir)
        if df_iv_matrix is None:
            df_iv_matrix = get_ivac_test_matrix(base_dir=base_dir, save_dir=save_dir)
        df = df_merged_coords_volt.join(df_iv_matrix.set_index('tid'), on='tid', how='left', lsuffix='_mcv', rsuffix='_iv')
        df.to_excel(filepath, index=False)
    return df

def make_all_net_d0zr_per_pid(base_dir, xym, save_dir, return_df=False):
    fn = 'net-d0zr_per_pid'
    fn_endswith = '_' + fn + '.xlsx'
    read_dir = join(base_dir, 'analyses', fn, 'xy' + xym)
    tids = [int(x.split(fn_endswith)[0].split('tid')[1]) for x in os.listdir(read_dir) if x.endswith(fn_endswith)]
    tids.sort()
    df = []
    for tid in tids:
        print("Reading tid: {} ...".format(tid))
        df_ = pd.read_excel(join(read_dir, 'tid{}{}'.format(tid, fn_endswith)))
        df_.insert(0, 'tid', tid)
        df.append(df_)
    df = pd.concat(df)
    df = df.sort_values(['tid', 'id'], ascending=[True, True])
    df.to_excel(join(save_dir, 'all' + fn_endswith), index=False)
    if return_df:
        return df

def get_all_net_d0zr_per_pid(base_dir, save_dir, xym):
    filepath = join(save_dir, 'all_net-d0zr_per_pid.xlsx')
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
    else:
        df = make_all_net_d0zr_per_pid(base_dir=base_dir, xym=xym, save_dir=save_dir, return_df=True)
    return df

def get_joined_net_d0zr_and_iv_matrix(df_net_d0zr_per_pid, df_iv_matrix, base_dir, xym, save_dir):
    filepath = join(save_dir, 'joined_net-d0zr_and_iv_matrix.xlsx')
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
    else:
        if df_net_d0zr_per_pid is None:
            df_net_d0zr_per_pid = get_all_net_d0zr_per_pid(base_dir=base_dir, save_dir=save_dir, xym=xym)
        if df_iv_matrix is None:
            df_iv_matrix = get_ivac_test_matrix(base_dir=base_dir, save_dir=save_dir)
        df = df_net_d0zr_per_pid.join(df_iv_matrix.set_index('tid'), on='tid', how='left', lsuffix='_net', rsuffix='_iv')
        df.to_excel(filepath, index=False)
    return df



if __name__ == "__main__":

    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    TEST_CONFIG = '03122025_W13-D1_C15-15pT_25nmAu'

    #TODO: scatter plot (per-pid) of voltage vs. z-position for all tests:
    # (DC tests will reveal little dependence b/c hysteresis. AC tests should show good dependence.)


    # ------------------------------------------------------------------------------------------------------------------
    # YOU SHOULD NOT NEED TO CHANGE BELOW
    # ------------------------------------------------------------------------------------------------------------------
    # ---
    # FILEPATHS
    # ---
    # directories
    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024'
    BASE_DIR = join(ROOT_DIR, 'zipper_paper/Testing/Zipper Actuation', TEST_CONFIG)
    SAVE_DIR = join(BASE_DIR, 'analyses')
    READ_COORDS = join(SAVE_DIR, 'coords')
    READ_SETTINGS = join(SAVE_DIR, 'settings')
    READ_NET_D0ZR = join(SAVE_DIR, 'net-d0zr_per_pid')
    SAVE_COMBINED = join(SAVE_DIR, 'combined')
    SAVE_COMBINED_MCV = join(SAVE_COMBINED, 'merged-coords-volt_per_tid')
    SAVE_COMBINED_NET_D0ZR = join(SAVE_COMBINED, 'net-d0zr_per_pid_per_tid')
    XYM = 'g'  # 'g' or 'm': use sub-pixel or discrete in-plane localization method
    # filenames
    FN_IV_MATRIX = 'iv_test_matrix.xlsx'
    FN_ALL_MERGED_COORDS_VOLT = 'all_merged-coords-volt.xlsx'
    FN_ALL_NET_D0ZR_PER_PID = 'all_net-d0zr_per_pid.xlsx'
    FN_JOINED_MERGED_COORDS_VOLT_AND_IV_MATRIX = 'joined_merged-coords-volt_and_iv_matrix.xlsx'
    FN_JOINED_NET_D0ZR_AND_IV_MATRIX = 'joined_net-d0zr_and_iv_matrix.xlsx'
    # filepaths
    FP_IV_MATRIX = join(SAVE_COMBINED, FN_IV_MATRIX)
    FP_ALL_MERGED_COORDS_VOLT = join(SAVE_COMBINED, FN_ALL_MERGED_COORDS_VOLT)
    FP_ALL_NET_D0ZR_PER_PID = join(SAVE_COMBINED, FN_ALL_NET_D0ZR_PER_PID)

    # make dirs
    for pth in [SAVE_COMBINED, SAVE_COMBINED_NET_D0ZR]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # initialize
    DF_IV_MATRIX = None
    DF_MCV = None
    DF_NET_D0ZR = None
    DF_MODEL_VDZ = None
    DF_MODEL_STRAIN = None
    ARR_MODEL_VDZ = None
    # make iv test matrix
    make_ivac_matrix = False
    if make_ivac_matrix:
        DF_IV_MATRIX = get_ivac_test_matrix(base_dir=BASE_DIR, save_dir=SAVE_COMBINED)
    # stack merged-coords-volt for all tids
    make_merged_coords_volt = False
    if make_merged_coords_volt:
        DF_MCV = make_all_merged_coords_volt(base_dir=BASE_DIR, save_dir=SAVE_COMBINED, return_df=True)
    # stack net-d0zr_per_pid for all tids
    make_net_d0zr_per_pid = False
    if make_net_d0zr_per_pid:
        DF_NET_D0ZR = make_all_net_d0zr_per_pid(base_dir=BASE_DIR, xym=XYM, save_dir=SAVE_COMBINED, return_df=True)
    # join net-d0zr_per_pid and iv test matrix
    join_net_d0zr_and_iv_matrix = False
    if join_net_d0zr_and_iv_matrix:
        DFDIV = get_joined_net_d0zr_and_iv_matrix(df_net_d0zr_per_pid=DF_NET_D0ZR, df_iv_matrix=DF_IV_MATRIX,
                                                  base_dir=BASE_DIR, xym=XYM, save_dir=SAVE_COMBINED)
    # read model data
    read_model_data = True
    if read_model_data:
        dict_settings = settings.get_settings( fp_settings=join(READ_SETTINGS, 'dict_settings.xlsx'), name='settings')
        if 'path_model' in dict_settings.keys():
            mfiles = [x for x in os.listdir(dict_settings['path_model'].strip("'")) if x.endswith('.xlsx')]
            mfile_z_by_v = [x for x in mfiles if x.endswith('z-by-v.xlsx')][0]
            mfile_strain_by_z = [x for x in mfiles if x.endswith('strain-by-z.xlsx')][0]

            DF_MODEL_VDZ = pd.read_excel(join(dict_settings['path_model'], mfile_z_by_v))
            DF_MODEL_STRAIN = pd.read_excel(join(dict_settings['path_model'], mfile_strain_by_z))
        elif 'path_model_dZ_by_V' or 'path_model_strain' in dict_settings.keys():
            DF_MODEL_VDZ = pd.read_excel(dict_settings['path_model_dZ_by_V'].strip("'"))
            DF_MODEL_STRAIN = pd.read_excel(dict_settings['path_model_strain'].strip("'"))
        if DF_MODEL_VDZ is not None:
            MKEY, MVAL, VMAX = 'pre_stretch', 1.146, 300
            model_V, model_dZ = plotting.pre_process_model_dZ_by_V_for_compare(
                dfm=DF_MODEL_VDZ, mkey=MKEY, mval=MVAL, extend_max_voltage=VMAX)
            ARR_MODEL_VDZ = (model_V, model_dZ)
    # ---

    #TODO: should incorporate test-type filters into directory name where outputs get saved
    only_test_types = ['STD1', 'STD2', 'STD3', 'VAR3']
    threshold_pids_by_dz_quantile = 0.1  # b/c dz is negative, smaller quantiles correspond to larger deflections
    # ---

    plot_merged_coords_volt_per_pid_by_tid = True
    if plot_merged_coords_volt_per_pid_by_tid:
        SAVE_SUB_ANALYSIS = join(SAVE_COMBINED_MCV, 'per_pid_by_all-volt-freq')
        if not os.path.exists(SAVE_SUB_ANALYSIS):
            os.makedirs(SAVE_SUB_ANALYSIS)


        DF_MCVIV = get_joined_merged_coords_volt_and_iv_matrix(
            df_merged_coords_volt=DF_MCV,
            df_iv_matrix=DF_IV_MATRIX,
            base_dir=BASE_DIR,
            save_dir=SAVE_COMBINED,
        )

        df = DF_MCVIV
        tids = df.tid.unique()
        df_ascending = []
        for tid in tids:
            df_tid = df[df['tid'] == tid].reset_index(drop=True)
            df_tid = df_tid[df_tid['STEP'] <= df_tid['STEP'].iloc[df_tid['VOLT'].abs().idxmax()]]
            df_ascending.append(df_tid)
        df_ascending = pd.concat(df_ascending)
        df_ascending.to_excel(join(SAVE_COMBINED, 'joined_merged-coords-volt_and_iv_matrix_ascending-only.xlsx'))

        plotting.plot_heatmap_merged_coords_volt_all_pids_by_volt_freq(
            df=df_ascending, # DF_MCVIV,
            path_save=SAVE_COMBINED_MCV,
            threshold_pids_by_dz_quantile=threshold_pids_by_dz_quantile,
            only_test_types=only_test_types,
        )
        raise ValueError()
        plotting.plot_merged_coords_volt_per_pid_by_volt_freq(
            df=DF_MCVIV,
            path_save=SAVE_SUB_ANALYSIS,
            threshold_pids_by_dz_quantile=threshold_pids_by_dz_quantile,
            only_test_types=only_test_types,
            arr_model_VdZ=ARR_MODEL_VDZ,
        )

    # ---

    plot_net_d0zr_per_pid_by_tid = False
    if plot_net_d0zr_per_pid_by_tid:
        SAVE_SUB_ANALYSIS = join(SAVE_COMBINED_NET_D0ZR, 'net-d0zr_per_pid_by_volt-freq')
        if not os.path.exists(SAVE_SUB_ANALYSIS):
            os.makedirs(SAVE_SUB_ANALYSIS)

        DFDIV = get_joined_net_d0zr_and_iv_matrix(df_net_d0zr_per_pid=DF_NET_D0ZR, df_iv_matrix=DF_IV_MATRIX,
                                                  base_dir=BASE_DIR, xym=XYM, save_dir=SAVE_COMBINED)

        plot_all_pid = True
        if plot_all_pid:
            plotting.plot_net_d0zr_all_pids_by_volt_freq(
                df=DFDIV,
                path_save=SAVE_COMBINED_NET_D0ZR,
                threshold_pids_by_dz_quantile=threshold_pids_by_dz_quantile,
                only_test_types=only_test_types,
                arr_model_VdZ=ARR_MODEL_VDZ,
            )

            plotting.plot_heatmap_net_d0zr_all_pids_by_volt_freq(
                df=DFDIV,
                path_save=SAVE_COMBINED_NET_D0ZR,
                threshold_pids_by_dz_quantile=threshold_pids_by_dz_quantile,
                only_test_types=only_test_types,
            )

            for shift_V_by_X_log_freq in [0, 2, 4]:
                plotting.plot_net_d0zr_all_pids_by_volt_freq_errorbars_per_tid(
                    df=DFDIV,
                    path_save=SAVE_COMBINED_NET_D0ZR,
                    threshold_pids_by_dz_quantile=threshold_pids_by_dz_quantile,
                    only_test_types=only_test_types,
                    shift_V_by_X_log_freq=shift_V_by_X_log_freq,
                    arr_model_VdZ=ARR_MODEL_VDZ,
                )

        plot_per_pid = True
        if plot_per_pid:
            plotting.plot_net_d0zr_per_pid_by_volt_freq(
                df=DFDIV,
                path_save=SAVE_SUB_ANALYSIS,
                only_test_types=only_test_types,
                arr_model_VdZ=ARR_MODEL_VDZ,
            )









    # --- BELOW: OLD... GOOD BUT NEED TO BE CLEANED UP

    plot_overlay_pid_by_tid = False
    if plot_overlay_pid_by_tid:
        SAVE_SUB_ANALYSIS = join(SAVE_COMBINED, 'pid_dz_by_tid')
        if not os.path.exists(SAVE_SUB_ANALYSIS):
            os.makedirs(SAVE_SUB_ANALYSIS)

        SAVE_IDS = [
            'STD2 230V',
            'STD3 230V',
            'VAR3 230V LowFreq',
            'STD1SIN 230V',
        ]
        TIDSS = [
            [53, 20, 24, 50],
            [54, 32, 21, 25, 51],
            [35, 34, 33, 41],
            [43, 42, 40],
        ]
        LBLSS = [
            [0.25, 1, 10, 50],
            [0.25, 0.25, 1, 10, 50],
            [100, 150, 250, 1000],
            [250, 500, 750],
        ]
        LEGEND_TITLES = ['Freq (kHz)', 'Freq (kHz)', 'Freq (Hz)', 'Freq (mHz)']

        PIDS = None  # [1, 6, 12, 14, 17, 18, 20, 22, 27]  # None: plot all pids
        px, py1, py2 = 't_sync', 'd0z', 'd0rg'

        for SAVE_ID, TIDS, LBLS, LGND_TTL in zip(SAVE_IDS, TIDSS, LBLSS, LEGEND_TITLES):

            SAVE_SUB_SUB_ANALYSIS = join(SAVE_SUB_ANALYSIS, SAVE_ID)
            if not os.path.exists(SAVE_SUB_SUB_ANALYSIS):
                os.makedirs(SAVE_SUB_SUB_ANALYSIS)

            DFS = {}
            for TID in TIDS:
                DF = pd.read_excel(join(READ_COORDS, 'tid{}_merged-coords-volt.xlsx'.format(TID)))
                if PIDS is not None:
                    DF = DF[DF['id'].isin(PIDS)]
                else:
                    PIDS = DF['id'].unique()
                DFS[TID] = DF

            for PID in PIDS:
                # sort TIDS and LBLS by LBLS
                LBLS, TIDS = zip(*sorted(zip(LBLS, TIDS)))

                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(7, 5))
                for TID, LBL in zip(TIDS, LBLS):
                    DF = DFS[TID]
                    DF_PID = DF[DF['id'] == PID]
                    ax1.plot(DF_PID[px], DF_PID[py1], '-', lw=0.5, label=LBL)
                    ax2.plot(DF_PID[px], DF_PID[py2], '-', lw=0.5, label=LBL)

                ax1.set_ylabel(r'$\Delta_{o} z \: (\mu m)$')
                ax1.grid(alpha=0.2)
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title=LGND_TTL, fontsize='x-small')

                ax2.set_ylabel(r'$\Delta_{o} r \: (\mu m)$')
                ax2.grid(alpha=0.2)
                ax2.set_xlabel(r'$t \: (s)$')

                plt.tight_layout()
                plt.savefig(join(SAVE_SUB_SUB_ANALYSIS, 'pid{}.png'.format(PID)),
                            dpi=300, facecolor='w', bbox_inches='tight')
                plt.close(fig)


    plot_effective_pull_in_voltage2 = False
    if plot_effective_pull_in_voltage2:

        TIDS = [5, 13, 17, 22, 27, 31, 41, 50, 67]
        VAPPS = [100, 125, 150, 170, 190, 200, 220, 240, 260]

        IV_ACDC = 'AC'
        SAVE_ID = '250 Hz SQU - except tid67'
        px, py = 'VOLT', 'dz_mean'
        VAPP_0DZ = None  # 90
        quantile = 0.12  # 075

        PATH_REPRESENTATIVE = join(SAVE_DIR, 'net-d0zr_per_pid')  # 'representative_test{}')
        FN_NET_DZR = 'tid{}_net-d0zr_per_pid.xlsx'
        FN_SAVE_NET_DZR = 'net-d0zr_per_pid.xlsx'
        SAVE_SUB_ANALYSIS = join(SAVE_COMBINED, SAVE_ID)
        SAVE_PID_D0Z_BY_VOLTAGE = join(SAVE_SUB_ANALYSIS, 'pid_d0z_by_voltage')

        # make dirs
        for pth in [SAVE_COMBINED, SAVE_SUB_ANALYSIS, SAVE_PID_D0Z_BY_VOLTAGE]:
            if not os.path.exists(pth):
                os.makedirs(pth)

        DFDS = []
        for TID, VAPP in zip(TIDS, VAPPS):
            # Read merged coords-IV data
            # FN_MERGED = 'tid{}_merged-coords-volt.xlsx'.format(TID)
            # DF = pd.read_excel(join(SAVE_COORDS, FN_MERGED))
            DFD = pd.read_excel(join(PATH_REPRESENTATIVE.format(TID), 'xy' + XYM, FN_NET_DZR.format(TID)))

            if TID == TIDS[0]:
                D0FD = DFD.copy()
                D0FD['tid'] = -1
                D0FD['VOLT'] = 0
                D0FD[py] = 0
                DFDS.append(D0FD)

                if VAPP_0DZ is not None:
                    D1FD = DFD.copy()
                    D1FD['tid'] = 0
                    D1FD['VOLT'] = VAPP_0DZ
                    D1FD[py] = 0
                    DFDS.append(D1FD)

            DFD['tid'] = TID
            DFD['VOLT'] = VAPP
            DFDS.append(DFD)
        DFD = pd.concat(DFDS)
        DFD = DFD.sort_values('VOLT', ascending=True)
        DFD.to_excel(join(SAVE_SUB_ANALYSIS, 'combined_' + FN_SAVE_NET_DZR), index=False)

        # ---

        # plot
        include_tid_lbls = True

        q = DFD.groupby('id').min()[py].quantile(quantile)
        pids = DFD[DFD[py] < q]['id'].unique()
        for pid in pids:
            DFD_PID = DFD[DFD['id'] == pid]

            fig, ax = plt.subplots()

            if include_tid_lbls:
                ax.plot(DFD_PID[px], DFD_PID[py], 'k-', zorder=3.1)
                for tid in DFD_PID['tid'].unique():
                    DFD_PID_T = DFD_PID[DFD_PID['tid'] == tid]
                    ax.plot(DFD_PID_T[px], DFD_PID_T[py], 'o', ms=7, zorder=3.2, label=tid)
            else:
                ax.plot(DFD_PID[px], DFD_PID[py], '-o', ms=7, zorder=3.1)
            ax.set_xlabel(r'$V_{app} \: (V)$')
            ax.set_ylabel(r'$z \: (\mu m)$')
            ax.grid(alpha=0.2)
            if include_tid_lbls:
                ax.legend(loc='lower left', title=r'$T_{ID}$', ncols=2)
            ax.set_title(SAVE_ID)
            fig.savefig(join(SAVE_PID_D0Z_BY_VOLTAGE, 'pid{}.png'.format(pid)))
            plt.close(fig)


