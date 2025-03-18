import os
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


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

def get_tids_from_iv_matrix(df_iv_matrix, dict_test_group):
    # example dict_test_group: {'test_type': 'STD1', 'output_volt': 230, 'awg_wave': 'SQU'}
    df = df_iv_matrix.copy()
    for k, v in dict_test_group.items():
        df = df[df[k] == v]
    tids = df['tid'].unique()
    return tids


if __name__ == "__main__":

    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    TEST_CONFIG = '03122025_W13-D1_C15-15pT_25nmAu'


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
    SAVE_COMBINED = join(SAVE_DIR, 'combined')
    SECOND_PASS_XYM = 'g'  # 'g' or 'm': use sub-pixel or discrete in-plane localization method

    # make dirs
    for pth in [SAVE_COMBINED]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # make iv test matrix
    READ_IV = join(BASE_DIR, 'I-V', 'xlsx')
    FP_IV_MATRIX = join(SAVE_COMBINED, 'iv_test_matrix.xlsx')
    if not os.path.exists(FP_IV_MATRIX):
        DF_IV_MATRIX = make_ivac_test_matrix(read_iv=READ_IV, filepath_save=FP_IV_MATRIX)
    else:
        DF_IV_MATRIX = pd.read_excel(FP_IV_MATRIX)
        DF_IV_MATRIX = DF_IV_MATRIX.astype({'tid': int, 'test_type': str, 'output_volt': int, 'awg_freq': float, 'awg_wave': str})

    # ---

    # ---

    plot_overlay_pid_by_tid = True
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





    plot_effective_pull_in_voltage = False
    if plot_effective_pull_in_voltage:

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
            DFD = pd.read_excel(join(PATH_REPRESENTATIVE.format(TID), 'xy' + SECOND_PASS_XYM, FN_NET_DZR.format(TID)))

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


