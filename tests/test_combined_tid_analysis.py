import os
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    TEST_CONFIG = '03072025_W12-D1_C19-30pT_20+10nmAu'


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
    SAVE_COORDS = join(SAVE_DIR, 'coords')
    SAVE_COMBINED = join(SAVE_DIR, 'combined')
    SECOND_PASS_XYM = 'g'  # 'g' or 'm': use sub-pixel or discrete in-plane localization method


    # make dirs
    for pth in [SAVE_COMBINED]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # -

    # ---

    plot_overlay_pid_by_tid = True
    if plot_overlay_pid_by_tid:
        SAVE_SUB_ANALYSIS = join(SAVE_COMBINED, 'pid_dz_by_tid')
        if not os.path.exists(SAVE_SUB_ANALYSIS):
            os.makedirs(SAVE_SUB_ANALYSIS)

        SAVE_IDS = [
            'VAR3 300V',
            '300V 350Hz',
            '300V 250Hz',
            'VAR3 250V',
        ]
        TIDSS = [
            [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42],
            [43, 44, 45],
            [49, 50, 51, 52, 53],
        ]
        LBLSS = [
            [1000, 500, 1500, 250, 400, 350, 300, 325, 1000, 200, 150],
            ['25%', '35%', '45%'],
            ['25%', '35%', '45%'],
            [500, 250, 200, 150, 100],
        ]
        LEGEND_TITLES = ['Frequency', 'Duty Cycle', 'Duty Cycle', 'Frequency']

        PIDS = [27, 32, 33, 31, 30, 28, 35, 26, 34, 25, 14, 22]
        px, py = 't_sync', 'dz'

        for SAVE_ID, TIDS, LBLS, LGND_TTL in zip(SAVE_IDS, TIDSS, LBLSS, LEGEND_TITLES):

            SAVE_SUB_SUB_ANALYSIS = join(SAVE_SUB_ANALYSIS, SAVE_ID)
            if not os.path.exists(SAVE_SUB_SUB_ANALYSIS):
                os.makedirs(SAVE_SUB_SUB_ANALYSIS)

            DFS = {}
            for TID in TIDS:
                DF = pd.read_excel(join(SAVE_COORDS, 'tid{}_merged-coords-volt.xlsx'.format(TID)))
                DF = DF[DF['id'].isin(PIDS)]
                DFS[TID] = DF

            for PID in PIDS:
                # sort TIDS and LBLS by LBLS
                LBLS, TIDS = zip(*sorted(zip(LBLS, TIDS)))

                fig, ax = plt.subplots(figsize=(7, 3.5))
                for TID, LBL in zip(TIDS, LBLS):
                    DF = DFS[TID]
                    DF_PID = DF[DF['id'] == PID]
                    ax.plot(DF_PID[px], DF_PID[py], '-', lw=0.5, label=LBL)
                ax.set_xlabel(r'$t \: (s)$')
                ax.set_ylabel(r'$\Delta z \: (\mu m)$')
                ax.grid(alpha=0.2)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=LGND_TTL)
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


