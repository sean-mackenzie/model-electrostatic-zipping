# tests/test_compare_data_to_model.py

# imports
import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # model
    FP_MODEL = ('/Users/mackenzie/Desktop/zipper_paper/Modeling/apply model to my wafers/'
                'first-pass_by-wid/wid14/wid14_fid3_sweep-E_z-by-v.xlsx')
    DFM = pd.read_excel(FP_MODEL)
    # data
    FP_DATA = ('/Users/mackenzie/Desktop/zipper_paper/Testing/Zipper Actuation/'
               '01132025_W14-F1_C9-0pT/analyses-sigma=0.5/coords/tid1_merged-coords-volt.xlsx')
    DF = pd.read_excel(FP_DATA)
    # save
    PATH_SAVE = ('/Users/mackenzie/Desktop/zipper_paper/Testing/Zipper Actuation/'
                 '01132025_W14-F1_C9-0pT/analyses-sigma=0.5/representative_test1')
    # ---
    # -
    # -- setup
    # model
    mx, my = 'U', 'z'
    Es = [2e6]
    # data
    dx, dy = 'VOLT', 'dz'
    pids = [1]
    # --- pre-processing
    DFM = DFM[DFM[mx] < 205]
    DF = DF[DF['STEP'] < 41]
    # -
    # plot
    fig, ax = plt.subplots(figsize=(3.75, 2.5))
    for E in Es:
        DFM_ = DFM[DFM.E == E]
        ax.plot(DFM_[mx], DFM_[my] * -1e6, 'r-', label='Model')
    for pid in pids:
        DF_ = DF[DF['id'] == pid]
        ax.plot(DF_[dx], DF_[dy], 'ko', ms=1.5, label='Data')
    ax.set_xlabel(r'$V_{app} \: (V)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        join(PATH_SAVE, 'compare_data_to_model_pid=1_E=2MPa_pre-stretch=1.025.png'),
        dpi=300,
        facecolor='w',
        bbox_inches='tight',
    )
    plt.show()