
from os.path import join
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import plotting

# ---


def compare_dZ_by_V_with_models(df, dfm, path_results, save_id, dz, mkey, mvals, title=None,):
    # -- setup
    # data
    dx, dy = 'VOLT', dz
    df = df[df['STEP'] <= df['STEP'].iloc[df['VOLT'].abs().idxmax()]]
    # -
    # plot
    fig, ax = plt.subplots(figsize=(3.75, 2.5))
    if isinstance(mvals, (list, np.ndarray)):
        lss = ['-', '--', '-.', ':']
        clrs = ['k', 'r', 'b', 'g']
        for mval, clr, ls in zip(mvals, clrs, lss):
            dfm_mval = dfm[dfm[mkey] == mval]
            x, y = plotting.pre_process_model_dZ_by_V_for_compare(
                dfm=dfm_mval, mkey=mkey, mval=mval, extend_max_voltage=df['VOLT'].max())
            ax.plot(x, y, clr+ls, label=np.round(mval * 1e-6, 2))
    else:
        dfm_mval = dfm[dfm[mkey] == mvals]
        x, y = plotting.pre_process_model_dZ_by_V_for_compare(
            dfm=dfm_mval, mkey=mkey, mval=mvals, extend_max_voltage=df['VOLT'].max())
        ax.plot(x, y, 'r-', label='Model')

    ax.plot(df[dx].abs(), df[dy], 'ko', ms=1.5, label=save_id)
    ax.set_xlabel(r'$V_{app} \: (V)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.grid(alpha=0.25)
    ax.legend(fontsize='small', loc='lower left')
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(
        join(path_results, f'compare_data_to_model_{save_id}_{dz}.png'),
        dpi=300,
        facecolor='w',
        bbox_inches='tight',
    )
    plt.close()


def compare_dZmin_by_V_with_model(df, dfm, path_results, save_id, dz, mkey, mvals, title=None,):
    # -- setup
    # data
    dx, dy = 'VOLT', dz
    df = df[df['STEP'] <= df['STEP'].iloc[df['VOLT'].abs().idxmax()]]
    # -
    # plot
    fig, ax = plt.subplots(figsize=(3.75, 2.5))
    # -
    # model
    if isinstance(mvals, (list, np.ndarray)):
        lss = ['-', '--', '-.', ':']
        clrs = ['k', 'r', 'b', 'g']
        for mval, clr, ls in zip(mvals, clrs, lss):
            dfm_mval = dfm[dfm[mkey] == mval]
            x, y = plotting.pre_process_model_dZ_by_V_for_compare(
                dfm=dfm_mval, mkey=mkey, mval=mval, extend_max_voltage=df['VOLT'].max())
            ax.plot(x, y, clr + ls, label=np.round(mval * 1e-6, 2))
    else:
        dfm_mval = dfm[dfm[mkey] == mvals]
        x, y = plotting.pre_process_model_dZ_by_V_for_compare(
            dfm=dfm_mval, mkey=mkey, mval=mvals, extend_max_voltage=df['VOLT'].max())
        ax.plot(x, y, 'r-', label='Model')
    # -
    # data
    xp, yp = df[dx].abs(), df[dy]
    ax.plot(xp[0], yp[0], 'ko', ms=1.5, label=save_id)

    ii = np.arange(len(xp))
    yp_min = yp[0]
    for i, xi, yi in zip(ii, xp[1:], yp[1:]):
        if yi < yp_min:
            ax.plot(xi, yi, 'ko', ms=1.5)
            yp_min = yi

    #ax.plot(xp, yp, 'ko', ms=1.5, label='Data')
    #ax.plot(df[dx].abs(), df[dy], 'ko', ms=1.5, label='Data')
    ax.set_xlabel(r'$V_{app} \: (V)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.grid(alpha=0.25)
    ax.legend(fontsize='small', loc='lower left')
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(
        join(path_results, f'compare_data_to_model_{save_id}_{dz}-min.png'),
        dpi=300,
        facecolor='w',
        bbox_inches='tight',
    )
    plt.close()



# ---

if __name__ == "__main__":

    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    TEST_CONFIG = '03122025_W13-D1_C15-15pT_25nmAu'
    # Coords
    TID = 1  # 6, 7,
    ONLY_PIDS = None # if None, will plot all pids or defer (sequentially) to following filters
    D0Z_MIN = -175
    D0Z_QUANTILE = 0.02
    # Model
    MODEL_DIR = 'comp_sweep_comp_E'
    MODEL_MKEY = 'comp_E'
    MODEL_MVAL = 4600000
    LABEL_MVAL = '4.6MPa'
    VMAX = 300  # if VMAX is lower than model's Vmax, then do nothing

    # ---

    # directories
    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024'
    BASE_DIR = join(ROOT_DIR, 'zipper_paper/Testing/Zipper Actuation', TEST_CONFIG)
    ANALYSES_DIR = join(BASE_DIR, 'analyses')
    READ_MODEL = join(ANALYSES_DIR, 'modeling')
    READ_COORDS = join(ANALYSES_DIR, 'coords')
    READ_SETTINGS = join(ANALYSES_DIR, 'settings')
    READ_NET_D0ZR = join(ANALYSES_DIR, 'net-d0zr_per_pid')
    SAVE_COMBINED = join(ANALYSES_DIR, 'combined')
    SAVE_DIR = join(ANALYSES_DIR, 'custom', f'tid{TID}-vs-{MODEL_DIR}')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # ---

    read_coords = True
    if read_coords:
        FP_COORDS = 'tid{}_merged-coords-volt.xlsx'.format(TID)
        DF_COORDS = pd.read_excel(join(READ_COORDS, FP_COORDS))

    read_model = True
    if read_model:
        mfiles = [x for x in os.listdir(join(READ_MODEL, MODEL_DIR)) if x.endswith('.xlsx')]
        mfile_z_by_v = [x for x in mfiles if x.endswith('z-by-v.xlsx')][0]
        mfile_strain_by_z = [x for x in mfiles if x.endswith('strain-by-z.xlsx')][0]

        DF_MODEL_VDZ = pd.read_excel(join(READ_MODEL, MODEL_DIR, mfile_z_by_v))
        DF_MODEL_STRAIN = pd.read_excel(join(READ_MODEL, MODEL_DIR, mfile_strain_by_z))

    # ---

    df = DF_COORDS
    only_pids = ONLY_PIDS
    df_model_VdZ = DF_MODEL_VDZ
    save_dir = SAVE_DIR
    model_mkey, model_mval = MODEL_MKEY, MODEL_MVAL
    model_mvals = df_model_VdZ[model_mkey].unique()

    save_compare_target_model = join(save_dir, f'{MODEL_MKEY}={LABEL_MVAL}')
    save_compare_sweep_model = join(save_dir, f'sweep-{MODEL_MKEY}')
    for pth in [save_compare_target_model, save_compare_sweep_model]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    titles = [None, f'{MODEL_MKEY}={LABEL_MVAL}']

    # ---
    xym = 'g'
    px, py, pr, pdx, pdy, pdr, pd0x, pd0y, pd0r = [k + xym for k in ['x', 'y', 'r', 'dx', 'dy', 'dr', 'd0x', 'd0y', 'd0r']]
    pz, pdz, pd0z = 'z', 'dz', 'd0z'
    pdz_lock_in, pd0z_lock_in = 'dz_lock_in', 'd0z_lock_in'
    # -
    # compare with model
    dz_compare_with_model = pd0z  # pd0z or pdz
    dz_lock_in_compare_with_model = dz_compare_with_model + '_lock_in'
    if dz_lock_in_compare_with_model in df.columns:
        dz_ = dz_lock_in_compare_with_model
    else:
        dz_ = dz_compare_with_model

    # ---

    if only_pids is None:
        if D0Z_MIN is not None:
            only_pids = df[df[pd0z] < D0Z_MIN]['id'].unique()
        else:
            only_pids = df[df[pd0z] < df[pd0z].quantile(D0Z_QUANTILE)]['id'].unique()

    for mv, sp, ttl in zip([model_mvals, model_mval], [save_compare_sweep_model, save_compare_target_model], titles):

        save_dz = join(sp, dz_)
        save_dz_min = join(sp, dz_ + '-min')
        for pth in [save_dz, save_dz_min]:
            if not os.path.exists(pth):
                os.makedirs(pth)

        for pid in only_pids:
            df_pid = df[df['id'] == pid].reset_index(drop=True)
            compare_dZ_by_V_with_models(
                df=df_pid,
                dfm=df_model_VdZ,
                path_results=save_dz,
                save_id=f'pid{int(pid)}',
                dz=dz_,
                mkey=model_mkey,
                mvals=mv,
                title=ttl,
            )

            compare_dZmin_by_V_with_model(
                df=df_pid,
                dfm=df_model_VdZ,
                path_results=save_dz_min,
                save_id=f'pid{int(pid)}',
                dz=dz_compare_with_model,
                mkey=model_mkey,
                mvals=mv,
                title=ttl,
            )






    # -









    # -

    print("Completed without errors.")