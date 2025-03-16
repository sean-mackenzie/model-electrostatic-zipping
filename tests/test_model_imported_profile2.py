# tests/test_model_sweep.py

# imports
import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.pyplot as plt

from utils import energy, plotting, shapes, empirical, settings
from utils.empirical import dict_from_tck
from tests.test_manually_fit_tck_to_surface_profile import manually_fit_tck
from model_energy_minimization import solve_energy_iterative_shape_function


def plot_sweep_z_by_v(df_roots, key, vals, path_save, save_id, key_title=None):
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for j in range(len(df_roots)):
        ax.plot(df_roots[j].U, df_roots[j].z * 1e6, mrkrs[j], label=vals[j])

    ax.set_xlabel('V (V)', fontsize=14)
    ax.set_ylabel('z (um)', fontsize=14)
    if key_title is not None:
        ax.legend(title=key_title)
    else:
        ax.legend(title=key)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + f'_sweep-{key}_z-by-v.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()



if __name__ == '__main__':
    # general inputs
    TEST_CONFIG = '01102025_W13-D1_C9-0pT'
    WID = 13
    TID = 1

    # directories
    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation'
    BASE_DIR = join(ROOT_DIR, TEST_CONFIG)
    ANALYSES_DIR = join(BASE_DIR, 'analyses')
    READ_SETTINGS = join(ANALYSES_DIR, 'settings')
    SAVE_DIR = join(ANALYSES_DIR, 'modeling')
    READ_TCK = join(SAVE_DIR, 'tck')
    for pth in [SAVE_DIR, READ_TCK]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # settings
    FP_SETTINGS = join(READ_SETTINGS, 'dict_settings.xlsx')
    FP_TEST_SETTINGS = join(READ_SETTINGS, 'dict_tid{}_settings.xlsx'.format(TID))
    DICT_SETTINGS = settings.get_settings(fp_settings=FP_SETTINGS, name='settings', update_dependent=False)
    FID = DICT_SETTINGS['fid_process_profile']
    FN_TCK = 'fid{}_tc_k=3.xlsx'.format(FID)
    FP_TCK = join(READ_TCK, FN_TCK)

    # surface profile
    SURFACE_PROFILE_SUBSET = 'full'  # 'full', 'left_half', 'right_half'
    DF_SURFACE = empirical.read_surface_profile(
        DICT_SETTINGS,
        subset=SURFACE_PROFILE_SUBSET,
        hole=False,
        fid_override=None,
    )

    # specific inputs
    DEPTH = DF_SURFACE['z'].abs().max()
    RADIUS = DICT_SETTINGS['radius_microns'] + 20
    MAX_DZ_FOR_STRAIN_PLOT = DEPTH - 3
    UNITS = 1e-6
    NUM_SEGMENTS = 3500  # NOTE: this isn't necessarily the final number of solver segments
    # ---
    # --- --- MANUALLY FIT TCK AND EXPORT
    SUBSET = 'right_half'
    SMOOTHING = 50
    NUM_POINTS = 500
    DEGREE = 3
    DICT_TCK_SETTINGS = {
        'wid': WID,
        'fid': FID,
        'depth': DEPTH,
        'radius': RADIUS,
        'subset': SUBSET,
        'smoothing': SMOOTHING,
        'num_points': NUM_POINTS,
        'degree': DEGREE,
    }
    # fit tck
    tck = manually_fit_tck(df=DF_SURFACE, subset=SUBSET, radius=RADIUS,
                           smoothing=SMOOTHING, num_points=NUM_POINTS, degree=DEGREE,
                           path_save=READ_TCK)
    # export tck
    DF_TCK = pd.DataFrame(np.vstack([tck[0], tck[1]]).T, columns=['t', 'c'])
    DF_TCK_SETTINGS = pd.DataFrame.from_dict(data=DICT_TCK_SETTINGS, orient='index', columns=['v'])
    with pd.ExcelWriter(FP_TCK) as writer:
        for sheet_name, df, idx, idx_lbl in zip(['tck', 'settings'], [DF_TCK, DF_TCK_SETTINGS], [False, True], [None, 'k']):
            df.to_excel(writer, sheet_name=sheet_name, index=idx, index_label=idx_lbl)

    dict_fid = dict_from_tck(WID, FID, DEPTH, RADIUS, UNITS, NUM_SEGMENTS, fp_tck=FP_TCK)

    SAVE_ID = 'wid{}_fid{}_test1'.format(WID, FID)
    # ---
    config = 'MoB'
    shape = 'circle'
    diameter = dict_fid['radius'] * 2 * UNITS
    depth = DEPTH * UNITS
    t = 20e-6  # membrane thickness = 25 microns
    pre_stretch = 1.025  # pre-stretch
    t_diel = 2e-6

    # material
    Youngs_Modulus = 2e6
    # mu_memb = Youngs_Modulus / 3  # 0.42e6
    J_memb = 80.4
    eps_r_memb = 3.0
    eps_r_diel = 3.0
    surface_roughness_diel = 0.0

    # voltage (can be made a dependent variable or hard-coded?)
    U_is = np.arange(5, 255, 5)

    # profile to use
    px, py = dict_fid['r'], dict_fid['z']
    plt.plot(px * 1e3, py * 1e6)
    plt.xlabel('mm')
    plt.ylabel('um')
    plt.grid(alpha=0.25)
    #plt.suptitle(save_id)
    #plt.savefig(join(save_dir, save_id + '.png'), facecolor='w')
    plt.show()
    plt.close()
    #raise ValueError()
    #"""


    # dict_actuator: shape, profile_x, profile_z, membrane_thickness, pre_stretch, dielectric_thickness
    dict_actuator = {
        'shape': shape,
        'diameter': diameter,
        'depth': depth,
        'membrane_thickness': t,
        'pre_stretch': pre_stretch,
        'dielectric_thickness': t_diel,
        'profile_x': px,
        'profile_z': py,
    }
    dict_material = {
        'E': Youngs_Modulus,  # 'mu': mu_memb,
        'Jm': J_memb,
        'eps_r_memb': eps_r_memb,
        'eps_r_diel': eps_r_diel,
        'surface_roughness_diel': surface_roughness_diel,
    }

    # execution modifiers

    append_dfs = False
    """
    If False,
        INPUTS: U_is is an array of values
        RETURN: df_roots1
        POST-PROCESSING: use df_roots1 to plot z vs. V

    If True, 
        INPUTS: U_is is a single value (presumably, voltage where 100% pull-in occurs)
        RETURN: dfs, df_roots
        POST-PROCESSING: use dfs[0] to calculate strain, then plot
    """

    export_excel = False  # Note: if U_is is array, export_excel can be a U_i value (e.g., 100) to export only that
    """
    If number,
        export only dataframe corresponding to that voltage value to .xlsx (useful for validation)
    If True,
        export all dataframes to .xlsx (usually, undesirable)
    If False:
        do nothing
    """

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # RUN FUNCTION - SOLVE AND PLOT Z-BY-V
    # ------------------------------------------------------------------------------------------------------------------

    # setup
    d = dict_material  # dict_actuator dict_material
    k = 'E'  # 'pre_stretch' 'dia_flat' 'x0'
    vs = [2e6, 3e6, 4e6]
    k_fig = "E (MPa)"
    vs_fig = [2, 3, 4]
    k_xlsx = 'E'
    vs_xlsx = ['2MPa', '3MPa', '4MPa']

    # solve
    df_roots = []
    for i in range(len(vs)):
        d.update({k: vs[i]})
        df_roots1 = solve_energy_iterative_shape_function(config,
                                                          dict_actuator,
                                                          dict_material,
                                                          U_is,
                                                          append_dfs=append_dfs,
                                                          export_excel=export_excel,  # 100,
                                                          save_id=SAVE_ID,
                                                          path_save=SAVE_DIR,
                                                          )

        df_roots.append(df_roots1)

    df_s = []
    for df_, v in zip(df_roots, vs):
        df_[k] = v
        df_s.append(df_)
    df_s = pd.concat(df_s)
    df_s.to_excel(join(SAVE_DIR, SAVE_ID + '_sweep-{}_z-by-v.xlsx'.format(k)), index=False)

    plot_sweep_z_by_v(df_roots, key=k, vals=vs_fig, path_save=SAVE_DIR, save_id=SAVE_ID, key_title=k_fig)

    raise ValueError()

    # ------------------------------------------------------------------------------------------------------------------
    # RUN FUNCTION - SOLVE AND PLOT STRAIN-BY-Z
    # ------------------------------------------------------------------------------------------------------------------

    # setup

    # voltage
    U_is = [df_roots[-1].U.iloc[-1]]

    # solve
    append_dfs = True
    export_excel = False  # Note: if U_is is array, export_excel can be a U_i value (e.g., 100) to export only that
    export_excel_strain = True

    dfs = []
    df_roots = []
    for i in [0]: # range(len(vs)):
        d.update({k: vs[i]})
        dfs1, df_roots1 = solve_energy_iterative_shape_function(config,
                                                          dict_actuator,
                                                          dict_material,
                                                          U_is,
                                                          append_dfs=append_dfs,
                                                          export_excel=export_excel,  # 100,
                                                          save_id=save_id)
        dfs.append(dfs1)
        df_roots.append(df_roots1)

    # ---

    # the below plotting method should be packaged into a function
    mrkrs = ['b-', 'r--', 'g-.', 'k-']

    # ---

    # plot
    path_save = save_dir

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(6, 6))

    for j in range(len(dfs)):

        df = dfs[j][0]
        vval = vs[j]
        ls = mrkrs[j]

        df['strain_xy'] = df['stretch_i'] / df['stretch_i'].iloc[0]
        df['strain_z'] = df['t_i'] / df['t_i'].iloc[0]
        df['strain_z_inv'] = 1 / df['strain_z']
        df['disp_r_microns'] = (df['x_i'] * (df['strain_xy'] - 1)) / 2 * 1e6  # divide by 2 = radial displacement

        if export_excel_strain:
            df.to_excel(join(path_save, save_id + '_strain-by-z_{}_{}.xlsx'.format(k_xlsx, vs_xlsx[j])))

        # plot
        if max_dz_for_strain_plot is not None:
            df = df[df['dZ'] * 1e6 < max_dz_for_strain_plot]

        ax1.plot(df['dZ'] * 1e6, df['t_i'] * 1e6, ls, label=vs_fig[j])

        if ls == 'b-':
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-', label='in-plane')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--', label='out-of-plane')
        else:
            ax2.plot(df['dZ'] * 1e6, df['strain_xy'], color=ls[0], ls='-')
            ax2.plot(df['dZ'] * 1e6, df['strain_z_inv'], color=ls[0], ls='--')

        ax3.plot(df['dZ'] * 1e6, df['disp_r_microns'], ls)

        # -

    ax1.set_ylabel('t (um)', fontsize=14)
    ax1.grid(alpha=0.25)
    ax1.legend(title=k_fig)

    ax2.set_ylabel('Strain', fontsize=14)
    ax2.grid(alpha=0.25)
    ax2.legend()

    ax3.set_ylabel(r'$\Delta r \: (\mu m)$', fontsize=14)
    ax3.set_xlabel('z (um)', fontsize=14)
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + f'_sweep-{k}_strain-by-z.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()