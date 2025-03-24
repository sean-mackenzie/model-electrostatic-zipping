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
        ax.legend(title=key_title, loc='upper left', fontsize='small')
    else:
        ax.legend(title=key, loc='upper left', fontsize='small')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + f'_sweep-{key}_z-by-v.png'), dpi=300, facecolor='w')
    else:
        plt.show()
    plt.close()



if __name__ == '__main__':
    """
    NOTES ON MEMBRANE'S ESTIMATED PARAMETERS WHERE NO DIRECT MEASUREMENTS ARE AVAILABLE:
        1. C22-20pT_20nmAu (i.e., before the 2nd deposition):
            Pre-Stretch:
                * measured 1.263
            Bulge Test (must use comparable dataset):
                * C22-20pT_20+10nmAu (pre-stretch = 1.263):
                    A. E = 1 MPa, residual stress = 320 kPa
                    B. E = 14 MPa, residual stress = 310 kPa
                * C17-20pT-25nmAu (pre-stretch = 1.25):
                    A. E = 1 MPa, residual stress = 345-375 kPa
                    B. E = 23 MPa, residual stress = 305 kPa
                * C7-20pT-20nmAu (pre-stretch was not measured):        (these parameters were used for DC tests)
                    A. E = 1 MPa, residual stress = 304 kPa
            --> Conclusion:
                I'm going to use C17-20pT: E = 1 MPa, residual stress = 350 kPa to start.
        2. C21-15pT_30nmAu:
            Pre-Stretch: measured 1.159
            Bulge Test: 
                A. E = 1 MPa, residual stress = 250 kPa     (n = 4)         (USED FOR MODELING)
                B. E = 10.4 MPa, residual stress = 220 kPa  (n = 1)         (DID NOT USE FOR MODELING)
        3. C15-15pT-25nmAu:
            Pre-stretch: 
                * measured 1.146
            Bulge Test:
                A. E between 45-75 MPa, residual stress between 160-220 kPa... 
                    NOTE: for the measured pre-stretch, E=1 MPa, stress=220 kPa: 
                        Gent stress = 245 kPa, Gent pre-stretch = 1.129
                
                so, probably best to use a comparable.
                    Comparable bulge tests:
                        * C14-15pT-20nmAu (pre-stretch = 1.126):
                            A. E = 1 MPa, residual stress = 263 kPa
                        * C18-30pT-25+10nmAu (pre-stretch = 1.135):
                            Did not measure bulge test. 
                        * C19-30pT-20nm+10nmAu (pre-stretch = 1.131):
                            A. E = 1 MPa, residual stress = 265 kPa     (n = 2)
                            B. E = 10 MPa, residual stress = 262 kPa    (n = 1)
                        * C21-15pT-30nmAu (pre-stretch = 1.159):
                            A. E = 1 MPa, residual stress = 250 kPa     (n = 4)
            --> Conclusion:
                I'm going to use the same parameters as C21-15pT-30nmAu: E = 1 MPa, residual stress = 250 kPa to start.
        
    """
    # ---
    """ NOTE: these values are computed via Bulge Test and/or calculate_pre_stretch_from_residual_stress.py."""
    MEMB_ID = 'C15-15pT-25nmAu'
    MEMB_YOUNGS_MODULUS_BULGE_TEST = 75e6  # Pa
    MEMB_RESIDUAL_STRESS_BULGE_TEST = 220e3  # Pa
    EXPERIMENTAL_PRE_STRETCH_NOMINAL = 1.15
    EXPERIMENTAL_PRE_STRETCH_MEASURED = 1.146
    GENT_MODEL_COMPUTED_RESIDUAL_STRESS_FROM_PRE_STRETCH_MEASURED = 245e3  # Pa
    GENT_MODEL_COMPUTED_PRE_STRETCH = 1.129
    GENT_MODEL_J_MEMB = 54  # Elastosil 200 um thick: 54 or 16; Maffli et al. NuSil: 80.4

    # ---

    # general inputs
    TEST_CONFIG = '03122025_W13-D1_C15-15pT_25nmAu'
    WID = 13
    TID = 1
    ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation'
    BASE_DIR = join(ROOT_DIR, TEST_CONFIG)
    ANALYSES_DIR = join(BASE_DIR, 'analyses')
    READ_SETTINGS = join(ANALYSES_DIR, 'settings')
    SAVE_DIR = join(ANALYSES_DIR, 'modeling')
    # settings
    FP_SETTINGS = join(READ_SETTINGS, 'dict_settings.xlsx')
    FP_TEST_SETTINGS = join(READ_SETTINGS, 'dict_tid{}_settings.xlsx'.format(TID))
    DICT_SETTINGS = settings.get_settings(fp_settings=FP_SETTINGS, name='settings', update_dependent=False)
    FID_PROCESS_PROFILE = DICT_SETTINGS['fid_process_profile']


    """ NOTE: The model actually uses these values here. You can play with these values. """
    MODEL_USE_PRE_STRETCH = 1.146  # GENT_MODEL_COMPUTED_PRE_STRETCH  # pre-stretch
    MODEL_USE_YOUNGS_MODULUS = 1.0e6  # (Pa) MEMB_YOUNGS_MODULUS_BULGE_TEST
    MODEL_USE_THICKNESS_DIELECTRIC = 2.1e-6
    MODEL_USE_SURFACE_ROUGHNESS = 1e-9

    FID_OVERRIDE = None  # None
    SURFACE_PROFILE_SUBSET = 'left_half'  # 'left_half', 'right_half', 'full'
    TCK_SMOOTHING = 1.0
    SWEEP_PARAM = 'pre_stretch'  # 'E', 'pre_stretch', 'Jm'
    DICT_SETTINGS_RADIUS = DICT_SETTINGS['radius_microns']
    MODEL_USE_RADIUS = DICT_SETTINGS_RADIUS + 25

    NUM_SEGMENTS = 1000  # NOTE: this isn't necessarily the final number of solver segments
    COMPUTE_DEPTH_DEPENDENT_STRAIN = True  # True False

    if FID_OVERRIDE is None:
        FID = DICT_SETTINGS['fid_process_profile']
    else:
        FID = FID_OVERRIDE

    SAVE_ID = 'wid{}_fid{}'.format(WID, FID)
    SAVE_DIR = join(SAVE_DIR, 'fid{}_E{}_PS{}_{}_s={}_sweep-{}'.format(FID,
                                                                       np.round(MODEL_USE_YOUNGS_MODULUS *1e-6, 1), MODEL_USE_PRE_STRETCH,
                                                                       SURFACE_PROFILE_SUBSET, TCK_SMOOTHING, SWEEP_PARAM))

    SWEEP_VOLTAGES = np.arange(65, 250, 1)

    if SWEEP_PARAM == 'E':
        SWEEP_K = 'E'
        SWEEP_VS = np.array([1, 1.1, 1.2]) * 1e6
        SWEEP_K_FIG = "E (MPa)"
        SWEEP_VS_FIGS = np.round(SWEEP_VS * 1e-6, 1)
        SWEEP_K_XLSX = 'E'
        SWEEP_VS_XLSX = [f'{x}MPa' for x in SWEEP_VS_FIGS]
    elif SWEEP_PARAM == 'pre_stretch':
        SWEEP_K = 'pre_stretch'
        SWEEP_VS = [1.1, 1.129, 1.146]
        SWEEP_K_FIG = "Pre-stretch"
        SWEEP_VS_FIGS = SWEEP_VS
        SWEEP_K_XLSX = 'pre_stretch'
        SWEEP_VS_XLSX = SWEEP_VS
    elif SWEEP_PARAM == 'Jm':
        SWEEP_K = 'Jm'
        SWEEP_VS = [16, 54, 80]
        SWEEP_K_FIG = "Jm"
        SWEEP_VS_FIGS = SWEEP_VS
        SWEEP_K_XLSX = 'Jm'
        SWEEP_VS_XLSX = SWEEP_VS
    elif SWEEP_PARAM == 'FINAL':
        SWEEP_K = 'E'
        SWEEP_VS = [MODEL_USE_YOUNGS_MODULUS]
        SWEEP_K_FIG = "E (MPa)"
        SWEEP_VS_FIGS = [np.round(MODEL_USE_YOUNGS_MODULUS / 1e6, 2)]
        SWEEP_K_XLSX = 'E'
        SWEEP_VS_XLSX = ['{} MPa'.format(np.round(MODEL_USE_YOUNGS_MODULUS / 1e6, 2))]
    else:
        raise ValueError('Invalid sweep parameter: {}'.format(SWEEP_PARAM))

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    #  directories
    READ_TCK = join(SAVE_DIR, 'tck')
    for pth in [SAVE_DIR, READ_TCK]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    FN_TCK = 'fid{}_tc_k=3.xlsx'.format(FID)
    FP_TCK = join(READ_TCK, FN_TCK)

    # surface profile
    INCLUDE_THROUGH_HOLE = True
    DF_SURFACE = empirical.read_surface_profile(
        DICT_SETTINGS,
        subset='full',  # this should always be 'full', because profile will get slice during tck
        hole=INCLUDE_THROUGH_HOLE,
        fid_override=FID_OVERRIDE,
    )

    # specific inputs
    DEPTH = DF_SURFACE['z'].abs().max()
    RADIUS = MODEL_USE_RADIUS
    MAX_DZ_FOR_STRAIN_PLOT = DEPTH - 2
    UNITS = 1e-6

    # ---
    PRE_PROCESS_TCK = True
    # if PRE_PROCESS_TCK or not os.path.exists(FP_TCK):
    # --- --- MANUALLY FIT TCK AND EXPORT
    SMOOTHING = TCK_SMOOTHING  # 50
    NUM_POINTS = 500
    DEGREE = 3
    # fit tck
    tck, rmin, rmax = manually_fit_tck(df=DF_SURFACE, subset=SURFACE_PROFILE_SUBSET, radius=RADIUS,
                           smoothing=SMOOTHING, num_points=NUM_POINTS, degree=DEGREE,
                           path_save=READ_TCK)
    DICT_TCK_SETTINGS = {
        'wid': WID,
        'fid': FID,
        'depth': DEPTH,
        'radius': RADIUS,
        'radius_min': rmin,
        'radius_max': rmax,
        'subset': SURFACE_PROFILE_SUBSET,
        'smoothing': SMOOTHING,
        'num_points': NUM_POINTS,
        'degree': DEGREE,
    }
    # export tck
    DF_TCK = pd.DataFrame(np.vstack([tck[0], tck[1]]).T, columns=['t', 'c'])
    DF_TCK_SETTINGS = pd.DataFrame.from_dict(data=DICT_TCK_SETTINGS, orient='index', columns=['v'])
    with pd.ExcelWriter(FP_TCK) as writer:
        for sheet_name, df, idx, idx_lbl in zip(['tck', 'settings'], [DF_TCK, DF_TCK_SETTINGS], [False, True], [None, 'k']):
            df.to_excel(writer, sheet_name=sheet_name, index=idx, index_label=idx_lbl)
    #else:
    #    DF_TCK = pd.read_excel(FP_TCK, sheet_name='tck')

    dict_fid = dict_from_tck(WID, FID, DEPTH, RADIUS, UNITS, NUM_SEGMENTS, fp_tck=FP_TCK, r_min=rmin)

    # ----

    # ---

    # ---
    config = 'MoB'
    shape = 'circle'
    diameter = dict_fid['radius'] * 2 * UNITS
    depth = DEPTH * UNITS
    pre_stretch = MODEL_USE_PRE_STRETCH
    t = 20e-6  # membrane thickness = 25 microns
    t_diel = MODEL_USE_THICKNESS_DIELECTRIC

    # material
    Youngs_Modulus = MODEL_USE_YOUNGS_MODULUS
    # mu_memb = Youngs_Modulus / 3  # 0.42e6
    J_memb = GENT_MODEL_J_MEMB  # 80.4
    eps_r_memb = 3.0
    eps_r_diel = 3.0
    surface_roughness_diel = MODEL_USE_SURFACE_ROUGHNESS  # units: meters

    # voltage (can be made a dependent variable or hard-coded?)
    U_is = SWEEP_VOLTAGES   # np.arange(25, 150, 0.5)  # (5, 255, 5)

    DICT_MODEL_SETTINGS = {
        'save_id': SAVE_ID,
        'test_config': TEST_CONFIG,
        'wid': WID,
        'fid_tested': FID_PROCESS_PROFILE,
        'membrane_id': MEMB_ID,
        'surface_fid_override': FID_OVERRIDE,
        'surface_profile_subset': SURFACE_PROFILE_SUBSET,
        'surface_include_hole': INCLUDE_THROUGH_HOLE,
        'depth': DEPTH,
        'radius': RADIUS,
        'dict_settings_radius': DICT_SETTINGS_RADIUS,
        'tck_smoothing': SMOOTHING,
        'max_dz_for_strain_plot': MAX_DZ_FOR_STRAIN_PLOT,
        'memb_youngs_modulus_bulge_test': MEMB_YOUNGS_MODULUS_BULGE_TEST,
        'memb_residual_stress_bulge_test': MEMB_RESIDUAL_STRESS_BULGE_TEST,
        'experimental_pre_stretch_nominal': EXPERIMENTAL_PRE_STRETCH_NOMINAL,
        'experimental_pre_stretch_measured': EXPERIMENTAL_PRE_STRETCH_MEASURED,
        'gent_model_computed_residual_stress_from_pre_stretch_measured': GENT_MODEL_COMPUTED_RESIDUAL_STRESS_FROM_PRE_STRETCH_MEASURED,
        'gent_model_computed_pre_stretch': GENT_MODEL_COMPUTED_PRE_STRETCH,
        'gent_model_j_memb': GENT_MODEL_J_MEMB,
        'units': UNITS,
        'model_num_segments': NUM_SEGMENTS,
        'model_config': config,
        'model_shape': shape,
        'model_diameter_um': diameter,
        'model_depth_um': depth,
        'model_thickness_membrane_um': t,
        'model_pre_stretch': pre_stretch,
        'model_thickness_dielectric_um': t_diel,
        'model_youngs_modulus': Youngs_Modulus,
        'model_J_memb': J_memb,
        'model_eps_r_memb': eps_r_memb,
        'model_eps_r_diel': eps_r_diel,
        'model_surface_roughness_diel': surface_roughness_diel,
        'model_Vmin': U_is.min(),
        'model_Vmax': U_is.max(),
        'model_Vstep': U_is[1] - U_is[0],
    }
    DF_MODEL_SETTINGS = pd.DataFrame.from_dict(data=DICT_MODEL_SETTINGS, orient='index', columns=['v'])
    DF_MODEL_SETTINGS.to_excel(join(SAVE_DIR, SAVE_ID + '_model_settings.xlsx'), index=True, index_label='k')

    # profile to use
    px, py = dict_fid['r'], dict_fid['z']
    #px = np.linspace(px.min(), px.max(), 3000)
    #py = np.linspace(py.max(), py.min(), 3000)
    if not os.path.exists(join(SAVE_DIR, SAVE_ID + '_profile.png')):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(px * 1e3, py * 1e6)
        ax.set_xlabel('mm')
        ax.set_ylabel('um')
        ax.grid(alpha=0.25)
        ax.set_title('dz/segment = {} um ({} segs)'.format(np.round(np.min(py * 1e6) / len(px), 2), len(px)))
        plt.suptitle(SAVE_ID)
        plt.tight_layout()
        plt.savefig(join(SAVE_DIR, SAVE_ID + '_profile.png'), facecolor='w')
        plt.show()
        plt.close()
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
    if SWEEP_K in dict_material.keys():
        d = dict_material
    elif SWEEP_K in dict_actuator.keys():
        d = dict_actuator
    else:
        raise ValueError('Invalid sweep key: {}'.format(SWEEP_K))
    # d = dict_material  # dict_actuator dict_material
    k = SWEEP_K  # 'E'  # 'pre_stretch' 'dia_flat' 'x0'
    vs = SWEEP_VS  # [2e6, 2.5e6, 3e6, 4e6]
    k_fig = SWEEP_K_FIG  # "E (MPa)"
    vs_fig = SWEEP_VS_FIGS  # [2, 2.5, 3, 4]
    k_xlsx = SWEEP_K_XLSX  # 'E'
    vs_xlsx = SWEEP_VS_XLSX  # ['2MPa', '2.5MPa', '3MPa', '4MPa']

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
    df_s.to_excel(join(SAVE_DIR, SAVE_ID + '_model_z-by-v.xlsx'), index=False)

    plot_sweep_z_by_v(df_roots, key=k, vals=vs_fig, path_save=SAVE_DIR, save_id=SAVE_ID, key_title=k_fig)

    # raise ValueError()



    # ------------------------------------------------------------------------------------------------------------------
    # RUN FUNCTION - SOLVE AND PLOT STRAIN-BY-Z
    # ------------------------------------------------------------------------------------------------------------------

    if COMPUTE_DEPTH_DEPENDENT_STRAIN:
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
                                                              save_id=SAVE_ID)
            dfs.append(dfs1)
            df_roots.append(df_roots1)

        # ---

        # the below plotting method should be packaged into a function
        mrkrs = ['b-', 'r--', 'g-.', 'k-']

        # ---

        # plot
        path_save = SAVE_DIR

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
                df.to_excel(join(path_save, SAVE_ID + '_model_strain-by-z.xlsx'))

            # plot
            if MAX_DZ_FOR_STRAIN_PLOT is not None:
                df = df[df['dZ'] * 1e6 < MAX_DZ_FOR_STRAIN_PLOT]

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
            plt.savefig(join(path_save, SAVE_ID + f'_sweep-{k}_strain-by-z.png'), dpi=300, facecolor='w')
        else:
            plt.show()
        plt.close()