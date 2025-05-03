import os
from os.path import join
from itertools import combinations
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import smu, awg
from utils import plotting, settings
from utils.analyses import get_surface_profile_dict
from utils.empirical import calculate_apparent_radial_displacement_due_to_rotation


def make_ivacdc_test_matrix(read_settings, filepath_save):
    files = [f for f in os.listdir(read_settings) if f.startswith('dict_tid') and f.endswith('_settings.xlsx')]
    res = []
    for f in files:
        filepath = join(read_settings, f)
        df_ = pd.read_excel(filepath)
        if 'awg_wave' in df_['k'].values:
            s = awg.read_settings_to_dict(filepath=filepath)
            if 'awg_mod_ampl_dwell' in s.keys():
                dwell = s['awg_mod_ampl_dwell']
            elif 'awg_mod_ampl_dwell_on' in s.keys():
                dwell = s['awg_mod_ampl_dwell_on']
            else:
                raise ValueError('awg_mod_ampl_dwell or awg_mod_ampl_dwell_on not found in settings file.')
            res.append([s['tid'], 'AC', s['test_type'], s['output_volt'], s['awg_freq'], s['awg_wave'],
                        dwell, s['awg_mod_ampl_step'], s['awg_mod_ampl_num_steps']])
        else:
            s = smu.read_settings_to_dict(filepath=filepath)
            res.append([s['tid'], 'DC', s['smu_test_type'], s['smu_vmax'], 0.1, 'SQU',
                        s['smu_source_delay_time'], s['smu_dv'], s['smu_step_max']])

    df = pd.DataFrame(res, columns=['tid', 'acdc', 'test_type', 'output_volt', 'awg_freq', 'awg_wave',
                                    'ampl_dwell_time', 'ampl_dv', 'ampl_num_steps'])
    df = df.astype({'tid': int, 'acdc': str, 'test_type': str, 'output_volt': int, 'awg_freq': float, 'awg_wave': str,
                    'ampl_dwell_time': float, 'ampl_dv': float, 'ampl_num_steps': int})
    df = df.sort_values(by='tid', ascending=True)
    df.to_excel(filepath_save, index=False)
    return df

def get_ivacdc_test_matrix(base_dir, save_dir):
    read_settings = join(base_dir, 'analyses', 'settings')
    fp_iv_matrix = join(save_dir, 'iv_acdc_test_matrix.xlsx')
    if not os.path.exists(fp_iv_matrix):
        df_iv_matrix = make_ivacdc_test_matrix(read_settings=read_settings, filepath_save=fp_iv_matrix)
    else:
        df_iv_matrix = pd.read_excel(fp_iv_matrix)
        df_iv_matrix = df_iv_matrix.astype(
            {'tid': int, 'acdc': str, 'test_type': str, 'output_volt': int, 'awg_freq': float, 'awg_wave': str,
                    'ampl_dwell_time': float, 'ampl_dv': float, 'ampl_num_steps': int})
    return df_iv_matrix

def get_tids_from_iv_matrix(df_iv_matrix, dict_test_group):
    # example dict_test_group: {'test_type': 'STD1', 'output_volt': 230, 'awg_wave': 'SQU'}
    df = df_iv_matrix.copy()
    for k, v in dict_test_group.items():
        df = df[df[k] == v]
    tids = df['tid'].unique()
    return tids

def find_all_combinations_from_iv_matrix(df, min_number_per_group):
    # Columns to consider for grouping
    data_columns = ['test_type', 'output_volt', 'awg_freq', 'awg_wave']
    # Generate all combinations of 3 columns from the 4 data columns
    column_combinations = list(combinations(data_columns, 3))

    results = []
    # Iterate through each combination of 3 columns
    for combination in column_combinations:
        # Determine the 4th column that is not in the combination
        remaining_column = list(set(data_columns) - set(combination))[0]
        # Group by the 3 chosen columns
        grouped = df.groupby(list(combination))
        for group_values, group_data in grouped:
            # Extract the group TIDs (tid values)
            tids = group_data['tid'].tolist()
            # If the group has more than one row, add it to the results
            if len(tids) >= min_number_per_group:
                results.append({
                    'group_tids': tids,
                    'group_definition': dict(zip(combination, group_values)),
                    'remaining_column': remaining_column
                })
    return results

def get_all_combinations_from_iv_matrix(df_iv_matrix, min_number_per_group, base_dir, save_dir):
    fn_iv_matrix_combinations = 'iv_test_matrix_combinations_n={}.xlsx'.format(min_number_per_group)
    fp_iv_matrix_combinations = join(save_dir, fn_iv_matrix_combinations)

    if df_iv_matrix is None:
        df_iv_matrix = get_ivacdc_test_matrix(base_dir=base_dir, save_dir=save_dir)
        # df_iv_matrix = df_iv_matrix[['acdc', 'test_type', 'output_volt', 'awg_freq', 'awg_wave', 'ampl_dwell_time', 'ampl_dv']]
    combinations = find_all_combinations_from_iv_matrix(df=df_iv_matrix, min_number_per_group=min_number_per_group)

    if not os.path.exists(fp_iv_matrix_combinations):
        results_df = pd.DataFrame(combinations)  # Convert the results list into a pandas DataFrame
        # Split the 'group_definition' dictionary into individual columns in the DataFrame
        try:
            group_defs = pd.DataFrame(results_df['group_definition'].tolist())
        except KeyError:
            return None
        final_results_df = pd.concat([results_df.drop(columns=['group_definition']), group_defs], axis=1)
        final_results_df.to_excel(fp_iv_matrix_combinations, index=False)
    return combinations

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

def get_joined_merged_coords_volt_and_iv_matrix(df_merged_coords_volt, df_iv_matrix, base_dir, save_dir, df_mcviv=None):
    filepath = join(save_dir, 'joined_merged-coords-volt_and_iv_matrix.xlsx')

    if df_mcviv is not None:
        return df_mcviv
    elif os.path.exists(filepath):
        df = pd.read_excel(filepath)
    else:
        if df_merged_coords_volt is None:
            df_merged_coords_volt = get_all_merged_coords_volt(base_dir, save_dir)
        if df_iv_matrix is None:
            df_iv_matrix = get_ivacdc_test_matrix(base_dir=base_dir, save_dir=save_dir)
        df = df_merged_coords_volt.join(df_iv_matrix.set_index('tid'), on='tid', how='left', lsuffix='_mcv', rsuffix='_iv')
        df.to_excel(filepath, index=False)
    return df

def make_all_zipped_coords(base_dir, save_dir, return_df=False):
    fn_endswith = '_zipped_coords.xlsx'
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

def get_all_zipped_coords(df_zc, base_dir, save_dir):
    filepath = join(save_dir, 'all_zipped_coords.xlsx')
    if df_zc is not None:
        return df_zc
    elif os.path.exists(filepath):
        df = pd.read_excel(filepath)
    else:
        df = make_all_merged_coords_volt(base_dir=base_dir, save_dir=save_dir, return_df=True)
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
            df_iv_matrix = get_ivacdc_test_matrix(base_dir=base_dir, save_dir=save_dir)
        df = df_net_d0zr_per_pid.join(df_iv_matrix.set_index('tid'), on='tid', how='left', lsuffix='_net', rsuffix='_iv')
        df.to_excel(filepath, index=False)
    return df



if __name__ == "__main__":

    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    TEST_CONFIG = '01082025_W5-D1_C9-0pT'

    # Model params
    VMAX = 200  # if VMAX is lower than model's Vmax, then do nothing

    # Other params
    ONLY_TEST_TYPES = ['STD1', 'STD2', 'STD3', 'VAR3', '1', '2', '3', 1, 2, 3]
    ONLY_PIDS = None # if None, will plot all pids or defer to dz quantile threshold
    THRESHOLD_PIDS_BY_D0Z = -110  # recommend: 90% of maximum deflection (or, 90% of chamber depth)
    MIN_TIDS_PER_COMBINATION = 3
    FREQ_SWEEP_POLY_DEG = 2  # If None, then polynominal degree will be one less than number of frequencies
    read_model_data = True
    POLY_DEG_CORRECT_RADIAL_DISPLACEMENT = 4
    Z_CLIP_SURFACE_PROFILE = -0.125  # -0.125 for most; (W11: -0.8; W13: -0.85; W5: -0.05)

    ALL_TRUE = True  # True False
    if ALL_TRUE:
        make_ivac_matrix = True
        make_ivac_matrix_combinations = True
        make_merged_coords_volt = True
        make_zipped_coords = True
        make_net_d0zr_per_pid = True
        join_net_d0zr_and_iv_matrix = True
        plot_all_pids_net_d0zr_per_pid_by_tid = True
        plot_net_d0zr_frequency_sweeps_per_pid = True
        plot_heatmap_of_all_pids_net_d0zr = True
        plot_per_pid_net_d0zr_per_pid_by_tid = True
        plot_merged_coords_volt_parametric_sweeps_per_pid_by_tid = True
        plot_merged_coords_volt_per_pid_by_all_volt_freq = True
        plot_merged_coords_volt_heat_maps = True
        plot_merged_coords_volt_ascending_only = True
        plot_zipped_coords_on_model = True
    else:
        # Make/read Excel spreadsheet modifiers
        make_ivac_matrix = False
        make_ivac_matrix_combinations = False
        make_merged_coords_volt = False
        make_zipped_coords = False
        make_net_d0zr_per_pid = False
        join_net_d0zr_and_iv_matrix = False
        # plot modifiers
        plot_all_pids_net_d0zr_per_pid_by_tid = False  # compares with model
        plot_heatmap_of_all_pids_net_d0zr = False
        plot_per_pid_net_d0zr_per_pid_by_tid = False  # compares with model
        plot_net_d0zr_frequency_sweeps_per_pid = False
        plot_merged_coords_volt_parametric_sweeps_per_pid_by_tid = False
        plot_merged_coords_volt_per_pid_by_all_volt_freq = False  # compares with model
        plot_merged_coords_volt_heat_maps = False
        plot_merged_coords_volt_ascending_only = False  # compares with model
        plot_zipped_coords_on_model = True  # compares with model

    if '_W13' in TEST_CONFIG:
        Z_CLIP_SURFACE_PROFILE = -0.85
    elif '_W5' in TEST_CONFIG:
        Z_CLIP_SURFACE_PROFILE = -0.05
    elif '_W11' in TEST_CONFIG:
        # Z_CLIP_SURFACE_PROFILE = -0.8
        # SHOULD DOUBLE-CHECK THIS
        # pass
        raise ValueError("Make sure you are aware of this before running.")

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
    SAVE_COMBINED_ZC = join(SAVE_COMBINED, 'zipped-coords_per_tid')
    SAVE_COMBINED_NET_D0ZR = join(SAVE_COMBINED, 'net-d0zr_per_pid_per_tid')
    XYM = 'g'  # 'g' or 'm': use sub-pixel or discrete in-plane localization method
    # filenames
    FN_IV_MATRIX = 'iv_acdc_test_matrix.xlsx'
    FN_IV_MATRIX_COMBINATIONS = 'iv_test_matrix_combinations_n={}.xlsx'
    FN_ALL_MERGED_COORDS_VOLT = 'all_merged-coords-volt.xlsx'
    FN_ALL_ZIPPED_COORDS = 'all_zipped_coords.xlsx'
    FN_ALL_NET_D0ZR_PER_PID = 'all_net-d0zr_per_pid.xlsx'
    FN_JOINED_MERGED_COORDS_VOLT_AND_IV_MATRIX = 'joined_merged-coords-volt_and_iv_matrix.xlsx'
    FN_JOINED_NET_D0ZR_AND_IV_MATRIX = 'joined_net-d0zr_and_iv_matrix.xlsx'
    # filepaths
    FP_IV_MATRIX = join(SAVE_COMBINED, FN_IV_MATRIX)
    FP_IV_MATRIX_COMBINATIONS = join(SAVE_COMBINED, FN_IV_MATRIX_COMBINATIONS)
    FP_ALL_MERGED_COORDS_VOLT = join(SAVE_COMBINED, FN_ALL_MERGED_COORDS_VOLT)
    FP_ALL_ZIPPED_COORDS = join(SAVE_COMBINED, FN_ALL_ZIPPED_COORDS)
    FP_ALL_NET_D0ZR_PER_PID = join(SAVE_COMBINED, FN_ALL_NET_D0ZR_PER_PID)

    # make dirs
    for pth in [SAVE_COMBINED, SAVE_COMBINED_ZC, SAVE_COMBINED_NET_D0ZR]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # initialize
    DICT_SETTINGS = None
    DF_IV_MATRIX = None
    COMBINATIONS = None
    DF_MCV = None
    DF_ZC = None
    DF_MCVIV = None
    DF_NET_D0ZR = None
    DF_MODEL_VDZ = None
    DF_MODEL_STRAIN = None
    ARR_MODEL_VDZ = None
    # -
    func_apparent_r_displacement = None
    # ---
    # ---
    # make iv test matrix
    if make_ivac_matrix:
        DF_IV_MATRIX = get_ivacdc_test_matrix(base_dir=BASE_DIR, save_dir=SAVE_COMBINED)
    # make iv test matrix combinations
    if make_ivac_matrix_combinations:
        COMBINATIONS = get_all_combinations_from_iv_matrix(
            df_iv_matrix=DF_IV_MATRIX,
            min_number_per_group=MIN_TIDS_PER_COMBINATION,
            base_dir=BASE_DIR,
            save_dir=SAVE_COMBINED,
        )
    # stack merged-coords-volt for all tids
    if make_merged_coords_volt:
        DF_MCV = make_all_merged_coords_volt(base_dir=BASE_DIR, save_dir=SAVE_COMBINED, return_df=True)
    # stack zipped_coords for all tids
    if make_zipped_coords:
        DF_ZC = make_all_zipped_coords(base_dir=BASE_DIR, save_dir=SAVE_COMBINED, return_df=True)
    # stack net-d0zr_per_pid for all tids
    if make_net_d0zr_per_pid:
        DF_NET_D0ZR = make_all_net_d0zr_per_pid(base_dir=BASE_DIR, xym=XYM, save_dir=SAVE_COMBINED, return_df=True)
    # join net-d0zr_per_pid and iv test matrix
    if join_net_d0zr_and_iv_matrix:
        DFDIV = get_joined_net_d0zr_and_iv_matrix(df_net_d0zr_per_pid=DF_NET_D0ZR, df_iv_matrix=DF_IV_MATRIX,
                                                  base_dir=BASE_DIR, xym=XYM, save_dir=SAVE_COMBINED)
    # read model data
    if read_model_data or plot_zipped_coords_on_model:
        DICT_SETTINGS = settings.get_settings( fp_settings=join(READ_SETTINGS, 'dict_settings.xlsx'), name='settings')
        if 'path_model' in DICT_SETTINGS.keys():
            mfiles = [x for x in os.listdir(DICT_SETTINGS['path_model'].strip("'")) if x.endswith('.xlsx')]
            mfile_z_by_v = [x for x in mfiles if x.endswith('z-by-v.xlsx')][0]
            mfile_strain_by_z = [x for x in mfiles if x.endswith('strain-by-z.xlsx')][0]

            DF_MODEL_VDZ = pd.read_excel(join(DICT_SETTINGS['path_model'], mfile_z_by_v))
            DF_MODEL_STRAIN = pd.read_excel(join(DICT_SETTINGS['path_model'], mfile_strain_by_z))
        elif 'path_model_dZ_by_V' or 'path_model_strain' in DICT_SETTINGS.keys():
            DF_MODEL_VDZ = pd.read_excel(DICT_SETTINGS['path_model_dZ_by_V'].strip("'"))
            DF_MODEL_STRAIN = pd.read_excel(DICT_SETTINGS['path_model_strain'].strip("'"))
        if DF_MODEL_VDZ is not None:
            model_V, model_dZ = plotting.pre_process_model_dZ_by_V_for_compare(
                dfm=DF_MODEL_VDZ, mkey=DICT_SETTINGS['model_mkey'], mval=DICT_SETTINGS['model_mval'], extend_max_voltage=VMAX)
            ARR_MODEL_VDZ = (model_V, model_dZ)
    # read surface profile data
    if plot_zipped_coords_on_model:
        if DICT_SETTINGS is None:
            DICT_SETTINGS = settings.get_settings( fp_settings=join(READ_SETTINGS, 'dict_settings.xlsx'), name='settings')
        dict_surface_profilometry = get_surface_profile_dict(DICT_SETTINGS)
        surf_r, surf_z = dict_surface_profilometry['r'], dict_surface_profilometry['z']

        if '_W11' in TEST_CONFIG:
            surf_z = surf_z[surf_r < 620]
            surf_r = surf_r[surf_r < 620]
            raise ValueError("Make sure you are aware of this before running.")

        func_apparent_r_displacement = calculate_apparent_radial_displacement_due_to_rotation(
            surf_r=surf_r,
            surf_z=surf_z,
            poly_deg=POLY_DEG_CORRECT_RADIAL_DISPLACEMENT,
            membrane_thickness=DICT_SETTINGS['membrane_thickness'],
            z_clip=Z_CLIP_SURFACE_PROFILE,
            path_save=SAVE_COMBINED_ZC,
        )

        raise ValueError("The above tilt-correction does not take the actual thickness of the membrane into account."
                         "This can be easily done by also fitting a function to the membrane thickness vs. depth "
                         "data which is generated from the energy minimization model."
                         "NOTE: not performing this correction leads to a relatively significant error."
                         "For example, the thickness of C7-20pT after pre-stretch is ~14 microns. So, "
                         "using a membrane thickness of 20 microns for the tilt-correction would result in"
                         "an error of 30%."
                         "Some specific examples of this error that I can already see in the data:"
                         "  * W12 C7-20pT: fixing this error would result in much much better agreement."
                         "  * For the other C7-20pT tests, fixing the error might actually make the agreement worse.")

    # ---

    # --- plot using net-d0z dataset

    if (plot_all_pids_net_d0zr_per_pid_by_tid or plot_heatmap_of_all_pids_net_d0zr or
            plot_per_pid_net_d0zr_per_pid_by_tid or plot_net_d0zr_frequency_sweeps_per_pid):
        DFDIV = get_joined_net_d0zr_and_iv_matrix(
            df_net_d0zr_per_pid=DF_NET_D0ZR,
            df_iv_matrix=DF_IV_MATRIX,
            base_dir=BASE_DIR,
            xym=XYM,
            save_dir=SAVE_COMBINED,
        )

        if plot_net_d0zr_frequency_sweeps_per_pid and len(DFDIV['awg_freq'].unique()) > 2:
            SAVE_SUB_ANALYSIS = join(SAVE_COMBINED_NET_D0ZR, 'frequency_sweeps')
            if not os.path.exists(SAVE_SUB_ANALYSIS):
                os.makedirs(SAVE_SUB_ANALYSIS)

            # DFDIV = DFDIV[DFDIV['awg_freq'] < 1000]

            if COMBINATIONS is None:
                COMBINATIONS = get_all_combinations_from_iv_matrix(
                    df_iv_matrix=DF_IV_MATRIX,
                    min_number_per_group=MIN_TIDS_PER_COMBINATION,
                    base_dir=BASE_DIR,
                    save_dir=SAVE_COMBINED,
                )
            if COMBINATIONS is not None:
                plotting.plot_net_d0zr_frequency_sweeps_per_pid(
                    df=DFDIV,
                    combinations=COMBINATIONS,
                    only_pids=ONLY_PIDS,  # None: defer to dz quantile threshold
                    threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
                    path_save=SAVE_SUB_ANALYSIS,
                    poly_deg=FREQ_SWEEP_POLY_DEG,
                )

        # ---

        if plot_all_pids_net_d0zr_per_pid_by_tid:
            plotting.plot_net_d0zr_all_pids_by_volt_freq(
                df=DFDIV,
                path_save=SAVE_COMBINED_NET_D0ZR,
                threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
                only_test_types=ONLY_TEST_TYPES,
                arr_model_VdZ=ARR_MODEL_VDZ,
            )

            for shift_V_by_X_log_freq in [0, 2, 4]:
                plotting.plot_net_d0zr_all_pids_by_volt_freq_errorbars_per_tid(
                    df=DFDIV,
                    path_save=SAVE_COMBINED_NET_D0ZR,
                    threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
                    only_test_types=ONLY_TEST_TYPES,
                    shift_V_by_X_log_freq=shift_V_by_X_log_freq,
                    arr_model_VdZ=ARR_MODEL_VDZ,
                )

        if plot_heatmap_of_all_pids_net_d0zr:
            plotting.plot_heatmap_net_d0zr_all_pids_by_volt_freq(
                df=DFDIV,
                path_save=SAVE_COMBINED_NET_D0ZR,
                threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
                only_test_types=ONLY_TEST_TYPES,
            )

        if plot_per_pid_net_d0zr_per_pid_by_tid:
            SAVE_SUB_ANALYSIS = join(SAVE_COMBINED_NET_D0ZR, 'net-d0zr_per_pid_by_volt-freq')
            if not os.path.exists(SAVE_SUB_ANALYSIS):
                os.makedirs(SAVE_SUB_ANALYSIS)
            plotting.plot_net_d0zr_per_pid_by_volt_freq(
                df=DFDIV,
                path_save=SAVE_SUB_ANALYSIS,
                only_test_types=ONLY_TEST_TYPES,
                arr_model_VdZ=ARR_MODEL_VDZ,
            )


    # --- plot using merged-coords-volts dataset

    # the below reads data for all tids (so it's very slow)
    if (plot_merged_coords_volt_parametric_sweeps_per_pid_by_tid or
            plot_merged_coords_volt_per_pid_by_all_volt_freq or
            plot_merged_coords_volt_heat_maps):
        DF_MCVIV = get_joined_merged_coords_volt_and_iv_matrix(
            df_merged_coords_volt=DF_MCV,
            df_iv_matrix=DF_IV_MATRIX,
            base_dir=BASE_DIR,
            save_dir=SAVE_COMBINED,
            df_mcviv=DF_MCVIV,
        )
        # DF_MCVIV = DF_MCVIV[DF_MCVIV['acdc'] == 'AC']

        if plot_merged_coords_volt_parametric_sweeps_per_pid_by_tid:
            SAVE_SUB_ANALYSIS = join(SAVE_COMBINED_MCV, 'parametric_sweeps')
            if not os.path.exists(SAVE_SUB_ANALYSIS):
                os.makedirs(SAVE_SUB_ANALYSIS)

            if COMBINATIONS is None:
                COMBINATIONS = get_all_combinations_from_iv_matrix(
                    df_iv_matrix=DF_IV_MATRIX,
                    min_number_per_group=MIN_TIDS_PER_COMBINATION,
                    base_dir=BASE_DIR,
                    save_dir=SAVE_COMBINED,
                )
            if COMBINATIONS is not None:
                plotting.plot_parametric_sweeps_per_pid_trajectory_by_tid(
                    df_mcviv=DF_MCVIV,
                    combinations=COMBINATIONS,
                    only_pids=ONLY_PIDS,  # None: defer to dz quantile threshold
                    threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
                    path_save=SAVE_SUB_ANALYSIS,
                )

        if plot_merged_coords_volt_per_pid_by_all_volt_freq:
            SAVE_SUB_ANALYSIS = join(SAVE_COMBINED_MCV, 'per_pid_by_all-volt-freq')
            if not os.path.exists(SAVE_SUB_ANALYSIS):
                os.makedirs(SAVE_SUB_ANALYSIS)
            # DF_MCVIV = DF_MCVIV[DF_MCVIV['tid'].isin([1, 3])]
            plotting.plot_merged_coords_volt_per_pid_by_volt_freq(
                df=DF_MCVIV,
                path_save=SAVE_SUB_ANALYSIS,
                threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
                only_pids=ONLY_PIDS,  # None: defer to dz quantile threshold
                only_test_types=ONLY_TEST_TYPES,
                arr_model_VdZ=ARR_MODEL_VDZ,
            )

        if plot_merged_coords_volt_heat_maps:
            plotting.plot_heatmap_merged_coords_volt_all_pids_by_volt_freq(
                df=DF_MCVIV,
                path_save=SAVE_COMBINED_MCV,
                threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
                only_test_types=ONLY_TEST_TYPES,
                save_id='ascending+descending',
            )

    # -

    if plot_merged_coords_volt_ascending_only:
        SAVE_ASC_ANALYSIS = join(SAVE_COMBINED_MCV, 'ascending-only')
        SAVE_SUB_ANALYSIS = join(SAVE_ASC_ANALYSIS, 'per_pid_by_all-volt-freq')
        for pth in [SAVE_ASC_ANALYSIS, SAVE_SUB_ANALYSIS]:
            if not os.path.exists(pth):
                os.makedirs(pth)

        filepath_ascending = join(SAVE_COMBINED, 'joined_merged-coords-volt_and_iv_matrix_ascending-only.xlsx')
        if os.path.exists(filepath_ascending):
            df_ascending = pd.read_excel(filepath_ascending)
        else:
            df = get_joined_merged_coords_volt_and_iv_matrix(
                df_merged_coords_volt=DF_MCV,
                df_iv_matrix=DF_IV_MATRIX,
                base_dir=BASE_DIR,
                save_dir=SAVE_COMBINED,
                df_mcviv=DF_MCVIV,
            )

            tids = df.tid.unique()
            df_ascending = []
            for tid in tids:
                df_tid = df[df['tid'] == tid].reset_index(drop=True)
                df_tid = df_tid[df_tid['STEP'] <= df_tid['STEP'].iloc[df_tid['VOLT'].abs().idxmax()]]
                df_ascending.append(df_tid)
            df_ascending = pd.concat(df_ascending)
            df_ascending.to_excel(join(SAVE_COMBINED, 'joined_merged-coords-volt_and_iv_matrix_ascending-only.xlsx'))

        plotting.plot_heatmap_merged_coords_volt_all_pids_by_volt_freq(
            df=df_ascending,
            path_save=SAVE_ASC_ANALYSIS,
            threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
            only_test_types=ONLY_TEST_TYPES,
            save_id='ascending-only',
        )

        # DF_MCVIV = DF_MCVIV[DF_MCVIV['tid'].isin([1, 3])]
        plotting.plot_merged_coords_volt_per_pid_by_volt_freq(
            df=df_ascending,
            path_save=SAVE_SUB_ANALYSIS,
            threshold_pids_by_d0z=THRESHOLD_PIDS_BY_D0Z,
            only_pids=ONLY_PIDS,  # None: defer to dz quantile threshold
            only_test_types=ONLY_TEST_TYPES,
            arr_model_VdZ=ARR_MODEL_VDZ,
        )

    # -

    # ---

    # --- plot using zipped_coords dataset

    if plot_zipped_coords_on_model:
        DF_ZC = get_all_zipped_coords(
            df_zc=DF_ZC,
            base_dir=BASE_DIR,
            save_dir=SAVE_COMBINED,
        )

        # DF_ZC = DF_ZC[(DF_ZC['frame'] < 75) | (DF_ZC['frame'] > 370)]
        # DF_ZC = DF_ZC[DF_ZC['drg'] < 10]

        plotting.compare_corrected_zipped_stretch_with_model(
            df=DF_ZC,
            dfm=DF_MODEL_STRAIN,
            correction_function=func_apparent_r_displacement,
            pr='rg',
            pdr='drg',
            pdz='d0z',
            path_results=SAVE_COMBINED_ZC,
            save_id='poly-deg={}_zipped-df_corr-dr'.format(POLY_DEG_CORRECT_RADIAL_DISPLACEMENT),
            poly_deg_id=POLY_DEG_CORRECT_RADIAL_DISPLACEMENT,
        )


    # the below only reads data for specified tids (so it's
    # computationally much faster but requires manual input)
    # use this to create "custom" plots for specific comparisons
    plot_overlay_pid_by_tid = False
    if plot_overlay_pid_by_tid:
        SAVE_SUB_ANALYSIS = join(SAVE_COMBINED_MCV, 'pid_dz_by_tid')
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
