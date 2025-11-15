import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import smu, awg, dpt
from utils import settings, analyses, plotting, empirical


def get_zipping_interface_up_to_pull_in(df, include_extra_frames=4, within_percent_max_dz=2):
    arr1 = df['frame'].to_numpy()
    arr2 = df['zipping_interface_z'].to_numpy()
    # primary sort: r
    sorted_indices = np.argsort(arr1)
    # Use these indices to sort both arrays
    sorted1 = arr1[sorted_indices]
    sorted2 = arr2[sorted_indices]
    pull_in_earliest_frame = sorted1[np.argmax(np.abs(sorted2 - np.min(arr2)) < np.abs(np.min(arr2)) * within_percent_max_dz / 100)]
    # -
    df = df[df['frame'] <= pull_in_earliest_frame + include_extra_frames]
    return df


def calculate_zipping_interface(df, przdr, dict_surf, acdc, frames=None):
    if acdc == 'DC':
        keep_cols = ['frame', 't_sync', 'SOURCE_TIME_MIDPOINT', 'STEP', 'VOLT', 'CURR', 'TIMESTAMP']
    elif acdc == 'AC':
        keep_cols = ['frame', 't_sync', 'SOURCE_TIME_MIDPOINT', 'STEP', 'VOLT',
                     'TIMESTAMP_DAQ_HANDLER', 'MONITOR_TIME_MIDPOINT', 'MONITOR_VALUE']
    else:
        raise ValueError("Invalid acdc: {}".format(acdc))
    # initialize variables
    pr, pz, pdr = przdr
    surf_r, surf_z = dict_surf['r'] + dict_surf['dr'], dict_surf['z'] + dict_surf['dz']
    # -
    if frames is None:
        frames = df['frame'].unique()
        frames.sort()
    # -
    df_frames = []
    df_zipping_interface = []
    for frame in frames:
        df_frame = df[df['frame'] == frame].sort_values(pr)

        zipping_interface_r, zipping_interface_z = empirical.get_zipping_interface_rz(
            r=df_frame[pr].to_numpy(),
            z=df_frame[pz].to_numpy(),
            surf_r=surf_r,
            surf_z=surf_z,
        )

        # if you want to plot to double-check algorithm
        """
        if frame > 100:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(df_frame[pr], df_frame[pz], c='k', s=7)
            ax.plot(surf_r, surf_z, '-', color='gray', lw=2, label='SP', zorder=3.1)
            ax.axhline(zipping_interface_z, 0, 1, ls='--', color='gray', lw=0.95, alpha=0.25, zorder=3.0)
            ax.axvline(zipping_interface_r, 0, 1, ls='--', color='gray', lw=0.95, alpha=0.25, zorder=3.0)
            ax.set_title('Frame: {}'.format(frame))
            plt.show()
            plt.close()
            a = 1
        """

        # pure position of the zipping interface, with additional details useful for plotting
        df_zip = df_frame[keep_cols].groupby('frame').mean().reset_index()
        df_zip['zipping_interface_r'] = zipping_interface_r
        df_zip['zipping_interface_z'] = zipping_interface_z
        df_zipping_interface.append(df_zip)

        # particles that are on the sidewall, above and outside of the zipping interface
        df_frame = df_frame[(df_frame[pr] > zipping_interface_r) & (df_frame[pz] > zipping_interface_z)]
        df_frame['zipping_interface_r'] = zipping_interface_r
        df_frame['zipping_interface_z'] = zipping_interface_z
        df_frames.append(df_frame)

    df_frames = pd.concat(df_frames)
    df_zipping_interface = pd.concat(df_zipping_interface)
    return df_frames, df_zipping_interface

# ---

if __name__ == "__main__":

    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    TEST_CONFIG = '03052025_W13-D1_C19-30pT_20+10nmAux'
    TIDS = [0]  # [56, 62, 63] or np.arange(30, 70) or np.flip(np.arange(30, 70))
    ONLY_TIDS_WITH_TEST_SETTINGS = True
    IV_ACDC = 'AC'  # 'AC' or 'DC'

    # ---

    ANIMATE_FRAMES = np.arange(20, 190)  # None: defer to test_settings; to override test_settings: np.arange(20, 115)
    ANIMATE_RZDR = ('rg', 'd0z', 'drg')  # None = ('rg', 'd0z', 'drg'); or override: e.g., ('rg', 'dz', 'drg'). For cross-section plots and zipped coords
    # -
    # SETTINGS (True False)
    UPDATE_DEPENDENT = False  # True: update all dependent settings in dict_settings.xlsx
    PLOT_SETTINGS_IMAGE_OVERLAY = False  # only need to run once per test configuration
    # -
    # PRE-PROCESSING (True False)
    PRE_PROCESS = False
    # -
    # ANALYSES
    XYM = ['g']  # ['g', 'm']: use sub-pixel or discrete in-plane localization
    SECOND_PASS = False  # True False
    AVG_MAX_N = -1  # Gets used by both EXPORT_NET_D0ZR and SECOND_PASS
    EXPORT_NET_D0ZR = False  # True: export dfd0 to special directory
    EXPORT_ZIPPED_COORDS = False
    EXPORT_ZIPPING_INTERFACE = False
    PLOT_ZIPPING_INTERFACE, AND_SAVE = True, True  # Only plots if EXPORT_ZIPPING_INTERFACE is True
    ZIPPED_ONLY_FRAMES = None
    # -
    # ALTERNATIVE IS TO USE INITIAL COORDS
    EXPORT_INITIAL_COORDS = False  # False True
    D0F_IS_TID = 0  # ONLY USED IF DICT_TID{}_SETTINGS.XLSX IS NOT FOUND
    DROP_PIDS = []  # []: remove bad particles from ALL coords
    # -
    # ONLY USED IF DICT_TID{}_SETTINGS.XLSX IS NOT FOUND **AND** IV_ACDC == 'DC'
    START_FRAME, END_FRAMES = (0, 0), (0, 0)  # (a<x<b; NOT: a<=x<=b) only used if test_settings.xlsx not found


    # ------------------------------------------------------------------------------------------------------------------
    # YOU SHOULD NOT NEED TO CHANGE BELOW
    # ------------------------------------------------------------------------------------------------------------------

    for TID in TIDS:
        print("TID: {}".format(TID))
        # ---
        # --- RARELY USED SETTINGS
        USE_GENERIC_DC_TEST_SETTINGS = False  # in rare cases, do not parse FN_IV filename and instead use generic settings
        # ---
        PRE_PROCESS_COORDS = PRE_PROCESS  # If you change andor_keithley_delay time, you must pre-process coords.
        PRE_PROCESS_IV = PRE_PROCESS  # Only needs to be run once per-tid; not dependent on synchronization timing settings.
        MERGE_COORDS_AND_VOLTAGE = PRE_PROCESS
        # FILEPATHS
        # ---
        # directories
        ROOT_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024'
        BASE_DIR = join(ROOT_DIR, 'zipper_paper/Testing/Zipper Actuation', TEST_CONFIG)
        SAVE_DIR = join(BASE_DIR, 'analyses')
        SAVE_SETTINGS = join(SAVE_DIR, 'settings')
        SAVE_COORDS = join(SAVE_DIR, 'coords')
        # SAVE_COORDS_W_PIXELS = join(SAVE_COORDS, 'pixels')
        PATH_REPRESENTATIVE = join(SAVE_DIR, 'representative_test{}')
        # PATH_REPRESENTATIVE = join(SAVE_DIR, 'net-d0zr_per_pid')

        # -
        # settings
        FP_SETTINGS = join(SAVE_SETTINGS, 'dict_settings.xlsx')
        FP_TEST_SETTINGS = join(SAVE_SETTINGS, 'dict_tid{}_settings.xlsx'.format(TID))
        # -
        # 3D particle tracking
        READ_COORDS_DIR = join(BASE_DIR, 'results', 'test-idpt_test-{}'.format(TID))
        FN_COORDS_STARTS_WITH = 'test_coords_t'
        FN_COORDS_SAVE = 'tid{}_coords.xlsx'.format(TID)
        FN_COORDS_SAVE_ZIPPED = 'tid{}_zipped_coords.xlsx'.format(TID)
        FN_SAVE_ZIPPING_INTERFACE = 'tid{}_zipping_interface.xlsx'.format(TID)
        FN_COORDS_INITIAL_SAVE = 'tid{}_init_coords.xlsx'.format(TID)
        # FN_COORDS_INITIAL_READ = 'tid{}_init_coords.xlsx'.format(D0F_IS_TID)
        # -
        # Keithley and/or Agilent
        READ_IV_DIR = join(BASE_DIR, 'I-V', 'xlsx')
        FN_IV_STARTS_WITH = 'tid{}_'.format(TID)
        FN_IVDC_ENDS_WITH = 'dV.xlsx'  # if I-V file has this ending, then it is a DC test
        FN_IVAC_ENDS_WITH = '_data.xlsx'  # if I-V file has this ending, then it is an AC test
        FN_IV_SAVE = 'tid{}_I-V.xlsx'.format(TID)
        FN_IV_AVG_SAVE = 'tid{}_average-V-t.xlsx'.format(TID)
        # Merged (3DPT + Keithley)
        FN_MERGED = 'tid{}_merged-coords-volt.xlsx'.format(TID)
        # -
        # Conditional
        if ONLY_TIDS_WITH_TEST_SETTINGS:
            if not os.path.exists(FP_TEST_SETTINGS):
                print("Skipping TID{} because test settings were not found.".format(TID))
                continue
        # -
        # Initialize
        DF_ZIPPED = None
        # -
        # make dirs
        for pth in [SAVE_DIR, SAVE_SETTINGS, SAVE_COORDS]:
            if not os.path.exists(pth):
                os.makedirs(pth)

        SETTINGS_HANDLER_DICT = {
            'fp_settings': FP_SETTINGS,
            'update_dependent': UPDATE_DEPENDENT,
            'fp_test_settings': FP_TEST_SETTINGS,
            'iv_acdc': IV_ACDC,
            'use_generic_dc_test_settings': USE_GENERIC_DC_TEST_SETTINGS,  # 'pre_process_iv': PRE_PROCESS_IV,
            'read_iv_dir': READ_IV_DIR,
            'fn_iv_startswith': FN_IV_STARTS_WITH,
            'fn_ivdc_endswith': FN_IVDC_ENDS_WITH,
            'fn_ivac_endswith': FN_IVAC_ENDS_WITH,
            'drop_pids': DROP_PIDS,
            'start_frame': START_FRAME,
            'end_frames': END_FRAMES,
            'd0f_is_tid': D0F_IS_TID,
        }
        # -
        # --------------------------------------------------------------------------------------------------------------
        # ---
        # SETTINGS
        # ---
        DICT_SETTINGS, DICT_TEST = settings.settings_handler(settings_handler_dict=SETTINGS_HANDLER_DICT)

        # ---
        """
        if not os.path.exists(join(SAVE_COORDS, FN_MERGED)):
            PRE_PROCESS_COORDS = True  # If you change andor_keithley_delay time, you must pre-process coords.
            PRE_PROCESS_IV = True  # Only needs to be run once per-tid; not dependent on synchronization timing settings.
            MERGE_COORDS_AND_VOLTAGE = True
        else:
            PRE_PROCESS_COORDS = False  # If you change andor_keithley_delay time, you must pre-process coords.
            PRE_PROCESS_IV = False  # Only needs to be run once per-tid; not dependent on synchronization timing settings.
            MERGE_COORDS_AND_VOLTAGE = False
        """
        # PRE-PROCESSING
        # ---
        if PRE_PROCESS_COORDS or PRE_PROCESS_IV or MERGE_COORDS_AND_VOLTAGE:
            # --- 3D particle tracking data
            if PRE_PROCESS_COORDS:
                # Find and read coords
                FN_COORDS = [x for x in os.listdir(READ_COORDS_DIR) if x.startswith(FN_COORDS_STARTS_WITH)][0]
                DF = pd.read_excel(join(READ_COORDS_DIR, FN_COORDS))
                # drop bad pids
                if 'drop_pids' in DICT_TEST.keys():
                    if len(DICT_TEST['drop_pids']) > 0:
                        DROP_PIDS.extend(DICT_TEST['drop_pids'])
                if len(DROP_PIDS) > 0:
                    DF = DF[~DF['id'].isin(DROP_PIDS)]
                print("Dropping pids: {}".format(DROP_PIDS))
                # pre-process
                DF = dpt.pre_process_coords(df=DF, settings=DICT_SETTINGS, acdc=IV_ACDC)
                # get/calculate "initial coords", so that "relative coords" can be determined
                if EXPORT_INITIAL_COORDS:
                    D0F = dpt.calculate_initial_coords(df=DF, frames=DICT_TEST['dpt_start_frame'])
                    D0F.to_excel(join(SAVE_COORDS, FN_COORDS_INITIAL_SAVE), index=True)
                    raise ValueError("Always stop after exporting initial coords.")
                else:
                    # get "pre-test" (i.e., before any testing) coords
                    FN_COORDS_INITIAL_READ = 'tid{}_init_coords.xlsx'.format(DICT_TEST['d0f_is_tid'])
                    D0F = pd.read_excel(join(SAVE_COORDS, FN_COORDS_INITIAL_READ), index_col=0)
                    # calculate relative coords (relative to pre-start (i.e., frame = 0)
                    # AND relative to pre-tests (i.e., before any voltage was applied).
                    DF = dpt.calculate_relative_coords(df=DF, d0f=D0F, test_settings=DICT_TEST)
                    # calculate rolling min (window size = images per voltage level): 'dz_lock_in'
                    DF = dpt.calculate_lock_in_rolling_min(
                        df=DF, pcols=['dz', 'd0z'], settings=DICT_SETTINGS, test_settings=DICT_TEST)

                # export transformed dataframe
                # for reference: physical + pixel coordinates
                # DF.to_excel(join(SAVE_COORDS_W_PIXELS, FN_COORDS_SAVE), index=True)
                # for analysis: we need only keep physical coordinates (and frames) for compactness
                PHYS_COLS = ['frame', 't_sync', 'dt', 'id', 'cm',
                             'z', 'd0z', 'dz', 'd0z_lock_in', 'dz_lock_in',
                             'd0rg', 'drg', 'd0rm', 'drm', 'rg', 'rm',
                             'd0xg', 'dxg', 'd0xm', 'dxm', 'xg', 'xm',
                             'd0yg', 'dyg', 'd0ym', 'dym', 'yg', 'ym',
                             ]
                DF = DF[PHYS_COLS]
                DF.to_excel(join(SAVE_COORDS, FN_COORDS_SAVE), index=False)
            else:
                DF = pd.read_excel(join(SAVE_COORDS, FN_COORDS_SAVE))
            # --- Keithley source voltage data
            if PRE_PROCESS_IV:
                if IV_ACDC == 'DC':
                    # Find and read coords
                    DFV = pd.read_excel(
                        join(READ_IV_DIR, DICT_TEST['filename']),
                        index_col=0,
                        names=['VOLT', 'CURR', 'TIMESTAMP'],
                    )
                    DFV = smu.pre_process_keithley(
                        df=DFV,
                        delay=DICT_TEST['smu_source_delay_time'],
                        integration=DICT_SETTINGS['integration_time'],
                    )
                    DFV.to_excel(join(SAVE_COORDS, FN_IV_SAVE), index=False)
                    # Average voltage by time
                    #DFVG = smu.average_voltage_by_source_time(df=DFV)
                    #DFVG.to_excel(join(SAVE_COORDS, FN_IV_AVG_SAVE), index=False)
                    # Additional I-V monitoring data
                    DFVM = None
                elif IV_ACDC == 'AC':
                    # Find and read coords
                    DFV = pd.read_excel(
                        join(READ_IV_DIR, DICT_TEST['filename']),
                        # index_col=0,
                        names=['awg_volt', 'dt'],
                    )
                    DFV = awg.pre_process_awg(
                        df=DFV,
                        amplifier_gain=DICT_TEST['amplifier_gain'],
                    )
                    # DFV.to_excel(join(SAVE_COORDS, FN_IV_SAVE), index=False)
                    # -
                    # Average voltage by time
                    #DFVG = smu.average_voltage_by_source_time(df=DFV)
                    #DFVG.to_excel(join(SAVE_COORDS, FN_IV_AVG_SAVE), index=False)
                    # -
                    # Additional I-V monitoring data
                    DFVM = pd.read_excel(
                        join(READ_IV_DIR, DICT_TEST['filename']),
                        sheet_name='data_output',
                        names=['READ', 'TST', 'READ_ZCOR', 'MEAS_ZCOR'],
                    )
                    DFVM = DFVM.rename(columns={'TST': 'MONITOR_TIMESTAMP', 'MEAS_ZCOR': 'MONITOR_VALUE'})
                    DFVM['MONITOR_TIME_SHIFTED'] = DFVM['MONITOR_TIMESTAMP'] + DICT_SETTINGS['monitor_delay_to_awg_input']

                    # TODO: write a function to programmatically determine how to scale monitor time to match awg input time.
                    def scale_monitor_time_to_match_awg_input_time(dfvm, dict_settings, dict_test, filepath_test_settings=None):
                        scale_factor_estimate = dict_settings['monitor_time_scale_factor_est']

                        # function to determine best scale factor
                        scale_factor_found = scale_factor_estimate  # THIS IS A PLACEHOLDER

                        # scale monitor time
                        dfvm['MONITOR_TIME'] = dfvm['MONITOR_TIME_SHIFTED'] * scale_factor_found

                        # update dict_test and export
                        dict_test.update({'monitor_time_scale_factor': scale_factor_found})
                        if filepath_test_settings is not None:
                            settings.write_settings_to_df(
                                s=dict_test,
                                filepath_save=filepath_test_settings,
                            )

                        return dfvm, dict_test

                    DFVM, DICT_TEST = scale_monitor_time_to_match_awg_input_time(
                        dfvm=DFVM,
                        dict_settings=DICT_SETTINGS,
                        dict_test=DICT_TEST,
                        filepath_test_settings=SETTINGS_HANDLER_DICT['fp_test_settings'],
                    )

                    DFVM = DFVM[['MONITOR_TIME', 'MONITOR_VALUE']]

                    # -
                    # Use ExcelWriter to write to multiple sheets
                    with pd.ExcelWriter(join(SAVE_COORDS, FN_IV_SAVE)) as writer:
                        for sheet_name, df in zip(['data_input', 'data_output'], [DFV, DFVM]):
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                DFV = pd.read_excel(join(SAVE_COORDS, FN_IV_SAVE))
                try:
                    DFVM = pd.read_excel(join(SAVE_COORDS, FN_IV_SAVE), sheet_name='data_output')
                except ValueError:
                    DFVM = None
            # -
            # --- Merge (3DPT + Keithley)
            if MERGE_COORDS_AND_VOLTAGE:
                DF = dpt.merge_with_iv(
                    df=DF,
                    dfv=DFV,
                    dfvm=DFVM,
                )
                DF.to_excel(join(SAVE_COORDS, FN_MERGED), index=False)
        else:
            # KEITHLEY COLUMNS: 'SOURCE_TIME_MIDPOINT', 'STEP', 'VOLT', 'CURR', 'TIMESTAMP'
            DF = pd.read_excel(join(SAVE_COORDS, FN_MERGED))
        # --------------------------------------------------------------------------------------------------------------
        # ---
        # FIRST-PASS EVALUATION
        # ---
        """
        Verify settings 
        """
        if PLOT_SETTINGS_IMAGE_OVERLAY:
            plotting.plot_surface_profilometry(
                dict_settings=DICT_SETTINGS,
                savepath=join(SAVE_SETTINGS, 'surface_profile.png'),
            )


            plotting.plot_scatter_with_pid_labels(
                df=DF[DF['frame'] == DF['frame'].iloc[0]],
                pxy=('xg', 'yg'),
                dict_settings=DICT_SETTINGS,
                savepath=join(SAVE_SETTINGS, 'pid-labels_on_features.png'),
            )

            plotting.plot_scatter_on_image(
                df=DF[DF['frame'] == DF['frame'].iloc[0]],
                pxy=('xg', 'yg'),
                dict_settings=DICT_SETTINGS,
                savepath=join(SAVE_SETTINGS, 'pids-features_on_image.png'),
            )

        # --------------------------------------------------------------------------------------------------------------
        # ---
        # EXPORT NET DISPLACEMENT TO SPECIAL DIRECTORY
        if EXPORT_NET_D0ZR:
            for xym in XYM:
                analyses.export_net_d0zr_per_pid_per_tid(
                    df=DF,
                    xym=xym,
                    tid=TID,
                    dict_settings=DICT_SETTINGS,
                    dict_test=DICT_TEST,
                    path_results=SAVE_DIR,
                    average_max_n_positions=AVG_MAX_N,
                )

        # ---
        # EXPORT COORDINATES OF ONLY "ZIPPED" POSITIONS"
        if EXPORT_ZIPPED_COORDS or EXPORT_ZIPPING_INTERFACE or PLOT_ZIPPING_INTERFACE:
            DF_ZIPPED_COORDS, DF_ZIP_INTERFACE = calculate_zipping_interface(
                df=DF,
                przdr=ANIMATE_RZDR,
                dict_surf=analyses.get_surface_profile_dict(DICT_SETTINGS),
                acdc=IV_ACDC,
                frames=ZIPPED_ONLY_FRAMES,
            )
            if EXPORT_ZIPPED_COORDS and len(DF_ZIPPED_COORDS) > 0:
                DF_ZIPPED_COORDS.to_excel(join(SAVE_COORDS, FN_COORDS_SAVE_ZIPPED), index=False)
            if EXPORT_ZIPPING_INTERFACE and len(DF_ZIP_INTERFACE) > 0:
                DF_ZIP_INTERFACE.to_excel(join(SAVE_COORDS, FN_SAVE_ZIPPING_INTERFACE), index=False)
            if PLOT_ZIPPING_INTERFACE:
                df = DF_ZIP_INTERFACE
                include_extra_frames = 4
                within_percent_max_dz = 2
                # ---
                df_pull_in = get_zipping_interface_up_to_pull_in(df, include_extra_frames, within_percent_max_dz)
                dfg_mean = df_pull_in.groupby('VOLT').mean().reset_index()
                dfg_min = df_pull_in.groupby('VOLT').min().reset_index()

                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(3.75, 3.75))
                ax1.plot(df['VOLT'], df['zipping_interface_z'], '--', lw=0.75, color='gray', alpha=0.5)
                ax1.plot(df['VOLT'], df['zipping_interface_z'], 'k.', ms=1, label='Zipping Interface')
                ax1.set_ylabel('{} '.format(ANIMATE_RZDR[1]) + r'$(\mu m)$')
                ax1.grid(alpha=0.125)
                ax1.legend(fontsize='x-small')
                ax1.set_title(f'{TEST_CONFIG}: TID{TID}', fontsize='small')

                ax2.plot(df_pull_in['VOLT'], df_pull_in['zipping_interface_z'], 'k.', ms=1, alpha=0.35, label=f'pull-in+{include_extra_frames}frames')
                ax2.plot(dfg_mean['VOLT'], dfg_mean['zipping_interface_z'], '-', lw=0.6, color='tab:blue', alpha=0.75, label='mean(volt)')
                ax2.plot(dfg_min['VOLT'], dfg_min['zipping_interface_z'], '-', lw=0.6, color='tab:red', alpha=0.75, label='min(volt)')
                ax2.set_ylabel(r'$z_{interface, avg}$')
                ax2.grid(alpha=0.125)
                ax2.legend(fontsize='xx-small')
                ax2.set_xlabel('V (V)')
                plt.tight_layout()
                if AND_SAVE:
                    if not os.path.exists(PATH_REPRESENTATIVE.format(TID)):
                        os.makedirs(PATH_REPRESENTATIVE.format(TID))
                    try:
                        fig.savefig(join(PATH_REPRESENTATIVE.format(TID), 'tid{}_zipping_interface.png'.format(TID)),
                                    dpi=300, bbox_inches='tight', facecolor='w')
                    except FileNotFoundError:
                        print("Directory not found: {}".format(
                            join(PATH_REPRESENTATIVE.format(TID), 'tid{}_zipping_interface.png'.format(TID))))
                plt.show()
                plt.close(fig)

        # ---
        # SECOND-PASS EVALUATION
        # ---
        """
        Plot particle z-r displacement vs. (frame, time, and voltage)
        """
        if SECOND_PASS:
            for xym in XYM:
                analyses.second_pass(
                    df=DF,
                    xym=xym,
                    tid=TID,
                    dict_settings=DICT_SETTINGS,
                    dict_test=DICT_TEST,
                    path_results=PATH_REPRESENTATIVE,
                    animate_frames=ANIMATE_FRAMES,
                    animate_rzdr=ANIMATE_RZDR,
                    df_zipped=DF_ZIPPED,
                    average_max_n=AVG_MAX_N,
                )

    print("Completed without errors.")