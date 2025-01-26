import os
from os.path import join
import pandas as pd

import smu, dpt
from utils import settings, analyses, plotting, empirical


if __name__ == "__main__":
    """
    NOTES:
        1. requires: analyses/settings/dict_settings.xlsx
        2. if analyses/settings/dict_tid{}_settings.xlsx is not found, must enter: START_FRAME, END_FRAMES
    """

    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    TEST_CONFIG = '01102025_W13-D1_C9-0pT'
    TID = 1

    """
    NOTES:
        1. It would be best to really nail down my data analysis procedure BEFORE continuing (testing, interpretation, etc).
            The dataset [01092025_W10-A1_C9-0pT] is perfect for nailing down my procedure because it contains all interesting phenomena.
            (i.e., all phenomena that I expect to see in later experiments)
            
            This would require:
                1. Determining the pre-test state of the membrane (i.e., was it flat?)
                    * Approach #1:
                        1. use SPCT/GDPT to determine depth position of particles in individual flatness images,
                        2. use cross-correlation to "stitch the flatness images together" (i.e., determine x-y offset),
                        3. use the x-y offset between the images to shift the 3D particle positions in each image (i.e., stitch them together)
                        4. fit a plane and bivariate spline to the stitched 3D positions to evaluate flatness and curvature. 
                    * Approach #2:
                        1. determine depth position of particles in calibration images by determining z of peak intensity
                        2. evaluate if the results 3D distribution reveals anything
                2 Determining validity or (xg, yg) vs. (xm, ym)
                
            Other analyses:
                1. Identifying the zipping interface
                    * How? See next. 
                2. It would be interesting to add a plot showing: VELOCITY (i.e., change in position wrt time)
                    * Why? Because, it should reveal the zipping interface!
                        ** When a particle is on the suspended membrane, it is moving (i.e., VELOCITY != 0)
                        ** When a particle zips against the sidewall, it effectively becomes stationary (i.e., V = 0!)
                        ** So, a 1D plot of velocity vs. radius, or, 2D contour plot of velocity vs. (x, y),
                            should reveal the zipping interface. 
                    * But here is what may be really, really interesting:
                        ** Knowing the exact position of the zipping interface, we can evaluate SLIPPAGE ALONG THE SIDEWALL! 
    """
    # -
    # SETTINGS
    VERIFY_SETTINGS = False  # only need to run once per test configuration
    UPDATE_DEPENDENT = False  # True: update all dependent settings in dict_settings.xlsx
    # -
    # PRE-PROCESSING (True False)
    PRE_PROCESS_COORDS = False  # If you change andor_keithley_delay time, you must pre-process coords.
    PRE_PROCESS_IV = False  # Only needs to be run once; not dependent on synchronization timing settings.
    MERGE_COORDS_AND_VOLTAGE = False
    # -
    # ANALYSES
    SECOND_PASS_XYM = ['g']  # ['g', 'm']: use sub-pixel or discrete in-plane localization method
    # -
    # ---
    # -
    # ONLY USED IF DICT_TID{}_SETTINGS.XLSX IS NOT FOUND
    START_FRAME, END_FRAMES = (0, 8), (195, 205)  # only used if test_settings.xlsx is not found
    # -
    # ALTERNATE INITIAL COORDS
    EXPORT_INITIAL_COORDS = False
    USE_INITIAL_COORDS, USE_TID = False, 1

    # ------------------------------------------------------------------------------------------------------------------
    # YOU SHOULD NOT NEED TO CHANGE BELOW
    # ------------------------------------------------------------------------------------------------------------------
    # ---
    # FILEPATHS
    # ---
    # directories
    BASE_DIR = join('/Users/mackenzie/Desktop/zipper_paper/Testing/Zipper Actuation', TEST_CONFIG)
    SAVE_DIR = join(BASE_DIR, 'analyses')
    SAVE_SETTINGS = join(SAVE_DIR, 'settings')
    SAVE_COORDS = join(SAVE_DIR, 'coords')
    # SAVE_COORDS_W_PIXELS = join(SAVE_COORDS, 'pixels')
    PATH_REPRESENTATIVE = join(SAVE_DIR, 'representative_test{}')
    # -
    # settings
    FP_SETTINGS = join(SAVE_SETTINGS, 'dict_settings.xlsx')
    FP_TEST_SETTINGS = join(SAVE_SETTINGS, 'dict_tid{}_settings.xlsx'.format(TID))
    # -
    # 3D particle tracking
    READ_COORDS_DIR = join(BASE_DIR, 'results', 'test-idpt_test-{}'.format(TID))
    FN_COORDS_STARTS_WITH = 'test_coords_t'
    FN_COORDS_SAVE = 'tid{}_coords.xlsx'.format(TID)
    FN_COORDS_INITIAL_SAVE = 'tid{}_init_coords.xlsx'.format(TID)
    FN_COORDS_INITIAL_READ = 'tid{}_init_coords.xlsx'.format(USE_TID)
    # -
    # Keithley
    READ_IV_DIR = join(BASE_DIR, 'I-V')
    FN_IV_STARTS_WITH = 'tid{}_'.format(TID)
    FN_IV_SAVE = 'tid{}_I-V.xlsx'.format(TID)
    FN_IV_AVG_SAVE = 'tid{}_average-V-t.xlsx'.format(TID)
    # Merged (3DPT + Keithley)
    FN_MERGED = 'tid{}_merged-coords-volt.xlsx'.format(TID)
    # -
    # -
    # make dirs
    for pth in [SAVE_DIR, SAVE_SETTINGS, SAVE_COORDS]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # -
    # Pre-run checks
    if USE_INITIAL_COORDS is True and not os.path.exists(join(SAVE_COORDS, FN_COORDS_INITIAL_READ)):
        raise ValueError("Initial coords for tid {} do not exist.".format(USE_TID))
    # -
    # ------------------------------------------------------------------------------------------------------------------
    # ---
    # SETTINGS
    # ---
    # setup-specific
    # (i.e., for each set of calibration images)
    DICT_SETTINGS = settings.get_settings(fp_settings=FP_SETTINGS, name='settings', update_dependent=UPDATE_DEPENDENT)
    # -
    # test-specific
    # (i.e., for each set of test images)
    if os.path.exists(FP_TEST_SETTINGS):
        DICT_TEST = settings.get_settings(fp_settings=FP_TEST_SETTINGS, name='test')
    else:
        if PRE_PROCESS_IV:
            FN_IV = [x for x in os.listdir(READ_IV_DIR) if x.startswith(FN_IV_STARTS_WITH) and x.endswith('.xlsx')][0]
        else:
            FN_IV = None
        DICT_TEST = settings.make_test_settings(
            filename=FN_IV,
            start_frame=START_FRAME,
            end_frames=END_FRAMES,
            dict_settings=DICT_SETTINGS,
        )
        settings.write_settings_to_df(
            s=DICT_TEST,
            filepath_save=FP_TEST_SETTINGS,
        )
    # ---
    # PRE-PROCESSING
    # ---
    if PRE_PROCESS_COORDS or PRE_PROCESS_IV or MERGE_COORDS_AND_VOLTAGE:
        # --- 3D particle tracking data
        if PRE_PROCESS_COORDS:
            # Find and read coords
            FN_COORDS = [x for x in os.listdir(READ_COORDS_DIR) if x.startswith(FN_COORDS_STARTS_WITH)][0]
            DF = pd.read_excel(join(READ_COORDS_DIR, FN_COORDS))
            # pre-process
            DF = dpt.pre_process_coords(df=DF, settings=DICT_SETTINGS)
            # calculate "initial coords", so that "relative coords" can be determined
            DF0 = dpt.calculate_initial_coords(df=DF, test_settings=DICT_TEST)
            if EXPORT_INITIAL_COORDS:
                DF0.to_excel(join(SAVE_COORDS, FN_COORDS_INITIAL_SAVE), index=True)
                raise ValueError("Always stop always exporting initial coords.")
            if USE_INITIAL_COORDS:
                DF0 = pd.read_excel(join(SAVE_COORDS, FN_COORDS_INITIAL_READ))
            DF = dpt.calculate_relative_coords(df=DF, test_settings=DICT_TEST, df0=DF0)
            # export transformed dataframe
            # for reference: physical + pixel coordinates
            # DF.to_excel(join(SAVE_COORDS_W_PIXELS, FN_COORDS_SAVE), index=True)
            # for analysis: we need only keep physical coordinates (and frames) for compactness
            PHYS_COLS = ['frame', 't_sync', 'dt',
                         'id', 'cm', 'z', 'dz',
                         'drg', 'drm', 'rg', 'rm',
                         'dxg', 'dxm', 'xg', 'xm',
                         'dyg', 'dym', 'yg', 'ym',
                         ]
            DF = DF[PHYS_COLS]
            DF.to_excel(join(SAVE_COORDS, FN_COORDS_SAVE), index=False)
        else:
            DF = pd.read_excel(join(SAVE_COORDS, FN_COORDS_SAVE))
        # --- Keithley source voltage data
        if PRE_PROCESS_IV:
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
            DFVG = smu.average_voltage_by_source_time(df=DFV)
            DFVG.to_excel(join(SAVE_COORDS, FN_IV_AVG_SAVE), index=False)
        else:
            DFV = pd.read_excel(join(SAVE_COORDS, FN_IV_SAVE))
        # --- Merge (3DPT + Keithley)
        if MERGE_COORDS_AND_VOLTAGE:
            DF = dpt.merge_with_iv(
                df=DF,
                dfv=DFV,
            )
            DF.to_excel(join(SAVE_COORDS, FN_MERGED), index=False)
    else:
        # KEITHLEY COLUMNS: 'SOURCE_TIME_MIDPOINT', 'STEP', 'VOLT', 'CURR', 'TIMESTAMP'
        DF = pd.read_excel(join(SAVE_COORDS, FN_MERGED))

    # ------------------------------------------------------------------------------------------------------------------
    # ---
    # FIRST-PASS EVALUATION
    # ---
    """
    Verify settings 
    """
    if VERIFY_SETTINGS:
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

    # ------------------------------------------------------------------------------------------------------------------
    # ---
    # SECOND-PASS EVALUATION
    # ---
    """
    Plot particle z-r displacement vs. (frame, time, and voltage)
    """
    for XYM in SECOND_PASS_XYM:
        analyses.second_pass(
            df=DF,
            xym=XYM,
            tid=TID,
            dict_settings=DICT_SETTINGS,
            dict_test=DICT_TEST,
            path_results=PATH_REPRESENTATIVE,
        )

    # -

    print("Completed without errors.")