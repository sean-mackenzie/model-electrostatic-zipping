from os.path import join
import os
import pandas as pd

from utils import settings
import smu, awg



def test_settings_handler(settings_handler_dict, dict_settings):
    if os.path.exists(settings_handler_dict['fp_test_settings']):
        dict_test = settings.get_settings(
            fp_settings=settings_handler_dict['fp_test_settings'],
            name=settings_handler_dict['iv_acdc'],
        )
    else:
        if not settings_handler_dict['pre_process_iv']:
            # in rare cases (for DC tests only), we want to
            # return a dictionary of "generic" test settings
            # (e.g., because data is missing or incompatible)
            dict_test = make_generic_test_settings(settings_handler_dict, dict_settings)
        else:
            # find file with test settings
            fn_iv = [
                x for x in os.listdir(settings_handler_dict['read_iv_dir']) if
                x.startswith(settings_handler_dict['fn_iv_startswith']) and x.endswith('.xlsx')
            ][0]

            # make test settings dict
            if fn_iv.endswith(settings_handler_dict['fn_ivdc_endswith']):
                # make DC test settings dict
                dict_test = smu.make_test_settings(
                    filename=fn_iv,
                    start_frame=settings_handler_dict[ 'start_frame'],
                    end_frames=settings_handler_dict['end_frames'],
                    drop_pids=settings_handler_dict[ 'drop_pids' ],
                    dict_settings=dict_settings,
                )
            elif fn_iv.endswith(settings_handler_dict['fn_ivac_endswith']):

                # make AC test settings dict
                dict_test = awg.make_test_settings(
                    filename=fn_iv,
                    settings_handler_dict=settings_handler_dict,
                    dict_settings=dict_settings,
            )
            else:
                raise ValueError(f'fn_iv ({fn_iv}) does not end with'
                                 f' {settings_handler_dict["fn_ivdc_endswith"]} or'
                                 f' {settings_handler_dict["fn_ivac_endswith"]}')
        # export
        settings.write_settings_to_df(
            s=dict_test,
            filepath_save=settings_handler_dict['fp_test_settings'],
        )
    return dict_test


def make_generic_test_settings(settings_handler_dict, dict_settings):
    dict_test_generic = settings.make_test_settings_smu(
        filename=None,
        start_frame=settings_handler_dict['start_frame'],
        end_frames=settings_handler_dict['end_frames'],
        drop_pids=settings_handler_dict['drop_pids'],
        dict_settings=dict_settings,
    )
    return dict_test_generic


if __name__ == "__main__":
    # THESE ARE THE ONLY SETTINGS YOU SHOULD CHANGE
    TEST_CONFIG = '03072025_W12-D1_C19-30pT_20+10nmAu'
    TID = 61
    IV_ACDC = 'AC'

    # -
    # SETTINGS (True False)
    VERIFY_SETTINGS = True  # only need to run once per test configuration
    UPDATE_DEPENDENT = True  # True: update all dependent settings in dict_settings.xlsx
    # -
    # PRE-PROCESSING (True False)
    PRE_PROCESS_COORDS = True  # If you change andor_keithley_delay time, you must pre-process coords.
    PRE_PROCESS_IV = True  # Only needs to be run once; not dependent on synchronization timing settings.
    MERGE_COORDS_AND_VOLTAGE = True
    # -
    # ANALYSES
    SECOND_PASS_XYM = ['g']  # ['g', 'm']: use sub-pixel or discrete in-plane localization method
    # -
    # ---
    # -
    # ONLY USED IF DICT_TID{}_SETTINGS.XLSX IS NOT FOUND
    START_FRAME, END_FRAMES = (0, 4), (143, 147)  # (a<x<b; NOT: a<=x<=b) only used if test_settings.xlsx not found
    DROP_PIDS = []  # []: remove bad particles from coords
    # -
    # ALTERNATE INITIAL COORDS
    EXPORT_INITIAL_COORDS = False  # False True
    D0F_IS_TID = 1

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
    SAVE_SETTINGS = join(SAVE_DIR, 'settings')
    SAVE_COORDS = join(SAVE_DIR, 'coords')
    # -
    # settings
    FP_SETTINGS = join(SAVE_SETTINGS, 'dict_settings.xlsx')
    FP_TEST_SETTINGS = join(SAVE_SETTINGS, 'dict_tid{}_settings.xlsx'.format(TID))
    # -
    # Keithley
    READ_IV_DIR = join(BASE_DIR, 'I-V', 'xlsx')
    FN_IV_STARTS_WITH = 'tid{}_'.format(TID)
    FN_IVDC_ENDS_WITH = 'dV.xlsx'  # if I-V file has this ending, then it is a DC test
    FN_IVAC_ENDS_WITH = '_data.xlsx'  # if I-V file has this ending, then it is an AC test
    # -
    # -
    # make dirs
    for pth in [SAVE_DIR, SAVE_SETTINGS, SAVE_COORDS]:
        if not os.path.exists(pth):
            os.makedirs(pth)
    # -
    # ------------------------------------------------------------------------------------------------------------------
    # ---
    # SETTINGS
    # ---
    SETTINGS_HANDLER_DICT = {
        'fp_settings': FP_SETTINGS,
        'update_dependent': UPDATE_DEPENDENT,
        'fp_test_settings': FP_TEST_SETTINGS,
        'iv_acdc': IV_ACDC,
        'pre_process_iv': PRE_PROCESS_IV,
        'read_iv_dir': READ_IV_DIR,
        'fn_iv_startswith': FN_IV_STARTS_WITH,
        'fn_ivdc_endswith': FN_IVDC_ENDS_WITH,
        'fn_ivac_endswith': FN_IVAC_ENDS_WITH,
        'drop_pids': DROP_PIDS,
        'start_frame': START_FRAME,
        'end_frames': END_FRAMES,
    }

    dict_settings, dict_test = settings.settings_handler(settings_handler_dict)

    # setup-specific
    # (i.e., for each set of calibration images)
    DICT_SETTINGS = settings.get_settings(fp_settings=FP_SETTINGS, name='settings', update_dependent=UPDATE_DEPENDENT)
    # -
    # test-specific
    # (i.e., for each set of test images)
    """if os.path.exists(FP_TEST_SETTINGS):
        DICT_TEST = settings.get_settings(fp_settings=FP_TEST_SETTINGS, name='test')
    else:
        if PRE_PROCESS_IV:
            # find file with test settings
            FN_IV = [x for x in os.listdir(READ_IV_DIR) if x.startswith(FN_IV_STARTS_WITH) and x.endswith('.xlsx')][0]

        # make test settings dict
        if FN_IV.endswith(FN_IVDC_ENDS_WITH):
            # make DC test settings dict
            # (if FN_IV is None, then exports generic test settings)
            DICT_TEST = settings.make_test_settings_smu(
                filename=FN_IV,
                start_frame=START_FRAME,
                end_frames=END_FRAMES,
                drop_pids=DROP_PIDS,
                dict_settings=DICT_SETTINGS,
            )
        elif FN_IV.endswith(FN_IVAC_ENDS_WITH):
            # make AC test settings dict
            DICT_TEST = settings.make_test_settings_awg(
                filename=FN_IV,
                drop_pids=DROP_PIDS,
                dict_settings=DICT_SETTINGS,
            )
        # export
        settings.write_settings_to_df(
            s=DICT_TEST,
            filepath_save=FP_TEST_SETTINGS,
        )
    # ---"""

    DICT_TEST = test_settings_handler(
        settings_handler_dict=SETTINGS_HANDLER_DICT,
        dict_settings=DICT_SETTINGS,
    )
    a = 1

