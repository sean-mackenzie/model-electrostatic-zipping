import os
from os.path import join
import pandas as pd
import smu, awg

def get_fid(feature_label):
    """
    Determines a unique feature identifier (fid) based on the input feature
    label. The `feature_label` is analyzed by its starting and ending
    characters to compute the fid. The function raises exceptions for invalid
    or unsupported feature_label patterns.

    :param feature_label: The label used to derive the feature identifier.
    :type feature_label: str
    :return: The computed feature identifier based on the input label.
    :rtype: int
    :raises ValueError: If `feature_label` does not start with a supported
        character (a-f) or end with a supported value (1-3).
    """
    if feature_label.startswith('a'):
        fid = 0
    elif feature_label.startswith('b'):
        fid = 3
    elif feature_label.startswith('c'):
        fid = 6
    elif feature_label.startswith('d'):
        fid = 9
    elif feature_label.startswith('e'):
        fid = 12
    elif feature_label.startswith('f'):
        fid = 15
    else:
        raise ValueError()
    if feature_label.endswith('1'):
        pass
    elif feature_label.endswith('2'):
        fid += 1
    elif feature_label.endswith('3'):
        fid += 2
    else:
        raise ValueError()
    return fid


def get_dict_dtypes():
    """
    Generates and returns a dictionary containing key-value pairs where the keys are string names of
    parameters/settings and the values represent the expected data types of each parameter.

    The dictionary provides a mapping of configuration parameters to their associated data types
    for reference or validation purposes. This allows consistent handling and verification of the
    expected data types across various settings in an application or use case.

    :rtype: dict
    :return: A dictionary where the keys are parameter names (as strings) and the values indicate
             the corresponding data types associated with each parameter.
    """
    dict_dtypes = {
        'feature_label': str,
        'fid': int,
        'exposure_time': float,
        'frame_rate': float,
        'microns_per_pixel': float,
        'image_size': int,
        'field_of_view': float,
        'nplc': float,
        'integration_time': float,
        'source_delay_time_by_test': dict,
        'padding': int,
        'drop_frame_zero': bool,
        'drop_first_n_frames': int,
        'scale_z': float,
        'xyc_pixels': tuple,
        'xyc_microns': tuple,
        'radius_pixels': float,
        'radius_hole_pixels': float,
        'radius_microns': float,
        'radius_hole_microns': float,
        'andor_keithley_delay': float,
        'path_process_profiles': str,
        'fid_process_profile': int,
        'step_process_profile': int,
        'path_model': str,
        'model_mkey': str,
        'model_mval': float,
        'path_model_settings': str,
        'path_model_dZ_by_V': str,
        'path_model_strain': str,
        'path_image_overlay': str,
        'membrane_thickness': float,
        'andor_delay_to_awg_input': float,
        'monitor_delay_to_awg_input': float,
        'monitor_time_scale_factor_est': float,
    }
    return dict_dtypes


def get_dict_dtype_list(data_type):
    """
    Retrieves a list of keys based on the specified data type. The function
    supports predefined data types, such as 'int', 'str', 'eval', and 'special',
    and corresponds each to a specific list of strings. Raises an exception
    if the data type is not recognized.

    :param data_type: A string representing the data type. Supported values
                      are 'int', 'str', 'eval', and 'special'.
    :type data_type: str
    :return: A list of strings corresponding to the specified data type.
    :rtype: list
    :raises ValueError: If the data type is not among the supported values.
    """
    if data_type == 'int':
        keys = ['fid', 'image_size', 'padding', 'drop_first_n_frames', 'fid_process_profile', 'step_process_profile']
    elif data_type == 'str':
        keys = ['feature_label', 'model_mkey']  #, 'path_process_profiles', 'path_image_overlay', 'path_model', 'path_model_settings', 'path_model_dZ_by_V', 'path_model_strain']
    elif data_type == 'path':
        keys = ['path_process_profiles', 'path_image_overlay', 'path_model',
                'path_model_settings', 'path_model_dZ_by_V', 'path_model_strain']
    elif data_type == 'eval':
        keys = ['source_delay_time_by_test', 'xyc_pixels', 'xyc_microns']
    elif data_type == 'special':
        keys = []
    else:
        raise ValueError('Names limited to: [int, string, eval]. No list for floats')
    return keys


def read_settings_to_dict_handler(filepath, name, update_dependent=False):
    """
    Reads settings from an Excel file and returns a dictionary of settings with appropriate data types.
    The function processes data differently based on the `name` parameter, and handles specific conditions
    such as dependent settings updates when `update_dependent` is set to True.

    :param filepath: Path to the input Excel file containing the settings.
    :type filepath: str
    :param name: Specifies the type of settings to process. Must be one of ['settings', 'DC', 'AC'].
    :type name: str
    :param update_dependent: Determines whether dependent settings should be ignored in the processing.
    :type update_dependent: bool
    :return: A dictionary containing the processed settings from the input file with the appropriate data types.
    :rtype: dict
    :raises ValueError: If the `name` is not one of ['settings', 'DC', 'AC'].
    """
    if name not in ['settings', 'DC', 'AC']:
        raise ValueError('Name not in: [settings, DC, AC]')

    df = pd.read_excel(filepath, index_col=0)
    ks = df.index.values.tolist()
    vs = df.v.values.tolist()

    dict_settings = {}
    if name == 'settings':
        for k, v in zip(ks, vs):
                if k in get_dict_dtype_list(data_type='int'):
                    dict_settings.update({k: int(v)})
                elif k in get_dict_dtype_list(data_type='eval'):
                    dict_settings.update({k: eval(v)})
                elif k in get_dict_dtype_list(data_type='path'):
                    if isinstance(v, str):
                        v = v.strip("'")
                    dict_settings.update({k: str(v)})
                elif k in get_dict_dtype_list(data_type='str'):
                    dict_settings.update({k: str(v)})
                else:
                    dict_settings.update({k: float(v)})
        if update_dependent:
            dependent_settings = ['field_of_view', 'radius_microns', 'radius_hole_microns',
                                  'xyc_microns', 'integration_time', 'fid']
            for k in dependent_settings:
                _ = dict_settings.pop(k, None)
        dict_settings = check_dependent_settings(dict_settings, name, filepath)
    elif name == 'DC':
        dict_settings = smu.read_settings_to_dict(filepath=filepath)
    elif name == 'AC':
        dict_settings = awg.read_settings_to_dict(filepath=filepath)

    return dict_settings


def check_dependent_settings(dict_settings, name, filepath):
    """
    Check and update dependent settings within a given dictionary.

    This function inspects a dictionary of settings and verifies if certain
    dependent settings are missing. If missing, it calculates their values
    based on existing keys and updates the dictionary accordingly. The
    function also writes the updated settings to a file if any changes occur.

    :param dict_settings: Dictionary containing the configuration settings.
    :param name: Name identifying the particular configuration (e.g., 'settings').
    :param filepath: Path to save the updated settings file.
    :return: The updated settings dictionary.
    :rtype: dict
    """
    update_settings = False
    if name == 'settings':
        for k2, k1 in zip(['field_of_view', 'radius_microns', 'radius_hole_microns'],
                          ['image_size', 'radius_pixels', 'radius_hole_pixels']):
            if k2 not in dict_settings.keys():
                dict_settings.update({
                    k2: dict_settings[k1] * dict_settings['microns_per_pixel']
                })
                update_settings = True
        if 'xyc_microns' not in dict_settings.keys():
            dict_settings.update({
                'xyc_microns': (dict_settings['xyc_pixels'][0] * dict_settings['microns_per_pixel'],
                                dict_settings['xyc_pixels'][1] * dict_settings['microns_per_pixel'])
            })
            update_settings = True
        if 'integration_time' not in dict_settings.keys():
            dict_settings.update({
                'integration_time': dict_settings['nplc'] / 60,
            })
            update_settings = True
        if 'fid' not in dict_settings.keys():
            dict_settings.update({
                'fid': get_fid(feature_label=dict_settings['feature_label']),
            })
            if 'fid_process_profile' not in dict_settings.keys():
                dict_settings.update({
                    'fid_process_profile': dict_settings['fid'],
                })
            update_settings = True

    if update_settings:
        write_settings_to_df(s=dict_settings, filepath_save=filepath)
    return dict_settings


def write_settings_to_df(s, filepath_save):
    """
    Write settings dictionary into an Excel file using pandas.

    This function takes a dictionary representing settings, converts it
    into a pandas DataFrame, and saves it as an Excel file with the given
    file path. The keys of the dictionary are written as rows with their
    associated values under a specified column.

    :param s: A dictionary containing settings to be saved. The keys
        represent parameter names, and their associated values are the
        settings values.
    :param filepath_save: A string representing the file path where the
        Excel file will be saved.
    :return: None
    """
    df = pd.DataFrame.from_dict(data=s, orient='index', columns=['v'])
    df.to_excel(filepath_save, index=True, index_label='k', sheet_name='settings')


def get_settings(fp_settings, name, update_dependent=False):
    return read_settings_to_dict_handler(filepath=fp_settings, name=name, update_dependent=update_dependent)


def test_settings_handler(settings_handler_dict, dict_settings):
    """
    Handles the processing and generation of test settings based on the specified
    parameters and available resources. This includes retrieving pre-configured
    settings, creating generic settings, or generating specific AC/DC test
    settings by analyzing available data files. The function ensures the derived
    settings are appropriately handled and optionally saved to a provided file
    path, aiming to facilitate flexible and reliable test setup.

    :param settings_handler_dict: A dictionary containing various configuration
        parameters related to file paths, naming conventions, and specific flags
        (e.g., to control processing behavior) essential for determining and
        processing test settings.
    :type settings_handler_dict: dict
    :param dict_settings: A dictionary containing the structural and configurational
        test settings to be utilized in cases where specific or generic test
        settings need to be created or modified.
    :type dict_settings: dict
    :return: A dictionary representing the resolved or generated test settings,
        whether sourced from pre-existing configurations, dynamically generated AC/DC
        settings, or generic defaults when data is not available or applicable.
    :rtype: dict
    """
    if os.path.exists(settings_handler_dict['fp_test_settings']):
        dict_test = get_settings(
            fp_settings=settings_handler_dict['fp_test_settings'],
            name=settings_handler_dict['iv_acdc'],
        )
    else:
        if settings_handler_dict['use_generic_dc_test_settings']:
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
                    d0f_is_tid=settings_handler_dict['d0f_is_tid'],
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
        write_settings_to_df(
            s=dict_test,
            filepath_save=settings_handler_dict['fp_test_settings'],
        )
    return dict_test


def make_generic_test_settings(settings_handler_dict, dict_settings):
    """
    Generate a dictionary of generic test settings using the provided settings handler
    parameters and dictionary settings.

    This function combines specific parameters extracted from the provided
    `settings_handler_dict` with the provided dictionary settings `dict_settings`
    to create a comprehensive dictionary of settings tailored for executing tests.

    :param settings_handler_dict: Dictionary containing specific handler settings.
       It should include the following keys:
       - 'start_frame': Starting frame for processing.
       - 'end_frames': End frames for processing.
       - 'drop_pids': List of process IDs to ignore during execution.
    :param dict_settings: Additional dictionary of settings that will be included
       alongside the `settings_handler_dict` values.
    :return: A dictionary containing the combined test settings that
       consolidates the input values for practical usage.
    :rtype: dict
    """
    dict_test_generic = smu.make_test_settings(
        filename=None,
        start_frame=settings_handler_dict['start_frame'],
        end_frames=settings_handler_dict['end_frames'],
        drop_pids=settings_handler_dict['drop_pids'],
        dict_settings=dict_settings,
    )
    return dict_test_generic



def settings_handler(settings_handler_dict):
    """
    Handles the settings for a system by retrieving and testing the settings
    as per the provided settings handler dictionary. The function retrieves
    settings based on the provided configuration and subsequently tests
    the settings to ensure proper setup.

    :param settings_handler_dict: A dictionary containing necessary configurations
        for handling settings. It should include keys like 'fp_settings' for file
        path settings and 'update_dependent' for determining if dependent settings
        need to be updated.
    :type settings_handler_dict: dict
    :return: A tuple containing the settings dictionary and the test results
        dictionary after processing and validating the input.
    :rtype: tuple
    """
    dict_settings = get_settings(
        fp_settings=settings_handler_dict['fp_settings'],
        name='settings',
        update_dependent=settings_handler_dict['update_dependent'],
    )

    dict_test = test_settings_handler(
        settings_handler_dict=settings_handler_dict,
        dict_settings=dict_settings,
    )

    return dict_settings, dict_test


if __name__ == "__main__":

    def export_manual_settings(fp_settings):
        # --- Experimental parameters
        # Optical system
        EXPOSURE_TIME = 0.049735  # (s)
        FRAME_RATE = 20  # (Hz)
        MICRONS_PER_PIXEL = 1.6  # (microns/pixel) (note: "exact" = 1.5944 microns/pixel)
        IMAGE_SIZE = 512  # number of pixels (assuming size: L x L)
        FIELD_OF_VIEW = IMAGE_SIZE * MICRONS_PER_PIXEL  # microns (L x L)
        # Keithley
        NPLC = 1  # (number of power line cycles)
        INTEGRATION_TIME = NPLC / 60
        SOURCE_DELAY_TIME_BY_TEST = {1: 0.15, 2: 0.5}  # (s)
        # Relating external stimulus (time-dependent voltage)
        # to image space (frames)
        ANDOR_KEITHLEY_DELAY = 0.055  # (s) Assuming first frame = 1, Andor time (synchronized) = Andor time (raw) + delay
        # 3D particle tracking
        PADDING = 30  # (pixels)  will be subtracted from raw x-y coordinates
        DROP_FRAME_ZERO = True  # Usually, a faux baseline is used and needs to be removed
        SCALE_Z = -1.0  # Usually -1 b/c typical direction of calibration and displacement.
        # Relating physical space (microns)
        # to image space (pixels)
        XC_PIXELS, YC_PIXELS, A_PIXELS = -64, 268, 1080  # (pixels) diaphragm center x,y (w/o padding) and radius
        XC, YC, A = XC_PIXELS * MICRONS_PER_PIXEL, YC_PIXELS * MICRONS_PER_PIXEL, A_PIXELS * MICRONS_PER_PIXEL  # (microns)
        # ---
        # SETUP
        # ---
        dict_settings = {
            'exposure_time': EXPOSURE_TIME,
            'frame_rate': FRAME_RATE,
            'microns_per_pixel': MICRONS_PER_PIXEL,
            'image_size': IMAGE_SIZE,
            'field_of_view': FIELD_OF_VIEW,
            'nplc': NPLC,
            'integration_time': INTEGRATION_TIME,
            'source_delay_time_by_test': SOURCE_DELAY_TIME_BY_TEST,
            'padding': PADDING,
            'drop_frame_zero': DROP_FRAME_ZERO,
            'scale_z': SCALE_Z,
            'xyc_pixels': (XC_PIXELS, YC_PIXELS),
            'xyc_microns': (XC, YC),
            'radius_pixels': A_PIXELS,
            'radius_microns': A,
            'andor_keithley_delay': ANDOR_KEITHLEY_DELAY,
        }

        df_settings = pd.DataFrame.from_dict(data=dict_settings, orient='index', columns=['v'])
        df_settings.to_excel(fp_settings, index=True, index_label='k')


    # ---

    # ---

    FP_SETTINGS = 'PATH'
    export_manual_settings(fp_settings=FP_SETTINGS)