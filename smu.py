import numpy as np
import pandas as pd



def get_dict_dtypes():
    """
    Retrieves a dictionary structure where keys are strings representing data field
    names, and values are types indicating the expected data types for each field.
    This structure can be used to validate or enforce data type consistency when
    working with structured datasets.

    :return: Dictionary mapping field names (str) to their data types
    :param tuple | int, float, str, list | dict

    """
    dict_dtypes = {
        'tid': int,
        'filename': str,
        'dpt_start_frame': tuple,
        'dpt_end_frames': tuple,
        'animate_frames': tuple,
        'smu_test_type': int,
        'smu_source_delay_time': float,
        'smu_vmax': int,
        'smu_dv': float,
        'smu_step_max': int,
        'samples_per_voltage_level': int,
        'drop_pids': list,
        'd0f_is_tid': int,
        'path_model_dZ_by_V': str,
    }
    return dict_dtypes


def get_dict_dtype_list(data_type):
    """
    Retrieves a list of keys based on the specified data type. The keys are categorized
    according to the provided data type such as 'int', 'str', 'eval', or 'special'.

    :param data_type: The type of data for which the list of keys is to be retrieved.
        Accepted values are 'int', 'str', 'eval', or 'special'.
    :type data_type: str
    :return: A list of keys corresponding to the specified data type.
    :rtype: list[str]
    :raises ValueError: If the provided data type is invalid or not in the
        accepted list ['int', 'str', 'eval', 'special'].
    """
    if data_type == 'int':
        keys = ['tid', 'smu_test_type', 'smu_vmax', 'smu_step_max', 'samples_per_voltage_level', 'd0f_is_tid']
    elif data_type == 'str':
        keys = ['filename', 'path_model_dZ_by_V']
    elif data_type == 'eval':
        keys = ['dpt_end_frames', 'animate_frames', 'drop_pids']
    elif data_type == 'special':
        keys = ['dpt_start_frame']
    else:
        raise ValueError('Names limited to: [int, string, eval]. No list for floats')
    return keys


def read_settings_to_dict(filepath):
    """
    Reads an Excel file and converts its content into a dictionary with specific
    data-type conversions defined by custom logic.

    :param filepath: The path to the Excel file to read
        the settings from.
    :type filepath: str

    :return: A dictionary containing the processed settings where keys are
        derived from the index of the Excel file and values are converted
        based on their respective expected data types.
    :rtype: dict
    """
    df = pd.read_excel(filepath, index_col=0)
    ks = df.index.values.tolist()
    vs = df[df.columns.values.tolist()[0]].values.tolist()

    dict_settings = {}
    for k, v in zip(ks, vs):
        if k in get_dict_dtype_list(data_type='int'):
            dict_settings.update({k: int(v)})
        elif k in get_dict_dtype_list(data_type='str'):
            dict_settings.update({k: str(v)})
        elif k in get_dict_dtype_list(data_type='eval'):
            dict_settings.update({k: eval(v)})
        elif k in get_dict_dtype_list(data_type='special'):
            if isinstance(v, int):
                dict_settings.update({k: (0, int(v))})
            else:
                dict_settings.update({k: eval(v)})
        else:
            dict_settings.update({k: float(v)})
    return dict_settings


def make_test_settings(filename, start_frame, end_frames, drop_pids, d0f_is_tid, dict_settings):
    """
    Generate test settings dictionary by parsing the provided parameters and extracting
    necessary details from the filename.

    This function is designed to configure the test setup based on the provided inputs
    such as filename, frame details, process IDs to drop, and a dictionary of additional
    settings. The filename is parsed to extract Keithley voltage waveform details if present.

    :param filename: Name of the test file that may contain metadata information
    :param start_frame: A single integer or a tuple defining either the start frame or
        a range of frames
    :param end_frames: Ending frame(s) for the test
    :param drop_pids: List of process IDs to exclude from the test
    :param d0f_is_tid:
    :param dict_settings: Dictionary containing additional test parameters, such as
        source delay time associated with test types
    :return: A dictionary containing comprehensive settings to configure the test
    :rtype: dict
    """
    if isinstance(start_frame, int):
        start_frame = (0, start_frame)
    # parse I-V filename to get Keithley voltage waveform details
    if filename is not None:
        tid, test_type, vmax, dv, step_max = parse_voltage_waveform_from_filename(filename=filename)
    else:
        tid, test_type, vmax, dv, step_max = -1, 1, 0, 0, 0

    time_per_sample = 1 / dict_settings['frame_rate']
    time_per_voltage_level = dict_settings['source_delay_time_by_test'][test_type]
    samples_per_voltage_level = np.max([int(np.round(time_per_voltage_level / time_per_sample)), 1])

    # -
    dict_test = {
        'tid': tid,
        'filename': filename,
        'dpt_start_frame': start_frame,
        'dpt_end_frames': end_frames,
        'smu_test_type': test_type,
        'smu_source_delay_time': time_per_voltage_level,
        'smu_vmax': vmax,
        'smu_dv': dv,
        'smu_step_max': step_max,
        'samples_per_voltage_level': samples_per_voltage_level,
        'drop_pids': drop_pids,
        'd0f_is_tid': d0f_is_tid,
    }
    return dict_test


def parse_voltage_waveform_from_filename(filename):
    """
    Parses a given filename to extract voltage waveform-related parameters. The function splits
    the provided filename using predefined substrings to identify and extract the `tid` (Test Identifier),
    `test_type` (Test Type), `vmax` (Maximum Voltage), `dv` (Voltage Step Size), and `step_max`
    (Maximum Steps) information.

    Extracted parameters are subsequently calculated and returned based on specific assumptions
    about the filename's format. The format of the filename is assumed to follow a deterministic
    pattern that corresponds to the parsing logic within this function.

    The function returns a tuple containing the parsed and computed `tid`, `test_type`, `vmax`,
    `dv`, and `step_max` values for the voltage waveform analysis.

    tid, test_type, vmax, dv, step_max = smu.parse_voltage_waveform_from_filename(filename)

    :param filename: The input string filename in a predefined format containing the parameters.
    :type filename: str
    :return: A tuple with parsed `tid` (Test Identifier), `test_type` (Test Type as integer),
             `vmax` (Maximum Voltage as integer), `dv` (Voltage Step Size as float), and
             `step_max` (Maximum Steps as integer).
    :rtype: tuple
    """
    tid = filename.split('tid')[1].split('_test')[0]
    test_type = int(filename.split('tid{}_test'.format(tid))[1].split('_')[0])
    vmax = int(filename.split('tid{}_test{}_'.format(tid, test_type))[1].split('V_')[0])
    dv = float(filename.split('tid{}_test{}_{}V_'.format(tid, test_type, vmax))[1].split('dV')[0])
    step_max = int(np.abs(vmax) / dv)
    return tid, test_type, vmax, dv, step_max


def pre_process_keithley(df, delay, integration, latency=100e-6):
    """
    Preprocesses the Keithley device data to modify time referencing and,
    subsequently, prepares the DataFrame for further analysis. The function
    realigns timestamp data with respect to source delay calculations, adjusts
    time-series data such that individual voltage levels are set relative to a
    default timestamp while preserving order of operations. Finally, it augments
    and sorts the input DataFrame with additional required columns.

    df = pre_process_keithley_df(df, delay, integration)

    :param df: Input pandas DataFrame containing the device data.
               It must include columns `VOLT`, `CURR`, and `TIMESTAMP`.
    :type df: pandas.DataFrame
    :param delay: Delay offset in seconds added to the source reference
                  time in order to realign time indices.
    :type delay: float
    :param integration: Time in seconds related to the integration of the
                        signal during measurement.
    :type integration: float
    :param latency: Latency in seconds related to communication delays
                    between the device and recording session.
    :type latency: float
    :return: Processed pandas DataFrame with updated 'TIME', ordered, and
             columns represent recalculated time with composite structure.
    :rtype: pandas.DataFrame
    """
    KV, KI, KT = 'VOLT', 'CURR', 'TIMESTAMP'
    volt, curr, timestamp = df[KV].to_numpy(), df[KI].to_numpy(), df[KT].to_numpy()

    # --- Estimate continuous voltage values by time
    # time since first sample
    time_sample = timestamp - timestamp[0]
    # estimated time when voltage level was set
    time_source = time_sample - delay - integration
    # offset time so time = 0 is when voltage level was first set
    time_sample = time_sample - time_source[0]
    time_source = time_source - time_source[0]

    # concat dataframes
    df['TIME'] = pd.Series(time_sample)
    df2 = pd.DataFrame(np.vstack([volt, time_source]).T, columns=[KV, 'TIME'])
    df = pd.concat([df2, df])
    df = df.sort_values('TIME', ascending=True)
    # add column to indicate which voltage "step" each row belongs to
    df = df.reset_index(names='STEP')
    # round 'TIME' to decimals
    df = df.round({'TIME': 3})

    return df

def average_voltage_by_source_time(df):
    """
    Calculates the average voltage grouped by the 'STEP' column, computes the
    mean for each group, resets the index, and rounds the 'TIME' column to
    four decimal places.

    :param df: Input DataFrame containing the data with at least the columns
        'STEP' and 'TIME'.
    :type df: pandas.DataFrame
    :return: A new DataFrame with the average voltage grouped by 'STEP',
        where the 'TIME' column is rounded to four decimal places.
    :rtype: pandas.DataFrame
    """
    return df.groupby('STEP').mean().reset_index().round({'TIME': 4})