import os
from os.path import join
import numpy as np
import pandas as pd


def get_dict_dtypes():
    """
    Retrieves a dictionary mapping parameter names to their corresponding data types
    used for configuration or processing settings.

    This function returns a comprehensive dictionary where the keys represent the
    parameter names and the values define their associated data types. The dictionary
    is intended to facilitate the validation and handling of configuration parameters
    in applications or processing workflows.

    :return: A dictionary containing parameter names as keys and their corresponding
             data types as values.
    :rtype: dict
    """
    dict_dtypes = {
        'tid': int,
        'filename': str,
        'iv_acdc': str,
        'dpt_start_frame': eval,
        'dpt_end_frames': eval,
        'animate_frames': eval,
        'drop_pids': list,
        'save_dir': str,
        'save_id': str,
        'test_type': str,
        'output_volt': float,
        'output_dc_offset': float,
        'awg_wave': str,
        'awg_freq': float,
        'awg_square_duty_cycle': float,
        'awg_volt_unit': str,
        'awg_volt': float,
        'awg_dc_offset': float,
        'awg_mod_state': str,
        'awg_mod_wave': str,
        'awg_mod_freq': float,
        'awg_mod_depth': float,
        'awg_mod_source': str,
        'awg_mod_ampl_ext': str,
        'awg_mod_ampl_shape': str,
        'awg_mod_ampl_start': float,
        'awg_mod_ampl_step': float,
        'awg_mod_ampl_stop': float,
        'awg_mod_ampl_dwell_off': float,
        'awg_mod_ampl_dwell_on': float,
        'awg_mod_ampl_dwell_per_step': float,
        'awg_mod_ampl_num_steps': float,
        'awg_mod_ampl_cycles': float,
        'awg_mod_ampl_values': str,
        'awg_mod_dwell_values': str,
        'awg_output_termination': str,
        'amplifier_gain': float,
        'awg_min_allowable_amplitude': float,
        'output_min_possible_amplitude': float,
        'keithley_monitor': str,
        'keithley_nplc': float,
        'keithley_num_samples': float,
        'andor_trigger_keithley': str,
        'delay_agilent_after_andor': float,
        'keithley_integration_period': float,
        'keithley_fetch_delay': float,
        'keithley_ratio_fetch_to_integration': float,
        'keithley_monitor_units': str,
        'keithley_monitor_zero_bias': float,
        'keithley_monitor_to_measure': float,
        'keithley_measure_units': str,
        'keithley_timeout': float,
    }
    return dict_dtypes


def get_dict_dtype_list(data_type):
    """
    This function retrieves a list of dictionary keys corresponding to a specified
    data type. The type of keys returned depends on the provided `data_type`.
    If an invalid `data_type` is provided, a ValueError is raised.

    :param data_type: The data type for which the dictionary key list is requested.
        Supported values are 'int', 'str', 'eval', and 'special'.
        Providing an unsupported value raises a ValueError.
    :type data_type: str

    :return: A list of keys associated with the specified data type.
    :rtype: list

    :raises ValueError: If the given data_type is not one of the supported values
        ['int', 'str', 'eval', 'special'].
    """
    if data_type == 'int':
        keys = ['tid']
    elif data_type == 'str':
        keys = ['filename',
                'iv_acdc',
                'save_dir',
                'save_id',
                'test_type',
                'awg_wave',
                'awg_volt_unit',
                'awg_mod_state',
                'awg_mod_wave',
                'awg_mod_source',
                'awg_mod_ampl_ext',
                'awg_mod_ampl_shape',
                'awg_output_termination',
                'keithley_monitor',
                'andor_trigger_keithley',
                'keithley_monitor_units',
                'keithley_measure_units',
                ]
    elif data_type == 'eval':
        keys = ['drop_pids', 'dpt_start_frame', 'dpt_end_frames', 'animate_frames']
    elif data_type == 'special':
        keys = ['awg_mod_ampl_values', 'awg_mod_dwell_values']
    else:
        raise ValueError('Names limited to: [int, string, eval]. No list for floats')
    return keys


def read_settings_to_dict(filepath):
    """
    Reads settings from an Excel file and converts them into a dictionary with specific data types.

    This function reads the 'settings' sheet of the provided Excel file and processes each setting
    from the index column and the first column of the sheet. Using predefined data type categories
    through the `get_dict_dtype_list()` function, it determines the data type for each setting and
    constructs a dictionary with keys and correctly typed values.

    :param filepath: Path to the Excel file containing the settings.
    :type filepath: str
    :return: A dictionary where each key corresponds to a setting name and the associated value is cast
        to the appropriate Python type (int, str, evaluated expression, or float).
    :rtype: dict
    """
    df = pd.read_excel(filepath, index_col=0, sheet_name='settings')
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
            v = v.replace(' ', ',')
            v = v.replace('\n', '')
            v = v.replace(',,,', ',')
            v = v.replace(',,', ',')
            dict_settings.update({k: eval(v)})
        else:
            dict_settings.update({k: float(v)})
    return dict_settings


def make_test_settings(filename, settings_handler_dict, dict_settings):
    """
    Generate a test settings dictionary by reading_settings from a file and augmenting it with additional
    provided settings and hardcoded values.

    This function reads a settings file, transforms its content into a dictionary, and then enhances
    the resulting dictionary with information such as the file name, specific settings from
    `settings_handler_dict`, and other predefined key-value pairs.

    :param filename: Name of the file to read the base settings from.
    :type filename: str
    :param settings_handler_dict: A dictionary containing additional settings attributes such as
        directories and handler configurations.
    :type settings_handler_dict: dict
    :param dict_settings: Reserved parameter for any additional settings that may be applied or
        evaluated during the function operation.
    :type dict_settings: dict
    :return: A dictionary containing the consolidated test settings, including both those read
        from the file and additional or hardcoded values.
    :rtype: dict
    """

    dict_test = read_settings_to_dict(filepath=join(settings_handler_dict['read_iv_dir'], filename))

    if dict_test['test_type'] == 'STD1':
        dpt_end_frames = (84, 87)  # exactly: (83, 88) but not including 83 and 88
        animate_frames = (15, 190)
    elif dict_test['test_type'] == 'STD2':
        dpt_end_frames = (78, 89)  # exactly: (76, 92) but not including 76 and 92
        animate_frames = (25, 150)
    elif dict_test['test_type'] == 'STD3':
        dpt_end_frames = (28, 34)  # exactly: (25, 36) but not including 25 and 36
        animate_frames = (20, 64)  # two cycles: (20, 64); all cycles: (20, 190)
    elif dict_test['test_type'] == 'VAR3':
        dpt_end_frames = (30, 37)  # exactly: (28, 39) but not including 28 and 39
        animate_frames = (25, 66)  # two cycles: (25, 66); all cycles: (20, 190)
    else:
        raise ValueError('Only implemented test types: [STD1, STD2, STD3, VAR3]')

    dict_test.update({
        'filename': filename,
        'iv_acdc': settings_handler_dict['iv_acdc'],
        'dpt_start_frame': (0, 20),  # exactly: (0, 30)
        'dpt_end_frames': dpt_end_frames,
        'drop_pids': settings_handler_dict['drop_pids'],
        'animate_frames': animate_frames,
    })

    return dict_test


def pre_process_awg(df, amplifier_gain, leading_edge=True, slew_rate=0.001):
    # rename columns for clarity
    x, y = 'dt', 'awg_volt'
    xnew, ynew = 'TIMESTAMP_DAQ_HANDLER', 'VOLT_AWG_INPUT'
    df = df.rename(columns={x: xnew, y: ynew})

    # create column for voltage at output (i.e., post-amplification)
    ynew2 = 'VOLT'
    df[ynew2] = df[ynew] * amplifier_gain

    # generate staircase levels to visualize "continuous" voltage profile
    sampled_time = df[xnew]
    sampled_levels = df[ynew2]
    if leading_edge is True:
        t_steps = sampled_time[1:] - slew_rate
        l_steps = sampled_levels[:-1]
    else:
        t_steps = sampled_time[:-1] + slew_rate
        l_steps = sampled_levels[1:]
    df['TIME'] = pd.Series(sampled_time)
    df2 = pd.DataFrame(np.vstack([l_steps, t_steps]).T, columns=[ynew2, 'TIME'])
    df = pd.concat([df2, df])
    df = df.sort_values('TIME', ascending=True)

    # add column to indicate which voltage "step" each row belongs to
    df = df.reset_index(names='STEP')
    # round 'TIME' to decimals
    df = df.round({'TIME': 3})
    return df

def visualize_staircase_levels(sampled_time, sampled_levels, leading_edge, slew_rate=0.001):
    # add time points to show V(t) in between current sampling times
    if leading_edge is True:
        t_steps = sampled_time[1:] - slew_rate
        l_steps = sampled_levels[:-1]
    else:
        t_steps = sampled_time[:-1] + slew_rate
        l_steps = sampled_levels[1:]
    # concat
    stair_time = np.concatenate((t_steps, sampled_time))
    stair_levels = np.concatenate((l_steps, sampled_levels))
    # sort by time
    stair_time, stair_levels = list(zip(*sorted(zip(stair_time, stair_levels))))

    stair_time = np.array(stair_time)
    stair_levels = np.array(stair_levels)

    return stair_time, stair_levels