from os.path import join
import os
import pandas as pd

from utils import settings


def get_dict_dtypes(name):
    if name == 'awg':
        dict_dtypes = {
            'tid': int,
            'filename': str,
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
    else:
        raise ValueError('Name not in: [awg]')
    return dict_dtypes


def get_dict_dtype_list(data_type):
    if data_type == 'int':
        keys = ['tid']
    elif data_type == 'str':
        keys = ['filename',
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
        keys = ['drop_pids']
    elif data_type == 'special':
        keys = ['awg_mod_ampl_values', 'awg_mod_dwell_values']
    else:
        raise ValueError('Names limited to: [int, string, eval]. No list for floats')
    return keys


if __name__ == "__main__":

    filepath = ('/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation/'
                '03072025_W12-D1_C19-30pT_20+10nmAu/I-V/xlsx/tid61_testSTD1_200V_100HzSQU_data.xlsx')
    name = 'test'
    update_dependent = True

    # def read_settings_to_dict(filepath, name, update_dependent=False):
    if name not in ['settings', 'test']:
        raise ValueError('Name not in: [settings, test]')

    df = pd.read_excel(filepath, index_col=0, sheet_name='settings')
    ks = df.index.values.tolist()
    vs = df[df.columns.values.tolist()[0]].values.tolist()

    dict_settings = {}
    if name == 'test':
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
    else:
        raise ValueError('Name not in: [settings, test]')

    # return dict_settings
    a = 1
