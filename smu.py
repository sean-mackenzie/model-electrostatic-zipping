import numpy as np
import pandas as pd


def parse_voltage_waveform_from_filename(filename):
    """
    tid, test_type, vmax, dv, step_max = smu.parse_voltage_waveform_from_filename(filename)

    :param filename:
    :return:
    """
    tid = filename.split('tid')[1].split('_test')[0]
    test_type = int(filename.split('tid{}_test'.format(tid))[1].split('_')[0])
    vmax = int(filename.split('tid{}_test{}_'.format(tid, test_type))[1].split('V_')[0])
    dv = float(filename.split('tid{}_test{}_{}V_'.format(tid, test_type, vmax))[1].split('dV')[0])
    step_max = int(np.abs(vmax) / dv)
    return tid, test_type, vmax, dv, step_max

def pre_process_keithley(df, delay, integration, latency=100e-6):
    """
    df = pre_process_keithley_df(df, delay, integration)

    :param df:
    :param delay:
    :param integration:
    :param latency
    :return:
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