import os
from os.path import join
import pandas as pd
import smu


def get_dict_dtypes(name):
    if name == 'settings':
        dict_dtypes = {
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
            'scale_z': float,
            'xyc_pixels': tuple,
            'xyc_microns': tuple,
            'radius_pixels': float,
            'radius_microns': float,
            'andor_keithley_delay': float,
        }
    elif name == 'test':
        dict_dtypes = {
            'tid': int,
            'filename': str,
            'dpt_start_frame': int,
            'dpt_end_frames': tuple,
            'smu_test_type': int,
            'smu_source_delay_time': float,
            'smu_vmax': int,
            'smu_dv': float,
            'smu_step_max': int,
        }
    else:
        raise ValueError('Name not in: [settings, test]')
    return dict_dtypes


def read_settings_to_dict(filepath, name):
    if name not in ['settings', 'test']:
        raise ValueError('Name not in: [settings, test]')

    df = pd.read_excel(filepath, index_col=0)
    ks = df.index.values.tolist()
    vs = df.v.values.tolist()

    dict_settings = {}
    if name == 'settings':
        for k, v in zip(ks, vs):
                if k in ['image_size', 'padding']:
                    dict_settings.update({k: int(v)})
                elif k in ['source_delay_time_by_test', 'xyc_pixels', 'xyc_microns']:
                    dict_settings.update({k: eval(v)})
                else:
                    dict_settings.update({k: float(v)})
        dict_settings = check_dependent_settings(dict_settings, name, filepath)
    elif name == 'test':
        for k, v in zip(ks, vs):
            if k in ['tid', 'dpt_start_frame', 'smu_test_type', 'smu_vmax', 'smu_step_max']:
                dict_settings.update({k: int(v)})
            elif k in ['filename']:
                dict_settings.update({k: str(v)})
            elif k in ['dpt_end_frames']:
                dict_settings.update({k: eval(v)})
            else:
                dict_settings.update({k: float(v)})

    return dict_settings

def check_dependent_settings(dict_settings, name, filepath):
    update_settings = False
    if name == 'settings':
        for k2, k1 in zip(['field_of_view', 'radius_microns'], ['image_size', 'radius_pixels']):
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
    if update_settings:
        write_settings_to_df(s=dict_settings, filepath_save=filepath)
    return dict_settings


def write_settings_to_df(s, filepath_save):
    df = pd.DataFrame.from_dict(data=s, orient='index', columns=['v'])
    df.to_excel(filepath_save, index=True, index_label='k')


def get_settings(fp_settings, name):
    return read_settings_to_dict(filepath=fp_settings, name=name)


def make_test_settings(filename, start_frame, end_frames, dict_settings):
    # parse I-V filename to get Keithley voltage waveform details
    tid, test_type, vmax, dv, step_max = smu.parse_voltage_waveform_from_filename(filename=filename)
    # -
    dict_test = {
        'tid': tid,
        'filename': filename,
        'dpt_start_frame': start_frame,
        'dpt_end_frames': end_frames,
        'smu_test_type': test_type,
        'smu_source_delay_time': dict_settings['source_delay_time_by_test'][test_type],
        'smu_vmax': vmax,
        'smu_dv': dv,
        'smu_step_max': step_max,
    }
    return dict_test


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