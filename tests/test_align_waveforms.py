# tests/test_model_sweep.py
import os
# imports
from os.path import join
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from tifffile import imread
from skimage.transform import rescale

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

from utils.fit import fit_smoothing_spline


def calculate_phase_delay(y1, y2, time):
    """
    Calculates the phase delay between two periodic vectors y1 and y2 in a restricted time range,
    where time_start is the first time point y1 passes through 0 (with a positive gradient),
    and time_end is the last time point y1 passes through 0 (with a positive gradient).

    Parameters:
        y1 (np.ndarray): The first periodic signal.
        y2 (np.ndarray): The second periodic signal.
        time (np.ndarray): The array of time points corresponding to y1 and y2.

    Returns:
        float: The phase delay between y1 and y2 in radians.
    """

    # Ensure inputs are numpy arrays
    y1, y2, time = map(np.asarray, (y1, y2, time))

    # Find indices where y1 crosses zero with a positive gradient
    zero_crossings = np.where((y1[:-1] <= 0) & (y1[1:] > 0))[0]

    # Determine time_start and time_end based on the first and last zero-crossings
    if len(zero_crossings) < 2:  # Ensure we have enough zero crossings
        raise ValueError("Not enough zero crossings found in y1 to determine time range.")

    time_start = time[zero_crossings[0]]
    time_end = time[zero_crossings[-1]]

    # Restrict the data to the time_start < time < time_end range
    time_mask = (time > time_start) & (time < time_end)
    y1_valid = y1[time_mask]
    y2_valid = y2[time_mask]
    time_valid = time[time_mask]

    # Compute the time step (assuming uniform sampling within the restricted range)
    dt = time_valid[1] - time_valid[0]

    # Compute FFT of both signals within the restricted range
    fft_y1 = np.fft.fft(y1_valid)
    fft_y2 = np.fft.fft(y2_valid)

    # Find the dominant frequency index
    dominant_freq_index = np.argmax(np.abs(fft_y1))

    # Extract the phases at the dominant frequency
    phase_y1 = np.angle(fft_y1[dominant_freq_index])
    phase_y2 = np.angle(fft_y2[dominant_freq_index])

    # Calculate the phase delay
    phase_delay = phase_y2 - phase_y1

    # Normalize the phase delay to the range [-pi, pi]
    phase_delay_norm = (phase_delay + np.pi) % (2 * np.pi) - np.pi

    return phase_delay_norm, phase_y2, phase_y1, time_start, time_end

# Define the model function: (A1 + A2 * x) * sin(2 * pi * f * x + phase) + C
def sine_model(x, A1, A2, f, phase, C):
    return (A1 + A2 * x) * np.sin(2 * np.pi * f * x + phase) + C

# Define the model function: (A1 + A2 * |sin(2 * pi * f_mod * x)|) * sin(2 * pi * f * x + phase) + C
def modulated_sine_model(x, A1, A2, f, f_mod, phase, C):
    return (A1 + A2 * np.abs(np.sin(2 * np.pi * f_mod * x))) * np.sin(2 * np.pi * f * x + phase) + C

# ------------------------------------------------------------------------------------------------------------------
# CUSTOM PLOTS - FOR SPECIFIC TESTS OR FOR SPECIFIED PARTICLES

if __name__ == "__main__":

    from scipy.fft import rfft, rfftfreq

    n, T = 100, 0.01  # number of samples and sampling interval
    fcc = (20, 20.5)  # frequencies of sines
    t = np.arange(n) * T
    xx = (np.sin(2 * np.pi * fx_ * t) for fx_ in fcc)  # sine signals

    f = rfftfreq(n, T)  # frequency bins range from 0 Hz to Nyquist freq.
    XX = (rfft(x_) / n for x_ in xx)  # one-sided magnitude spectrum

    fg1, ax1 = plt.subplots(1, 1, tight_layout=True, figsize=(6., 3.))
    ax1.set(title=r"Magnitude Spectrum (no window) of $x(t) = \sin(2\pi f_x t)$ ",
            xlabel=rf"Frequency $f$ in Hertz (bin width $\Delta f = {f[1]}\,$Hz)",
            ylabel=r"Magnitude $|X(f)|/\tau$", xlim=(f[0], f[-1]))
    for X_, fc_, m_ in zip(XX, fcc, ('x-', '.-')):
        ax1.plot(f, abs(X_), m_, label=rf"$f_x={fc_}\,$Hz")

    ax1.grid(True)
    ax1.legend()
    plt.show()

    BASE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation'
    TEST_ID = '03052025_W13-D1_C19-30pT_20+10nmAu'
    TID = 72

    SAVE_DIR = join(BASE_DIR, TEST_ID, 'analyses/custom')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    """FP_READ = join(BASE_DIR, TEST_ID, 'analyses/coords/tid{}_I-V.xlsx'.format(TID))
    DFV = pd.read_excel(FP_READ, sheet_name='data_input')
    DFVM = pd.read_excel(FP_READ, sheet_name='data_output')

    x1 = DFV['TIME'].to_numpy()
    y1 = DFV['VOLT'].to_numpy() / DFV['VOLT'].abs().max()

    x2 = DFVM['MONITOR_TIME'].to_numpy()
    y2 = DFVM['MONITOR_VALUE'].to_numpy() / DFVM['MONITOR_VALUE'].abs().max()"""
    # tid72_merged-coords-volt
    FP_READ = join(BASE_DIR, TEST_ID, 'analyses/coords/tid{}_merged-coords-volt.xlsx'.format(TID))
    DF = pd.read_excel(FP_READ)

    time = DF['t_sync'].to_numpy()
    y1 = DF['VOLT'].to_numpy() / DF['VOLT'].abs().max()
    y2 = DF['MONITOR_VALUE'].to_numpy() / DF['MONITOR_VALUE'].abs().max()

    phase_delay_norm, phase_y2, phase_y1, time_start, time_end = calculate_phase_delay(y2, y1, time)
    phase_delay = phase_y2 - phase_y1

    # --- plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 6))
    ax1.plot(time, y1, '-', label='Input')
    ax1.plot(time, y2, '-', label='Output')
    ax2.plot(time, y1, '-', label='Input')
    ax2.plot(time + phase_delay, y2, '-', label='Output')
    plt.show()

    # To below script works
    """
    x1 = np.linspace(0, 12 * np.pi, 1000)
    y1 = np.sin(x1)

    x2 = x1
    y2 = np.zeros_like(x1)
    a, b, n = 200, 250, 1
    x2_ = np.linspace(n * np.pi, (3 + n) * np.pi, b)
    y2_ = np.sin(x2_)# + np.pi / 2)
    y2[a:a + b] = y2_

    x2 = x1
    y2 = np.sin(x2 + np.pi / 2)

    # Compute FFT of both signals within the restricted range
    fft_y1 = np.fft.fft(y1)
    fft_y2 = np.fft.fft(y2)

    # Find the dominant frequency index
    dominant_freq_index = np.argmax(np.abs(fft_y1))

    # Extract the phases at the dominant frequency
    phase_y1 = np.angle(fft_y1[dominant_freq_index])
    phase_y2 = np.angle(fft_y2[dominant_freq_index])

    # Calculate the phase delay
    phase_delay = phase_y2 - phase_y1

    # --- plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 6))
    ax1.plot(x1, y1, '-', label='Input')
    ax1.plot(x2, y2, '-', label='Output')
    ax2.plot(x1, y1, '-', label='Input')
    ax2.plot(x2 + phase_delay, y2, '-', label='Output')
    plt.show()
    """
    # The below script works
    """
    x1 = np.linspace(0, 12 * np.pi, 1000)
    y1 = np.sin(x1)

    x2 = x1
    y2 = np.zeros_like(x1)
    a, b, n = 200, 250, 1
    x2_ = np.linspace(n * np.pi, (3 + n) * np.pi, b)
    y2_ = np.sin(x2_)# + np.pi / 2)
    y2[a:a + b] = y2_
    # Compute FFT of both signals within the restricted range
    fft_y1 = np.fft.fft(y1)
    fft_y2 = np.fft.fft(y2)

    # Find the dominant frequency index
    dominant_freq_index = np.argmax(np.abs(fft_y1))

    # Extract the phases at the dominant frequency
    phase_y1 = np.angle(fft_y1[dominant_freq_index])
    phase_y2 = np.angle(fft_y2[dominant_freq_index])

    # Calculate the phase delay
    phase_delay = phase_y2 - phase_y1
    # calculate_phase_delay(y1, y2, time)
    # --- plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 6))
    ax1.plot(x1, y1, '-', label='Input')
    ax1.plot(x2, y2, '-', label='Output')
    ax2.plot(x1, y1, '-', label='Input')
    ax2.plot(x2 + phase_delay, y2, '-', label='Output')
    plt.show()
    """
    # The below script works
    """
    x1 = np.linspace(0, 12 * np.pi, 1000)
    y1 = np.sin(x1)

    x2 = x1
    y2 = np.zeros_like(x1)
    a, b, n = 200, 250, 1
    x2_ = np.linspace(n * np.pi, (3 + n) * np.pi, b)
    y2_ = np.sin(x2_)# + np.pi / 2)
    y2[a:a + b] = y2_

    phase_delay_norm, phase_y2, phase_y1, time_start, time_end = calculate_phase_delay(y1, y2, x2)
    phase_delay = phase_y1 - phase_y2

    # --- plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 6))
    ax1.plot(x1, y1, '-', label='Input')
    ax1.plot(x2, y2, '-', label='Output')
    ax2.plot(x1, y1, '-', label='Input')
    ax2.plot(x2 + phase_delay, y2, '-', label='Output')
    plt.show()
    """

