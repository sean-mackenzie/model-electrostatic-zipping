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


# ------------------------------------------------------------------------------------------------------------------
# CUSTOM PLOTS - FOR SPECIFIC TESTS OR FOR SPECIFIED PARTICLES


def calculate_phase_delay_full_length(y1, y2, time):
    """
    Calculates the phase delay between two periodic vectors y1 and y2 with respect to a shared
    time vector using the Fast Fourier Transform (FFT).

    Parameters:
        y1 (np.ndarray): The first periodic signal.
        y2 (np.ndarray): The second periodic signal.
        time (np.ndarray): The array of time points corresponding to y1 and y2.

    Returns:
        float: The phase delay between y1 and y2 in radians.
    """
    # Ensure the inputs are numpy arrays
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    time = np.asarray(time)

    # Compute the time step (assuming uniform sampling)
    dt = time[1] - time[0]
    sampling_freq = 1 / dt

    # Compute FFT of both signals
    fft_y1 = np.fft.fft(y1)
    fft_y2 = np.fft.fft(y2)
    freqs = np.fft.fftfreq(len(time), dt)

    # Find the index of the dominant frequency
    dominant_freq_index = np.argmax(np.abs(fft_y1))

    # Extract the phases at the dominant frequency
    phase_y1 = np.angle(fft_y1[dominant_freq_index])
    phase_y2 = np.angle(fft_y2[dominant_freq_index])

    # Calculate the phase delay
    phase_delay = phase_y2 - phase_y1

    # Normalize the phase delay to lie within [-pi, pi]
    phase_delay = (phase_delay + np.pi) % (2 * np.pi) - np.pi

    return phase_delay



def plot_vectors_with_phase_delay_full_length(y1, y2, time, phase_delay):
    """
    Plots two periodic signals y1 and y2 on the same figure and visualizes their phase delay.

    Parameters:
        y1 (np.ndarray): The first periodic signal.
        y2 (np.ndarray): The second periodic signal.
        time (np.ndarray): The shared time array for the signals.
        phase_delay (float): The phase delay between y1 and y2 in radians.
    """
    # Plot the signals
    plt.figure(figsize=(10, 6))
    plt.plot(time, y1, label="Signal y1", linestyle='-', color='blue')
    plt.plot(time, y2, label="Signal y2", linestyle='--', color='red')

    # Highlight the phase delay visually
    # Use the first time where y1 == 0 and y2 == 0 with positive gradients to draw phase delay
    index_y1 = np.where((y1[:-1] <= 0) & (y1[1:] > 0))[0][0]
    index_y2 = np.where((y2[:-1] <= 0) & (y2[1:] > 0))[0][0]

    time_y1 = time[index_y1]
    time_y2 = time[index_y2]

    # Compute the time delay equivalent of the phase delay
    dt = time[1] - time[0]  # Time step
    freq = 1 / (len(time) * dt)  # Approximation based on signal length
    time_delay = phase_delay / (2 * np.pi * freq)  # Convert phase delay to time delay

    # Add an arrow to show the temporal phase delay
    plt.annotate("Phase Delay",
                 xy=(time_y1, 0), xytext=(time_y2, 0),
                 arrowprops=dict(arrowstyle="<->", color='green', lw=2),
                 fontsize=12, color='green')

    # Label the phase delay in radians
    phase_delay_text = f"Phase Delay: {phase_delay:.2f} radians"
    plt.text(0.05, 0.9, phase_delay_text, transform=plt.gca().transAxes, fontsize=12, color='green',
             bbox=dict(facecolor='white', edgecolor='green'))

    # Add labels, title, and legend
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Amplitude")
    plt.title("Periodic Signals with Phase Delay Visualization")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add zero line for better visualization
    plt.legend()
    plt.grid()

    # Show the plot
    plt.tight_layout()
    plt.show()


def calculate_phase_delay_threshold_is_half_max_deflection(y1, y2, time):
    """
    Calculates the phase delay between two periodic vectors y1 and y2 within a time range
    determined by the first and last points where |y1| > 0.5 * max(|y1|).

    Parameters:
        y1 (np.ndarray): The first periodic signal.
        y2 (np.ndarray): The second periodic signal.
        time (np.ndarray): The array of time points corresponding to y1 and y2.

    Returns:
        float: The phase delay between y1 and y2 in radians.
    """

    # Ensure inputs are numpy arrays
    y1, y2, time = map(np.asarray, (y1, y2, time))

    # Get the threshold (half of the maximum absolute amplitude of y1)
    threshold = 0.5 * np.max(np.abs(y1))

    # Find the indices where |y1| exceeds this threshold
    valid_indices = np.where(np.abs(y1) > threshold)[0]

    # Determine time_start and time_end based on the first and last indices
    time_start = time[valid_indices[0]]
    time_end = time[valid_indices[-1]]

    # Apply a mask to restrict the signals and time array to the valid range
    time_mask = (time > time_start) & (time < time_end)
    y1_valid = y1[time_mask]
    y2_valid = y2[time_mask]
    time_valid = time[time_mask]

    # Compute the time step (assuming uniform sampling within the selected range)
    dt = time_valid[1] - time_valid[0]

    # Compute FFT of both signals within the valid range
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
    phase_delay = (phase_delay + np.pi) % (2 * np.pi) - np.pi

    return phase_delay, time_start, time_end



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


def plot_vectors_with_phase_delay(y1, y2, time, phase_delay, time_start, time_end):
    """
    Plots two periodic signals y1 and y2 on the same figure with values restricted to
    the range time_start < time < time_end and visualizes their phase delay.

    Parameters:
        y1 (np.ndarray): The first periodic signal.
        y2 (np.ndarray): The second periodic signal.
        time (np.ndarray): The shared time array for the signals.
        phase_delay (float): The phase delay between y1 and y2 in radians.
        time_start (float): The start time of the range to plot.
        time_end (float): The end time of the range to plot.
    """
    # Restrict the data to the time_start < time < time_end range
    time_mask = (time > time_start) & (time < time_end)
    y1_range = y1[time_mask]
    y2_range = y2[time_mask]
    time_range = time[time_mask]

    # Plot the signals
    plt.figure(figsize=(10, 6))
    plt.plot(time_range, y1_range / np.max(np.abs(y1_range)), label="Signal y1", linestyle='-', color='blue')
    plt.plot(time_range, y2_range / np.max(np.abs(y2_range)), label="Signal y2", linestyle='--', color='red')

    # Highlight the phase delay visually
    # Use the first time values in the range where y1 and y2 are zero-crossing with a positive gradient
    index_y1 = np.where((y1_range[:-1] <= 0) & (y1_range[1:] > 0))[0][0]
    index_y2 = np.where((y2_range[:-1] <= 0) & (y2_range[1:] > 0))[0][0]

    time_y1 = time_range[index_y1]
    time_y2 = time_range[index_y2]

    # Add an arrow to represent the temporal phase delay
    plt.annotate("Phase Delay",
                 xy=(time_y1, 0), xytext=(time_y2, 0),
                 arrowprops=dict(arrowstyle="<->", color='green', lw=2),
                 fontsize=12, color='green')

    # Label the phase delay in radians
    phase_delay_text = f"Phase Delay: {phase_delay:.2f} radians"
    plt.text(0.05, 0.9, phase_delay_text, transform=plt.gca().transAxes, fontsize=12, color='green',
             bbox=dict(facecolor='white', edgecolor='green'))

    # Add labels, title, and legend
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Amplitude")
    plt.title("Periodic Signals with Phase Delay Visualization (Restricted Range)")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add zero line for clarity
    plt.legend()
    plt.grid()

    # Show the plot
    plt.tight_layout()
    plt.show()



def fit_sine_wave(x, y):
    """
    Fits a sine wave with a linearly varying amplitude to the data `y` vs `x`
    and returns the fitted frequency and phase delay.

    Parameters:
        x (np.ndarray): Array of time points.
        y (np.ndarray): Array of amplitudes corresponding to the time points.

    Returns:
        tuple: Fitted frequency (f) in Hz and phase delay (phase) in radians.
    """

    # Initial guesses for the parameters
    A1_initial = (np.max(y) - np.min(y)) / 2  # Amplitude guess
    A2_initial = 0  # Assume amplitude varies slowly
    f_initial = 1 / (x[1] - x[0]) / 10  # Rough frequency guess
    phase_initial = 0  # Phase guess
    C_initial = np.mean(y)  # Offset guess

    initial_guess = [A1_initial, A2_initial, f_initial, phase_initial, C_initial]

    # Perform the curve fitting
    params, _ = curve_fit(sine_model, x, y, p0=initial_guess)

    # Extract the fitted parameters
    A1, A2, f, phase, C = params

    return params


def fit_sine_wave_with_modulation(x, y):
    """
    Fits a sine wave with a periodically varying (triangle-like) amplitude to the data `y` vs `x`
    and returns the fitted frequency and phase delay.

    Parameters:
        x (np.ndarray): Array of time points.
        y (np.ndarray): Array of amplitudes corresponding to the time points.

    Returns:
        tuple: Fitted main frequency (f), phase delay (phase), and modulation frequency (f_mod).
    """

    # Initial guesses for the parameters
    A1_initial = (np.max(y) - np.min(y)) / 2  # Amplitude guess
    A2_initial = 0.1 * A1_initial  # Assume modest modulation depth
    f_initial = 1 / (x[1] - x[0]) / 10  # Rough frequency guess for the main sine wave
    f_mod_initial = f_initial / 10  # Modulation frequency guess (slower than main frequency)
    phase_initial = 0  # Phase guess
    C_initial = np.mean(y)  # Offset guess

    initial_guess = [A1_initial, A2_initial, f_initial, f_mod_initial, phase_initial, C_initial]

    # Perform the curve fitting
    try:
        params, _ = curve_fit(modulated_sine_model, x, y, p0=initial_guess, maxfev=10000)
    except RuntimeError as e:
        raise ValueError("Curve fitting failed: " + str(e)) from e

    # Extract the fitted parameters
    A1, A2, f, f_mod, phase, C = params

    return params


# Define the model function: (A1 + A2 * x) * sin(2 * pi * f * x + phase) + C
def sine_model(x, A1, A2, f, phase, C):
    return (A1 + A2 * x) * np.sin(2 * np.pi * f * x + phase) + C

# Define the model function: (A1 + A2 * |sin(2 * pi * f_mod * x)|) * sin(2 * pi * f * x + phase) + C
def modulated_sine_model(x, A1, A2, f, f_mod, phase, C):
    return (A1 + A2 * np.abs(np.sin(2 * np.pi * f_mod * x))) * np.sin(2 * np.pi * f * x + phase) + C

def plot_pid_dz_vs_monitor_phase_delay(df, pz, pid, test_settings, path_results):
    """

    :param df:
    :param pz:
    :param pid:
    :param path_results:
    :return:
    """
    # hard-coded
    px, py1, py2 = 't_sync', 'dz', 'MONITOR_VALUE'

    df1 = df[(df[px] > 3.3) & (df[px] < 6.3)]
    t1 = df1[px].to_numpy()
    y1 = df1[py1].to_numpy() + df1[py1].abs().max() / 2

    # df = df[(df[px] > 3.3) & (df[px] < 6.3)]
    t2 = df[px].to_numpy()
    y2 = df[py2].to_numpy()

    params1 = fit_sine_wave_with_modulation(x=t1, y=y1)
    params2 = fit_sine_wave_with_modulation(x=t2, y=y2)

    y1new = modulated_sine_model(t1, *params1)
    y2new = modulated_sine_model(t2, *params2)

    plt.plot(t1, y1new, label='y1')
    plt.plot(t2, y2new, label='y2')
    plt.legend()
    plt.show()
    raise ValueError()

    phase_delay_norm, phase_y2, phase_y1, time_start, time_end = calculate_phase_delay(y1, y2, time)
    phase_delay = phase_y2 - phase_y1
    a = 1
    plot_vectors_with_phase_delay(y1, y2, time, phase_delay, time_start, time_end)



if __name__ == "__main__":

    BASE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation'
    TEST_ID = '02252025_W10-A1_C17-20pT'
    TID = 33

    SAVE_DIR = join(BASE_DIR, TEST_ID, 'analyses/representative_test{}/custom'.format(TID))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    FP_READ = join(BASE_DIR, TEST_ID, 'analyses/coords/tid{}_merged-coords-volt.xlsx'.format(TID))
    DF = pd.read_excel(FP_READ)

    # --- plot displacement trajectories
    ONLY_PIDS = [7, 9]  # if None, then plot all pids
    DF = DF[DF['id'].isin(ONLY_PIDS)]

    for PID in ONLY_PIDS:
        DFPID = DF[DF['id'] == PID].reset_index()
        DFPID['dz'] = DFPID['dz'] * -1