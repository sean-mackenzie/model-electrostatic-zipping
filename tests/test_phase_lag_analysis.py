import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

def calculate_phase_lag(time, voltage, displacement, modulation_frequency):
    """
    Calculates phase lag (in degrees and time) between voltage and displacement signals.

    Parameters:
        time (np.ndarray): Time array (uniformly sampled)
        voltage (np.ndarray): Input voltage signal (e.g., 1 Hz modulated envelope)
        displacement (np.ndarray): Output membrane displacement signal
        modulation_frequency (float): Frequency of modulation envelope [Hz]

    Returns:
        phase_lag_degrees (float): Phase lag in degrees
        time_lag (float): Time lag in seconds
    """

    # Normalize signals
    voltage_norm = (voltage - np.mean(voltage)) / np.std(voltage)
    displacement_norm = (displacement - np.mean(displacement)) / np.std(displacement)

    # Cross-correlation
    corr = correlate(displacement_norm, voltage_norm, mode='full')
    lags = correlation_lags(len(displacement), len(voltage), mode='full')
    time_step = time[1] - time[0]
    time_lags = lags * time_step

    # Find lag with max correlation
    lag_index = np.argmax(corr)
    time_lag = time_lags[lag_index]

    # Convert time lag to phase lag
    period = 1.0 / modulation_frequency
    phase_lag_degrees = (360.0 * time_lag / period) % 360

    return phase_lag_degrees, time_lag, corr, time_lags

# Example usage
if __name__ == "__main__":
    # Simulated example
    t = np.linspace(0, 10, 2000)  # 10 seconds, 2000 samples
    freq = 1  # Hz (modulation frequency)
    voltage = np.sin(2 * np.pi * freq * t)
    displacement = np.sin(2 * np.pi * freq * t - np.pi / 4)  # 45° phase lag

    phase_lag_deg, time_lag_sec, corr, lags = calculate_phase_lag(t, voltage, displacement, modulation_frequency=freq)

    print(f"Phase lag: {phase_lag_deg:.2f} degrees")
    print(f"Time lag: {time_lag_sec*1000:.2f} ms")

    # Plot correlation for inspection
    plt.figure(figsize=(8, 4))
    plt.plot(lags, corr)
    plt.title("Cross-correlation between voltage and displacement")
    plt.xlabel("Lag (s)")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
