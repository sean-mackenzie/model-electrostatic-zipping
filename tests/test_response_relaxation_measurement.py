# tests/test_model_sweep.py
import os
# imports
from os.path import join
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from tifffile import imread
from skimage.transform import rescale

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

from utils.fit import fit_smoothing_spline


# ------------------------------------------------------------------------------------------------------------------
# CUSTOM PLOTS - FOR SPECIFIC TESTS OR FOR SPECIFIED PARTICLES


def calculate_response_and_relaxation_times_i(df):
    """
    Calculate response and relaxation times from a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 't_sync' (time) and 'dz' (displacement).

    Returns:
    dict: A dictionary with computed response and relaxation times.
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Get the time and displacement array
    time = df['t_sync'].to_numpy()
    displacement = df['dz'].to_numpy()

    # Compute the 10% and 90% of the maximum displacement
    max_displacement = np.max(displacement)
    min_displacement = np.min(displacement)  # Useful for cases with baseline shifts
    threshold_10 = min_displacement + 0.1 * (max_displacement - min_displacement)
    threshold_90 = min_displacement + 0.9 * (max_displacement - min_displacement)

    # Find the times when the displacement crosses 10% and 90% thresholds
    # (Assume monotonic behavior in response and relaxation phases)
    response_start_time = time[np.argmax(displacement >= threshold_10)]  # 10% rising
    response_end_time = time[np.argmax(displacement >= threshold_90)]  # 90% rising
    relaxation_start_time = time[np.argmax(displacement <= threshold_90)]  # 90% falling
    relaxation_end_time = time[np.argmax(displacement <= threshold_10)]  # 10% falling

    # Calculate the response and relaxation times
    response_time = response_end_time - response_start_time
    relaxation_time = relaxation_end_time - relaxation_start_time

    # Return the times in a dictionary
    return {
        'response_time': response_time,
        'relaxation_time': relaxation_time
    }




def calculate_response_and_relaxation_times_2(df):
    """
    Calculate response and relaxation times for multiple rising and falling edges
    in a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with 't_sync' (time) and 'dz' (displacement) columns.

    Returns:
    list: A list of dictionaries containing the response and relaxation times
          for each identified rising or falling edge.
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Get the time and displacement array
    time = df['t_sync'].to_numpy()
    displacement = df['dz'].to_numpy()

    # Subroutine to identify edges (rising or falling)
    def find_edges(displacement):
        """
        Identify the ranges of rising and falling edges in the displacement data.

        Parameters:
        displacement (numpy.ndarray): The displacement data.

        Returns:
        list: A list of tuples indicating the start and end indices of each edge.
              Rising edges: displacement goes up (10% to 90%).
              Falling edges: displacement goes down (90% to 10%).
        """
        edges = []
        increasing = displacement[1:] > displacement[:-1]  # Boolean array of upward trends
        start_idx = None

        for i in range(1, len(displacement)):
            if increasing[i - 1] and start_idx is None:  # Start of a rising edge
                start_idx = i - 1
            elif not increasing[i - 1] and start_idx is not None:  # End of an edge
                edges.append((start_idx, i - 1))
                start_idx = None

        # Handle case where edge goes till the end of the dataset
        if start_idx is not None:
            edges.append((start_idx, len(displacement) - 1))

        return edges

    # Detect rising and falling edges
    rising_edges = find_edges(displacement)
    falling_edges = find_edges(-displacement)  # Reverse for falling trends

    # Subroutine to calculate response/relaxation time for a given range
    def calculate_time_range(start, end, displacement, time, rising=True):
        """
        Calculate the time taken for a 10% to 90% transition (or the reverse).

        Parameters:
        start (int): Start index of the edge.
        end (int): End index of the edge.
        displacement (numpy.ndarray): The displacement array.
        time (numpy.ndarray): The time array.
        rising (bool): Whether it's a rising edge (True) or falling edge (False).

        Returns:
        float: The calculated time duration.
        """
        segment_displacement = displacement[start:end + 1]
        segment_time = time[start:end + 1]

        # Calculate thresholds
        max_disp = np.max(segment_displacement)
        min_disp = np.min(segment_displacement)
        threshold_10 = min_disp + 0.1 * (max_disp - min_disp)
        threshold_90 = min_disp + 0.9 * (max_disp - min_disp)

        if rising:
            # For rising edge: 10% -> 90%
            t_10 = segment_time[np.argmax(segment_displacement >= threshold_10)]
            t_90 = segment_time[np.argmax(segment_displacement >= threshold_90)]
        else:
            # For falling edge: 90% -> 10%
            t_90 = segment_time[np.argmax(segment_displacement <= threshold_90)]
            t_10 = segment_time[np.argmax(segment_displacement <= threshold_10)]

        return t_90 - t_10

    # Calculate times for all rising and falling edges
    results = []
    for start, end in rising_edges:
        response_time = calculate_time_range(start, end, displacement, time, rising=True)
        results.append({'type': 'response', 'start': start, 'end': end, 'duration': response_time})

    for start, end in falling_edges:
        relaxation_time = calculate_time_range(start, end, displacement, time, rising=False)
        results.append({'type': 'relaxation', 'start': start, 'end': end, 'duration': relaxation_time})

    return results


def calculate_response_and_relaxation_times_for_single_edges(df, threshold):
    """
    Calculate response and relaxation times for multiple rising and falling edges in a DataFrame,
    while filtering rising edges based on a given displacement threshold.

    Parameters:
    df (pd.DataFrame): DataFrame with 't_sync' (time) and 'dz' (displacement) columns.
    threshold (float): Minimum required total change in displacement for a rising edge to be considered.

    Returns:
    list: A list of dictionaries containing the response and relaxation times
          for each identified rising or falling edge.
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Get the time and displacement array
    time = df['t_sync'].to_numpy()
    displacement = df['dz'].to_numpy()

    # Subroutine to identify edges (rising or falling)
    def find_edges(displacement, threshold=0, is_rising=True):
        """
        Identify the ranges of rising and falling edges in the displacement data
        and filter them based on a displacement threshold.

        Parameters:
        displacement (numpy.ndarray): The displacement data.
        threshold (float): The minimum required displacement change for an edge to be considered.
        is_rising (bool): Whether to look for rising edges (True) or falling edges (False).

        Returns:
        list: A list of tuples indicating the start and end indices of each edge.
        """
        edges = []
        segment_start = None
        trend = 1 if is_rising else -1  # +1 for rising, -1 for falling
        increasing = (np.diff(displacement) * trend) > 0  # Boolean array for the desired trend

        for i in range(1, len(displacement)):
            if increasing[i - 1]:
                if segment_start is None:
                    segment_start = i - 1
            else:
                if segment_start is not None:
                    # Edge ends here, check if it exceeds threshold
                    segment_end = i - 1
                    if (displacement[segment_end] - displacement[segment_start]) * trend >= threshold:
                        edges.append((segment_start, segment_end))
                    segment_start = None

        # Handle case where the final edge goes to the end of the dataset
        if segment_start is not None:
            segment_end = len(displacement) - 1
            if (displacement[segment_end] - displacement[segment_start]) * trend >= threshold:
                edges.append((segment_start, segment_end))

        return edges

    # Detect significant rising and falling edges
    rising_edges = find_edges(displacement, threshold, is_rising=True)
    falling_edges = find_edges(displacement, threshold, is_rising=False)

    # Subroutine to calculate response/relaxation time for a given range
    def calculate_time_range_deprecated(start, end, displacement, time, rising=True):
        """
        Calculate the time taken for a 10% to 90% transition (or the reverse).

        Parameters:
        start (int): Start index of the edge.
        end (int): End index of the edge.
        displacement (numpy.ndarray): The displacement array.
        time (numpy.ndarray): The time array.
        rising (bool): Whether it's a rising edge (True) or falling edge (False).

        Returns:
        float: The calculated time duration.
        """
        segment_displacement = displacement[start:end + 1]
        segment_time = time[start:end + 1]

        # Calculate thresholds for 10% and 90%
        max_disp = np.max(segment_displacement)
        min_disp = np.min(segment_displacement)
        threshold_10 = min_disp + 0.1 * (max_disp - min_disp)
        threshold_90 = min_disp + 0.9 * (max_disp - min_disp)

        if rising:
            # For rising edge: 10% -> 90%
            t_10 = segment_time[np.argmax(segment_displacement >= threshold_10)]
            t_90 = segment_time[np.argmax(segment_displacement >= threshold_90)]
        else:
            # For falling edge: 90% -> 10%
            t_90 = segment_time[np.argmax(segment_displacement <= threshold_90)]
            t_10 = segment_time[np.argmax(segment_displacement <= threshold_10)]

        return t_90 - t_10

    def calculate_time_range(start, end, displacement, time, rising=True):
        """
        Calculate the time taken for a 10% to 90% transition (or the reverse) using interpolation.

        Parameters:
        start (int): Start index of the edge.
        end (int): End index of the edge.
        displacement (numpy.ndarray): The displacement array.
        time (numpy.ndarray): The time array.
        rising (bool): Whether it's a rising edge (True) or falling edge (False).

        Returns:
        float: The calculated time duration.
        """
        # Extract the segment for displacement and time
        segment_displacement = displacement[start:end + 1]
        segment_time = time[start:end + 1]

        # Calculate thresholds for 10% and 90%
        max_disp = np.max(segment_displacement)  # Maximum displacement in the segment
        min_disp = np.min(segment_displacement)  # Minimum displacement in the segment
        threshold_10 = min_disp + 0.1 * (max_disp - min_disp)
        threshold_90 = min_disp + 0.9 * (max_disp - min_disp)

        # Use interpolation for accurate times at threshold crossings
        def interpolate_time_at_threshold(threshold):
            """
            Interpolate to find the time at which the displacement crosses the given threshold.

            Parameters:
            threshold (float): The displacement value (threshold) to find the closest time for.

            Returns:
            float: Interpolated time when displacement equals the threshold.
            """
            for i in range(len(segment_displacement) - 1):
                if (segment_displacement[i] <= threshold <= segment_displacement[i + 1]) or \
                        (segment_displacement[i] >= threshold >= segment_displacement[i + 1]):
                    # Linear interpolation formula
                    t1, t2 = segment_time[i], segment_time[i + 1]
                    d1, d2 = segment_displacement[i], segment_displacement[i + 1]
                    interpolated_time = t1 + (threshold - d1) * (t2 - t1) / (d2 - d1)
                    return interpolated_time

            # If no crossing is found, return None (unexpected for edge detection)
            return None

        # Interpolate times for 10% and 90% thresholds
        if rising:
            t_10 = interpolate_time_at_threshold(threshold_10)
            t_90 = interpolate_time_at_threshold(threshold_90)
        else:
            t_90 = interpolate_time_at_threshold(threshold_90)
            t_10 = interpolate_time_at_threshold(threshold_10)

        # Check if interpolation returned valid results
        if t_10 is None or t_90 is None:
            raise ValueError("Unable to interpolate times at thresholds for the given segment.")

        # Return the duration from 10% to 90%
        return t_90 - t_10

    # Calculate times for all edges
    results = []
    for start, end in rising_edges:
        response_time = calculate_time_range(start, end, displacement, time, rising=True)
        results.append({'type': 'response', 'start': start, 'end': end, 'duration': response_time})

    for start, end in falling_edges:
        relaxation_time = calculate_time_range(start, end, displacement, time, rising=False)
        results.append({'type': 'relaxation', 'start': start, 'end': end, 'duration': relaxation_time})

    return results


def calculate_response_and_relaxation_times_doesnt_return_minmax_disp(df, threshold):
    """
    Calculate response and relaxation times for multiple rising and falling edges in a DataFrame,
    ensuring that each pair of rising and falling edges share the same min and max displacement values.

    Parameters:
    df (pd.DataFrame): DataFrame with 't_sync' (time) and 'dz' (displacement) columns.
    threshold (float): Minimum required total change in displacement for a rising or falling edge.

    Returns:
    list: A list of dictionaries containing the response and relaxation times
          for each identified rising or falling edge.
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Get the time and displacement array
    time = df['t_sync'].to_numpy()
    displacement = df['dz'].to_numpy()

    # Subroutine to identify edges (rising or falling)
    def find_edges(displacement, threshold=0, is_rising=True):
        """
        Identify the ranges of rising and falling edges in the displacement data
        and filter them based on a displacement threshold.

        Parameters:
        displacement (numpy.ndarray): The displacement data.
        threshold (float): The minimum required displacement change for an edge to be considered.
        is_rising (bool): Whether to look for rising edges (True) or falling edges (False).

        Returns:
        list: A list of tuples indicating the start and end indices of each edge.
        """
        edges = []
        segment_start = None
        trend = 1 if is_rising else -1  # +1 for rising, -1 for falling
        increasing = (np.diff(displacement) * trend) > 0  # Boolean array for the desired trend

        for i in range(1, len(displacement)):
            if increasing[i - 1]:
                if segment_start is None:
                    segment_start = i - 1
            else:
                if segment_start is not None:
                    # Edge ends here, check if it exceeds threshold
                    segment_end = i - 1
                    if (displacement[segment_end] - displacement[segment_start]) * trend >= threshold:
                        edges.append((segment_start, segment_end))
                    segment_start = None

        # Handle case where the final edge goes to the end of the dataset
        if segment_start is not None:
            segment_end = len(displacement) - 1
            if (displacement[segment_end] - displacement[segment_start]) * trend >= threshold:
                edges.append((segment_start, segment_end))

        return edges

    # Detect significant rising and falling edges
    rising_edges = find_edges(displacement, threshold, is_rising=True)
    falling_edges = find_edges(displacement, threshold, is_rising=False)

    # Pair edges (rising first, followed by falling or vice versa)
    paired_edges = []
    while rising_edges and falling_edges:
        if rising_edges[0][0] < falling_edges[0][0]:
            paired_edges.append((rising_edges.pop(0), falling_edges.pop(0)))
        else:
            paired_edges.append((falling_edges.pop(0), rising_edges.pop(0)))

    # Handle any leftover edges
    for rise in rising_edges:
        paired_edges.append((rise, None))
    for fall in falling_edges:
        paired_edges.append((None, fall))

    # Subroutine to calculate response/relaxation time for a given range
    def calculate_time_range(start, end, displacement, time, min_disp, max_disp, rising=True):
        """
        Calculate the time taken for a 10% to 90% transition (or the reverse) using shared min/max values.

        Parameters:
        start (int): Start index of the edge.
        end (int): End index of the edge.
        displacement (numpy.ndarray): The displacement array.
        time (numpy.ndarray): The time array.
        min_disp (float): Minimum displacement for the paired edges.
        max_disp (float): Maximum displacement for the paired edges.
        rising (bool): Whether it's a rising edge (True) or falling edge (False).

        Returns:
        float: The calculated time duration.
        """
        # Extract the segment for displacement and time
        segment_displacement = displacement[start:end + 1]
        segment_time = time[start:end + 1]

        # Calculate thresholds for 10% and 90% using shared min_disp and max_disp
        threshold_10 = min_disp + 0.1 * (max_disp - min_disp)
        threshold_90 = min_disp + 0.9 * (max_disp - min_disp)

        # Interpolation to calculate precise time values at thresholds
        def interpolate_time_at_threshold(threshold):
            for i in range(len(segment_displacement) - 1):
                if (segment_displacement[i] <= threshold <= segment_displacement[i + 1]) or \
                        (segment_displacement[i] >= threshold >= segment_displacement[i + 1]):
                    t1, t2 = segment_time[i], segment_time[i + 1]
                    d1, d2 = segment_displacement[i], segment_displacement[i + 1]
                    return t1 + (threshold - d1) * (t2 - t1) / (d2 - d1)
            return None

        if rising:
            t_10 = interpolate_time_at_threshold(threshold_10)
            t_90 = interpolate_time_at_threshold(threshold_90)
        else:
            t_90 = interpolate_time_at_threshold(threshold_90)
            t_10 = interpolate_time_at_threshold(threshold_10)

        if t_10 is None or t_90 is None:
            raise ValueError("Unable to interpolate times at thresholds for the given segment.")

        return t_90 - t_10

    # Calculate times for paired edges
    results = []
    for rise, fall in paired_edges:
        if rise:
            start, end = rise
            min_disp, max_disp = np.min(displacement[start:end + 1]), np.max(displacement[start:end + 1])

            # Use these min/max values for the falling edge, if it exists
            if fall:
                fall_start, fall_end = fall
                min_disp = min(min_disp, np.min(displacement[fall_start:fall_end + 1]))
                max_disp = max(max_disp, np.max(displacement[fall_start:fall_end + 1]))

            # Rising edge
            response_time = calculate_time_range(start, end, displacement, time, min_disp, max_disp, rising=True)
            results.append({'type': 'response', 'start': start, 'end': end, 'duration': response_time})

        if fall:
            start, end = fall
            # Use the same min/max_disp values shared with the paired rising edge
            relaxation_time = calculate_time_range(start, end, displacement, time, min_disp, max_disp, rising=False)
            results.append({'type': 'relaxation', 'start': start, 'end': end, 'duration': relaxation_time})

    return results


def results_to_dataframe_doesnt_return_minmax_disp(results, df):
    """
    Convert the results of calculate_response_and_relaxation_times into
    a DataFrame, including corresponding 't_sync' and 'dz' values.

    Parameters:
    results (list): Output from calculate_response_and_relaxation_times().
                    A list of dictionaries, each containing 'type', 'start', 'end', and 'duration'.
    df (pd.DataFrame): Original DataFrame with 't_sync' (time) and 'dz' (displacement) columns.

    Returns:
    pd.DataFrame: DataFrame summarizing the results with 'type', 'start', 'end', 'duration',
                  as well as 't_sync_start', 'dz_start', 't_sync_end', and 'dz_end'.
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Initialize a list to store row data
    data = []

    # Iterate through each result entry
    for result in results:
        start_idx = result['start']
        end_idx = result['end']

        # Extract the 't_sync' and 'dz' values at start and end indices
        t_sync_start = df.loc[start_idx, 't_sync']
        dz_start = df.loc[start_idx, 'dz']
        t_sync_end = df.loc[end_idx, 't_sync']
        dz_end = df.loc[end_idx, 'dz']

        # Append the processed row to the data list
        data.append({
            'type': result['type'],  # 'response' or 'relaxation'
            'start': start_idx,
            'end': end_idx,
            'duration': result['duration'],  # Duration from results
            't_sync_start': t_sync_start,
            'dz_start': dz_start,
            't_sync_end': t_sync_end,
            'dz_end': dz_end
        })

    # Convert the data to a DataFrame
    results_df = pd.DataFrame(data)

    return results_df



def calculate_response_and_relaxation_times(df, threshold):
    """
    Calculate response and relaxation times for multiple rising and falling edges in a DataFrame,
    ensuring that each pair of rising and falling edges share the same min and max displacement values.

    Parameters:
    df (pd.DataFrame): DataFrame with 't_sync' (time) and 'dz' (displacement) columns.
    threshold (float): Minimum required total change in displacement for a rising or falling edge.

    Returns:
    list: A list of dictionaries containing the response and relaxation times
          for each identified rising or falling edge, along with min_disp and max_disp.
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Get the time and displacement array
    time = df['t_sync'].to_numpy()
    displacement = df['dz'].to_numpy()

    # Subroutine to identify edges (rising or falling)
    def find_edges(displacement, threshold=0, is_rising=True):
        """
        Identify the ranges of rising and falling edges in the displacement data
        and filter them based on a displacement threshold.

        Parameters:
        displacement (numpy.ndarray): The displacement data.
        threshold (float): The minimum required displacement change for an edge to be considered.
        is_rising (bool): Whether to look for rising edges (True) or falling edges (False).

        Returns:
        list: A list of tuples indicating the start and end indices of each edge.
        """
        edges = []
        segment_start = None
        trend = 1 if is_rising else -1  # +1 for rising, -1 for falling
        increasing = (np.diff(displacement) * trend) > 0  # Boolean array for the desired trend

        for i in range(1, len(displacement)):
            if increasing[i - 1]:
                if segment_start is None:
                    segment_start = i - 1
            else:
                if segment_start is not None:
                    # Edge ends here, check if it exceeds threshold
                    segment_end = i - 1
                    if (displacement[segment_end] - displacement[segment_start]) * trend >= threshold:
                        edges.append((segment_start, segment_end))
                    segment_start = None

        # Handle case where the final edge goes to the end of the dataset
        if segment_start is not None:
            segment_end = len(displacement) - 1
            if (displacement[segment_end] - displacement[segment_start]) * trend >= threshold:
                edges.append((segment_start, segment_end))

        return edges

    # Detect significant rising and falling edges
    rising_edges = find_edges(displacement, threshold, is_rising=True)
    falling_edges = find_edges(displacement, threshold, is_rising=False)

    # Pair edges (rising first, followed by falling or vice versa)
    paired_edges = []
    while rising_edges and falling_edges:
        if rising_edges[0][0] < falling_edges[0][0]:
            paired_edges.append((rising_edges.pop(0), falling_edges.pop(0)))
        else:
            paired_edges.append((falling_edges.pop(0), rising_edges.pop(0)))

    # Handle any leftover edges
    for rise in rising_edges:
        paired_edges.append((rise, None))
    for fall in falling_edges:
        paired_edges.append((None, fall))

    # Subroutine to calculate response/relaxation time for a given range
    def calculate_time_range(start, end, displacement, time, min_disp, max_disp, rising=True):
        """
        Calculate the time taken for a 10% to 90% transition (or the reverse) using shared min/max values.

        Parameters:
        start (int): Start index of the edge.
        end (int): End index of the edge.
        displacement (numpy.ndarray): The displacement array.
        time (numpy.ndarray): The time array.
        min_disp (float): Minimum displacement for the paired edges.
        max_disp (float): Maximum displacement for the paired edges.
        rising (bool): Whether it's a rising edge (True) or falling edge (False).

        Returns:
        float: The calculated time duration.
        """
        # Extract the segment for displacement and time
        segment_displacement = displacement[start:end + 1]
        segment_time = time[start:end + 1]

        # Calculate thresholds for 10% and 90% using shared min_disp and max_disp
        threshold_10 = min_disp + 0.1 * (max_disp - min_disp)
        threshold_90 = min_disp + 0.9 * (max_disp - min_disp)

        # Interpolation to calculate precise time values at thresholds
        def interpolate_time_at_threshold(threshold):
            for i in range(len(segment_displacement) - 1):
                if (segment_displacement[i] <= threshold <= segment_displacement[i + 1]) or \
                        (segment_displacement[i] >= threshold >= segment_displacement[i + 1]):
                    t1, t2 = segment_time[i], segment_time[i + 1]
                    d1, d2 = segment_displacement[i], segment_displacement[i + 1]
                    return t1 + (threshold - d1) * (t2 - t1) / (d2 - d1)
            return None

        if rising:
            t_10 = interpolate_time_at_threshold(threshold_10)
            t_90 = interpolate_time_at_threshold(threshold_90)
        else:
            t_90 = interpolate_time_at_threshold(threshold_90)
            t_10 = interpolate_time_at_threshold(threshold_10)

        if t_10 is None or t_90 is None:
            raise ValueError("Unable to interpolate times at thresholds for the given segment.")

        return t_90 - t_10

    # Calculate times for paired edges
    results = []
    for rise, fall in paired_edges:
        if rise:
            start, end = rise
            min_disp, max_disp = np.min(displacement[start:end + 1]), np.max(displacement[start:end + 1])

            # Use these min/max values for the falling edge, if it exists
            if fall:
                fall_start, fall_end = fall
                min_disp = min(min_disp, np.min(displacement[fall_start:fall_end + 1]))
                max_disp = max(max_disp, np.max(displacement[fall_start:fall_end + 1]))

            # Rising edge
            response_time = calculate_time_range(start, end, displacement, time, min_disp, max_disp, rising=True)
            results.append({
                'type': 'response',
                'start': start,
                'end': end,
                'duration': response_time,
                'min_disp': min_disp,
                'max_disp': max_disp
            })

        if fall:
            start, end = fall
            # Use the same min/max_disp values shared with the paired rising edge
            relaxation_time = calculate_time_range(start, end, displacement, time, min_disp, max_disp, rising=False)
            results.append({
                'type': 'relaxation',
                'start': start,
                'end': end,
                'duration': relaxation_time,
                'min_disp': min_disp,
                'max_disp': max_disp
            })

    return results


def results_to_dataframe(results, df):
    """
    Convert the results of calculate_response_and_relaxation_times into
    a DataFrame, including corresponding 't_sync' and 'dz' values for 'start' and 'end',
    as well as all other values from the results.

    Parameters:
    results (list): Output from calculate_response_and_relaxation_times().
                    A list of dictionaries, each containing 'type', 'start', 'end',
                    'duration', 'min_disp', and 'max_disp'.
    df (pd.DataFrame): Original DataFrame with 't_sync' (time) and 'dz' (displacement) columns.

    Returns:
    pd.DataFrame: DataFrame summarizing the results with all values from the results dictionary,
                  as well as 't_sync_start', 'dz_start', 't_sync_end', and 'dz_end'.
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Initialize a list to store row data
    data = []

    # Iterate through each result entry
    for result in results:
        start_idx = result['start']
        end_idx = result['end']

        # Extract the 't_sync' and 'dz' values at start and end indices
        t_sync_start = df.loc[start_idx, 't_sync']
        dz_start = df.loc[start_idx, 'dz']
        t_sync_end = df.loc[end_idx, 't_sync']
        dz_end = df.loc[end_idx, 'dz']

        if result['type'] == 'response':
            edge_indicator = 1
        elif result['type'] == 'relaxation':
            edge_indicator = -1
        else:
            raise ValueError("Invalid edge type.")

        # Append the processed row to the data list, including all result values
        data.append({
            'type': result['type'],  # 'response' or 'relaxation'
            'edge_dir': edge_indicator,
            'start': start_idx,
            'end': end_idx,
            'duration': np.abs(result['duration']),  # Time duration from results
            'min_disp': result['min_disp'],  # Minimum displacement
            'max_disp': result['max_disp'],  # Maximum displacement
            't_sync_start': t_sync_start,  # Start time
            'dz_start': dz_start,  # Start displacement
            't_sync_end': t_sync_end,  # End time
            'dz_end': dz_end,  # End displacement
            'signed_duration': result['duration'],  # Time duration from results
        })

    # Convert the data to a DataFrame
    results_df = pd.DataFrame(data)

    return results_df



def plot_displacement_with_edges_iter1(df, edges, path_save=None):
    """
    Plot the displacement data from the DataFrame and annotate rising and falling edges.

    Parameters:
    df (pd.DataFrame): DataFrame with 't_sync' (time) and 'dz' (displacement).
    edges (list): List of dictionaries containing rising and falling edges with their start and end indices.

    Returns:
    None
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Extract time and displacement
    time = df['t_sync'].to_numpy()
    displacement = df['dz'].to_numpy()

    # Plot the main displacement trajectory
    plt.figure(figsize=(12, 6))
    plt.plot(time, displacement, label='Displacement', color='blue', lw=2)
    plt.xlabel('Time (t_sync)')
    plt.ylabel('Displacement (dz)')
    plt.title('Displacement with Annotated Rising and Falling Edges')
    plt.grid(alpha=0.4)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Annotate edges
    for edge in edges:
        start, end = edge['start'], edge['end']
        edge_type = edge['type']
        color = 'green' if edge_type == 'response' else 'red'
        label = 'Rising Edge' if edge_type == 'response' else 'Falling Edge'

        # Highlight the segment
        plt.plot(
            time[start:end + 1], displacement[start:end + 1],
            color=color, lw=2, label=f'{label} (#{edges.index(edge) + 1})'
        )
        # Annotate start and end points
        plt.scatter(time[start], displacement[start], color=color, marker='o', s=50, label=f'Start ({label})')
        plt.scatter(time[end], displacement[end], color=color, marker='x', s=50, label=f'End ({label})')

    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    if path_save is not None:
        plt.savefig(path_save, dpi=300, bbox_inches='tight', facecolor='w')
    else:
        plt.show()



def plot_displacement_with_edges_iter2(df, results, path_save=None):
    """
    Plot displacement over time and overlay detected edges with horizontal bars
    representing `min_disp` to `max_disp` for each rising and falling edge pair.

    Parameters:
    df (pd.DataFrame): Original DataFrame with 't_sync' (time) and 'dz' (displacement) columns.
    results (list): Output from calculate_response_and_relaxation_times(), containing details
                    for rising and falling edges including `min_disp` and `max_disp`.

    Returns:
    None
    """
    # Ensure the DataFrame has the required columns
    if not {'t_sync', 'dz'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 't_sync' and 'dz' columns.")

    # Plot displacement over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['t_sync'], df['dz'], '-', color='gray', linewidth=1, zorder=3)
    plt.plot(df['t_sync'], df['dz'], 'o', label='Displacement', color='k', markersize=2, zorder=3.1)

    # Add horizontal bars for all edges in the results
    for result in results:
        start_idx = result['start']
        end_idx = result['end']
        min_disp = result['min_disp']
        max_disp = result['max_disp']
        edge_type = result['type']  # Either 'response' or 'relaxation'

        # Get the start and end times from the DataFrame
        t_sync_start = df.loc[start_idx, 't_sync']
        t_sync_end = df.loc[end_idx, 't_sync']

        # Plot horizontal bars corresponding to min_disp and max_disp
        color = 'green' if edge_type == 'response' else 'red'
        plt.hlines(y=min_disp, xmin=t_sync_start, xmax=t_sync_end, color=color, linestyle='-', label=f"{edge_type} min")
        plt.hlines(y=max_disp, xmin=t_sync_start, xmax=t_sync_end, color=color, linestyle='-', label=f"{edge_type} max")

        plt.fill_between(x=[t_sync_start, t_sync_end], y1=min_disp, y2=max_disp, color=color, alpha=0.2)

    # Customize the plot
    plt.xlabel('Time (t_sync)', fontsize=12)
    plt.ylabel('Displacement (dz)', fontsize=12)
    plt.title('Displacement Over Time with Detected Edges', fontsize=14)
    # plt.legend(loc='upper right')  # Show legend
    plt.grid(alpha=0.125)
    plt.tight_layout()

    if path_save is not None:
        plt.savefig(path_save, dpi=300, bbox_inches='tight', facecolor='w')
    else:
        plt.show()


if __name__ == "__main__":

    BASE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/zipper_paper/Testing/Zipper Actuation'
    TEST_ID = '03122025_W13-D1_C15-15pT_25nmAu'
    TID = 27

    SAVE_DIR = join(BASE_DIR, TEST_ID, 'analyses/custom/response-and-relaxation/tid{}'.format(TID))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    FP_READ = join(BASE_DIR, TEST_ID, 'analyses/coords/tid{}_merged-coords-volt.xlsx'.format(TID))
    DF = pd.read_excel(FP_READ)

    # --- plot displacement trajectories
    ONLY_PIDS = [10, 20, 17, 12, 22, 27]  # if None, then plot all pids
    DF = DF[DF['id'].isin(ONLY_PIDS)]

    for PID in ONLY_PIDS:
        DFPID = DF[DF['id'] == PID].reset_index()
        DFPID['dz'] = DFPID['dz'] * -1

        threshold_dz = DFPID['dz'].abs().max() * 0.5
        results = calculate_response_and_relaxation_times(df=DFPID, threshold=threshold_dz)
        df_results = results_to_dataframe(results, df=DFPID)
        df_results.to_excel(join(SAVE_DIR, 'table_response_and_relaxation_pid{}.xlsx'.format(PID)), index=False)

        # export statistics
        dfg = df_results.drop(columns='type')
        # Create an ExcelWriter object
        with pd.ExcelWriter(join(SAVE_DIR, 'stats_response_and_relaxation_pid{}.xlsx'.format(PID))) as writer:
            # Write each DataFrame to a specific sheet
            dfg[dfg['edge_dir'] == 1].describe().to_excel(writer, sheet_name='rising', index=True)
            dfg[dfg['edge_dir'] == -1].describe().to_excel(writer, sheet_name='falling', index=True)

        plot_displacement_with_edges_iter1(
            df=DFPID,
            edges=results,
            path_save=join(SAVE_DIR, 'fig_response_and_relaxation_pid{}.png'.format(PID)),
        )

        plot_displacement_with_edges_iter2(
            df=DFPID,
            results=results,
            path_save=join(SAVE_DIR, 'fig2_response_and_relaxation_pid{}.png'.format(PID)),
        )

