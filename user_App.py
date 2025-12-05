import streamlit as st
import os 
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from io import StringIO
import re

warnings.filterwarnings('ignore')

def compute_derivative(i_segment, v_segment):
    """Computes the first derivative (dI/dV)."""
    di_dv = np.diff(i_segment) / np.diff(v_segment)
    v_mid = (v_segment[:-1] + v_segment[1:]) / 2
    return v_mid, di_dv

def moving_average(signal, window_size=5):
    """Applies a moving average filter to a signal."""
    if window_size < 1:
        return signal.copy()
    if window_size % 2 == 0:
        window_size += 1
    
    pad = window_size // 2
    padded = np.pad(signal, pad_width=pad, mode='edge')
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode='valid')

def compute_second_derivative(v_segment, first_derivative):
    """ComputES the second derivative (d²I/dV²)."""
    d2i_dv2 = np.diff(first_derivative) / np.diff(v_segment)
    v_mid2 = (v_segment[:-1] + v_segment[1:]) / 2
    return v_mid2, d2i_dv2

def find_local_minima(signal, num_minima=None):
    """Finds local minima and returns the indices of the top N."""
    indices = argrelextrema(np.asarray(signal), np.less)[0]
    if num_minima is not None and len(indices) > 0:
        sorted_idx = np.argsort(signal[indices])
        indices = indices[sorted_idx[:num_minima]]
    return np.sort(indices)

def find_local_maxima(signal, num_maxima=None):
    """Finds local maxima and returns the indices of the top N."""
    indices = argrelextrema(np.asarray(signal), np.greater)[0]
    if num_maxima is not None and len(indices) > 0:
        sorted_idx = np.argsort(-signal[indices])
        indices = indices[sorted_idx[:num_maxima]]
    return np.sort(indices)

def remove_close_to_global_extrema(signal, v, indices, threshold_v=0.02, threshold_signal=1e-8):
    """Removes potential peaks that are too close to global extrema."""
    indices = np.array(indices, dtype=int)
    if indices.size == 0:
        return indices
    global_min_idx, global_max_idx = np.argmin(signal), np.argmax(signal)
    global_min_v, global_max_v = v[global_min_idx], v[global_max_idx]
    global_min_val, global_max_val = signal[global_min_idx], signal[global_max_idx]

    filtered_indices = [
        idx for idx in indices 
        if not (
            (abs(v[idx] - global_min_v) <= threshold_v or abs(v[idx] - global_max_v) <= threshold_v) or 
            (abs(signal[idx] - global_min_val) < threshold_signal or abs(signal[idx] - global_max_val) < threshold_signal)
        )
    ]
    return np.array(filtered_indices, dtype=int)

def select_minima_by_slope(v, signal, min_indices, max_indices, window_size=0.05):
    """Selects the best minima based on the slope of the signal around the point."""
    candidate_scores = []
    for idx in min_indices:
        v0 = v[idx]
        left_mask = (v >= v0 - window_size) & (v < v0)
        right_mask = (v > v0) & (v <= v0 + window_size)

        if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
            continue

        left_slope = np.polyfit(v[left_mask], signal[left_mask], 1)[0]
        right_slope = np.polyfit(v[right_mask], signal[right_mask], 1)[0]

        if left_slope < 0 and right_slope > 0:
            score = abs(left_slope) + abs(right_slope)
            candidate_scores.append((idx, score, left_slope, right_slope))
            
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in candidate_scores], candidate_scores

def select_top_peaks_by_distance(v, candidate_indices, min_distance, num_peaks):
    """Selects the top peaks based on a minimum distance between them."""
    selected = []
    for idx in candidate_indices:
        if not selected or all(abs(v[idx] - v[s]) >= min_distance for s in selected):
            selected.append(idx)
        if len(selected) >= num_peaks:
            break
    return selected

def find_side_points_raw(v_raw, i_raw, peak_raw_idx, window_length):
    """
    Finds the side points for baseline correction using the automatic window method.
    Returns the indices of the points closest to the window edges in voltage.
    """
    if peak_raw_idx is None or not (0 <= peak_raw_idx < len(v_raw)):
        return None, None
    
    v0 = v_raw[peak_raw_idx]
    v_left, v_right = v0 - window_length / 2, v0 + window_length / 2

    mask_left = (v_raw >= v_left) & (v_raw < v0)
    mask_right = (v_raw > v0) & (v_raw <= v_right)
    
    # Target voltage values for the baseline points
    left_v_target = v_left
    right_v_target = v_right
    
    left_idx = np.where(mask_left)[0][np.argmin(np.abs(v_raw[mask_left] - left_v_target))] if np.any(mask_left) else None
    right_idx = np.where(mask_right)[0][np.argmin(np.abs(v_raw[mask_right] - right_v_target))] if np.any(mask_right) else None
    
    return left_idx, right_idx

def get_baseline_indices(v_segment, left_idx, right_idx):
    """
    Determines the two indices defining the baseline line. 
    It ensures that left_idx corresponds to the starting point (lower V).
    """
    if left_idx is None or right_idx is None:
        return None, None
    
    v_left, v_right = v_segment[left_idx], v_segment[right_idx]
    
    # Identify which of the two selected points is the chronological start/end
    if v_left < v_right:
        start_idx, end_idx = left_idx, right_idx
    else:
        start_idx, end_idx = right_idx, left_idx
        
    return start_idx, end_idx

def apply_baseline_correction(v_segment, i_segment, left_idx, right_idx):
    """
    Applies linear baseline correction by calculating slope from two points 
    and applying the linear function (baseline) across the entire segment.
    """
    i_baseline = i_segment.copy()
    
    if left_idx is None or right_idx is None:
        # If no points are provided, the baseline is the raw current itself.
        return i_segment.copy()

    # Ensure indices are within bounds
    if not (0 <= left_idx < len(v_segment) and 0 <= right_idx < len(v_segment)):
        return i_segment.copy()

    # Determine chronological start and end indices of the selected points
    # This is important for calculating the line equation
    start_idx, end_idx = get_baseline_indices(v_segment, left_idx, right_idx)
    
    x1, x2 = v_segment[start_idx], v_segment[end_idx]
    y1, y2 = i_segment[start_idx], i_segment[end_idx]
    
    if x2 == x1:
        # Avoid division by zero: if V points are the same, use a constant baseline
        slope = 0
        intercept = y1
    else:
        # Compute the slope and intercept
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    
    # Apply the linear function (y = slope * x + intercept) to all voltage points
    i_baseline = slope * v_segment + intercept
    
    return i_baseline

# -----------------------------------------------------------------------------
# ANALYSIS FUNCTION (Unchanged)
# -----------------------------------------------------------------------------
@st.cache_data
def analyze_file(file_content, filename,
                 segment_pattern="A",
                 smooth_window=7, slope_window=0.05,
                 num_minima=5, num_maxima=5, threshold_close=0.02,
                 min_distance_between_minima=0.03,
                 num_peaks_ox=1, num_peaks_red=1,
                 window_length_ox=0.2, window_length_red=0.2,
                 manual_ox_peak_v=None, manual_red_peak_v=None,
                 manual_ox_baseline_v=None, manual_red_baseline_v=None
                 ):
    
    data = pd.read_csv(StringIO(file_content), engine='python').dropna()
    
    try:
        extracted_value = float(filename.split('_')[6])
    except Exception:
        extracted_value = 0.0

    cycle_info = {'voltage': 10, 'current': 11, 'cycle_label': 'Cycle 6'}
    results = []

    voltage = data.iloc[:, cycle_info['voltage']].astype(float).values
    current = data.iloc[:, cycle_info['current']].astype(float).values
    
    if segment_pattern == "A":
        segments = [(0, 109), (110, 210), (211, 317), (318, 399)]
    elif segment_pattern == "B":
        segments = [(0, 99), (100, 199), (200, 299), (300, 399)]
    else:
        st.warning(f"Invalid pattern '{segment_pattern}'. Defaulting to 'A'.")
        segments = [(0, 109), (110, 210), (211, 317), (318, 399)]

    ox_v = np.concatenate([voltage[s:e+1] for s, e in [segments[3], segments[0]]])
    ox_i = np.concatenate([current[s:e+1] for s, e in [segments[3], segments[0]]])
    red_v = np.concatenate([voltage[s:e+1] for s, e in [segments[1], segments[2]]])
    red_i = np.concatenate([current[s:e+1] for s, e in [segments[1], segments[2]]])

    current_window_ox = window_length_ox + (extracted_value * 0.1)
    current_window_red = window_length_red + (extracted_value * 0.1)
    
    # --- Automatic Peak Detection (for potential fallback) ---
    ox_v_der1, ox_deriv1 = compute_derivative(ox_i, ox_v)
    ox_deriv1_smooth = moving_average(ox_deriv1, window_size=smooth_window)
    ox_v_der2, ox_deriv2 = compute_second_derivative(ox_v_der1, ox_deriv1_smooth)
    ox_deriv2_smooth = moving_average(ox_deriv2, window_size=smooth_window)
    ox_min_indices = find_local_minima(ox_deriv2_smooth, num_minima)
    ox_min_indices = remove_close_to_global_extrema(ox_deriv2_smooth, ox_v_der2, ox_min_indices, threshold_v=threshold_close)
    ox_best_minima, _ = select_minima_by_slope(ox_v_der2, ox_deriv2_smooth, ox_min_indices, None, window_size=slope_window)
    ox_final_selected = select_top_peaks_by_distance(ox_v_der2, ox_best_minima, min_distance_between_minima, num_peaks_ox)
    
    red_v_der1, red_deriv1 = compute_derivative(red_i, red_v)
    red_deriv1_smooth = moving_average(red_deriv1, window_size=smooth_window)
    red_v_der2, red_deriv2 = compute_second_derivative(red_v_der1, red_deriv1_smooth)
    red_deriv2_smooth = moving_average(-1 * red_deriv2, window_size=smooth_window)
    red_min_indices = find_local_minima(red_deriv2_smooth, num_minima)
    red_min_indices = remove_close_to_global_extrema(red_deriv2_smooth, red_v_der2, red_min_indices, threshold_v=threshold_close)
    red_best_minima, _ = select_minima_by_slope(red_v_der2, red_deriv2_smooth, red_min_indices, None, window_size=slope_window)
    red_final_selected = select_top_peaks_by_distance(red_v_der2, red_best_minima, min_distance_between_minima, num_peaks_red)
    
    # --- Peak Selection Logic (Manual Override) ---
    ox_raw_peak_idx, red_raw_peak_idx = None, None
    if manual_ox_peak_v is not None:
        ox_raw_peak_idx = np.argmin(np.abs(ox_v - manual_ox_peak_v))
    elif ox_final_selected:
        raw_idxs = [np.argmin(np.abs(ox_v - ox_v_der2[idx])) for idx in ox_final_selected]
        ox_raw_peak_idx = raw_idxs[np.argmax([abs(ox_i[r]) for r in raw_idxs])]

    if manual_red_peak_v is not None:
        red_raw_peak_idx = np.argmin(np.abs(red_v - manual_red_peak_v))
    elif red_final_selected:
        raw_idxs = [np.argmin(np.abs(red_v - red_v_der2[idx])) for idx in red_final_selected]
        red_raw_peak_idx = raw_idxs[np.argmax([abs(red_i[r]) for r in raw_idxs])]
    
    # --- Baseline Selection Logic (Manual Override) ---
    ox_side_raw = None
    if ox_raw_peak_idx is not None:
        if manual_ox_baseline_v is not None and len(manual_ox_baseline_v) == 2:
            # Manual baseline selection
            left_v, right_v = manual_ox_baseline_v
            left_idx = np.argmin(np.abs(ox_v - left_v))
            right_idx = np.argmin(np.abs(ox_v - right_v))
            ox_side_raw = (left_idx, right_idx)
        else:
            # Automatic baseline selection
            ox_side_raw = find_side_points_raw(ox_v, ox_i, ox_raw_peak_idx, current_window_ox)

    red_side_raw = None
    if red_raw_peak_idx is not None:
        if manual_red_baseline_v is not None and len(manual_red_baseline_v) == 2:
            # Manual baseline selection
            left_v, right_v = manual_red_baseline_v
            left_idx = np.argmin(np.abs(red_v - left_v))
            right_idx = np.argmin(np.abs(red_v - right_v))
            red_side_raw = (left_idx, right_idx)
        else:
            # Automatic baseline selection
            red_side_raw = find_side_points_raw(red_v, red_i, red_raw_peak_idx, current_window_red)
    
    # --- Baseline Correction & Results ---
    
    # Use the MODIFIED apply_baseline_correction function
    ox_i_corrected = apply_baseline_correction(ox_v, ox_i, ox_side_raw[0] if ox_side_raw else None, ox_side_raw[1] if ox_side_raw else None)
    red_i_corrected = apply_baseline_correction(red_v, red_i, red_side_raw[0] if red_side_raw else None, red_side_raw[1] if red_side_raw else None)
    
    ox_i_subtracted = ox_i - ox_i_corrected
    red_i_subtracted = red_i - red_i_corrected
    
    ox_subtracted_peak = ox_i_subtracted[ox_raw_peak_idx] if ox_raw_peak_idx is not None else None
    red_subtracted_peak = red_i_subtracted[red_raw_peak_idx] if red_raw_peak_idx is not None else None
    
    results.append({
        "cycle_label": cycle_info['cycle_label'], "filename": filename, "scan_rate": extracted_value,
        "ox_v": ox_v, "ox_i": ox_i, "red_v": red_v, "red_i": red_i,
        "ox_i_corrected": ox_i_corrected, "red_i_corrected": red_i_corrected,
        "ox_i_subtracted": ox_i_subtracted, "red_i_subtracted": red_i_subtracted,
        "ox_raw_peak_idx": ox_raw_peak_idx, "red_raw_peak_idx": red_raw_peak_idx,
        "ox_v_der2": ox_v_der2, "ox_deriv2_smooth": ox_deriv2_smooth,
        "red_v_der2": red_v_der2, "red_deriv2_smooth": red_deriv2_smooth,
        "ox_final_selected": ox_final_selected, "red_final_selected": red_final_selected,
        "ox_subtracted_peak": ox_subtracted_peak, "red_subtracted_peak": red_subtracted_peak,
        "window_length_ox": current_window_ox, "window_length_red": current_window_red,
        "ox_baseline_pts": ox_side_raw, 
        "red_baseline_pts": red_side_raw
    })
    return results

# -----------------------------------------------------------------------------
# MODIFIED PLOTTING FUNCTION
# -----------------------------------------------------------------------------

def create_interactive_plot(all_cycles_res, **kwargs):
    if not all_cycles_res:
        return go.Figure().update_layout(title_text="No cycles found or analysis failed.")
    
    res = all_cycles_res[0]

    # Create a 2x2 grid for the four requested plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('1. Raw Cyclic Voltammogram (I vs V)', '2. Second Derivative (d²I/dV²)', 
                        '3. Peak Detection and Baseline', '4. Baseline Subtracted Signal (I_sub vs V)'),
        vertical_spacing=0.1, horizontal_spacing=0.1
    )

    # --- Plot 1: Raw Data ---
    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i"], mode='lines', name='Oxidation Raw', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i"], mode='lines', name='Reduction Raw', line=dict(color='blue')), row=1, col=1)
    
    # --- Plot 2: Second Derivative ---
    fig.add_trace(go.Scatter(x=res["ox_v_der2"], y=res["ox_deriv2_smooth"], mode='lines', name='Ox d2', line=dict(color='red'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=res["red_v_der2"], y=res["red_deriv2_smooth"], mode='lines', name='Red d2', line=dict(color='blue'), showlegend=False), row=1, col=2)
    if res["ox_final_selected"]:
        fig.add_trace(go.Scatter(x=[res["ox_v_der2"][i] for i in res["ox_final_selected"]], y=[res["ox_deriv2_smooth"][i] for i in res["ox_final_selected"]], mode='markers', name='Auto Ox Peaks d2', marker=dict(color='darkred', size=8, symbol='circle')), row=1, col=2)
    if res["red_final_selected"]:
        fig.add_trace(go.Scatter(x=[res["red_v_der2"][i] for i in res["red_final_selected"]], y=[res["red_deriv2_smooth"][i] for i in res["red_deriv2_smooth"]], mode='markers', name='Auto Red Peaks d2', marker=dict(color='navy', size=8, symbol='circle')), row=1, col=2)
    
    # --- Plot 3: Peak Detection & Baseline ---
    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i"], mode='lines', name='Ox raw', line=dict(color='red', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i"], mode='lines', name='Red raw', line=dict(color='blue', width=1)), row=2, col=1)
    
    # Oxidation Peak and Baseline Points/Line
    if res["ox_raw_peak_idx"] is not None:
        idx = res["ox_raw_peak_idx"]
        fig.add_trace(go.Scatter(x=[res["ox_v"][idx]], y=[res["ox_i"][idx]], mode='markers', name='Ox Peak', marker=dict(color='red', size=10, symbol='star')), row=2, col=1)
        left, right = res["ox_baseline_pts"]
        if left is not None and right is not None:
            # Baseline line (spanning the entire segment range)
            fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i_corrected"], mode='lines', name='Ox Baseline Line', line=dict(color='black', width=3, dash='dot'), showlegend=False), row=2, col=1)
            # Two selected points (markers)
            baseline_x = [res["ox_v"][left], res["ox_v"][right]]
            baseline_y = [res["ox_i"][left], res["ox_i"][right]]
            fig.add_trace(go.Scatter(x=baseline_x, y=baseline_y, mode='markers', name='Ox Baseline Pts', marker=dict(color='black', size=8)), row=2, col=1)
    
    # Reduction Peak and Baseline Points/Line
    if res["red_raw_peak_idx"] is not None:
        idx = res["red_raw_peak_idx"]
        fig.add_trace(go.Scatter(x=[res["red_v"][idx]], y=[res["red_i"][idx]], mode='markers', name='Red Peak', marker=dict(color='blue', size=10, symbol='star')), row=2, col=1)
        left, right = res["red_baseline_pts"]
        if left is not None and right is not None:
            # Baseline line (spanning the entire segment range)
            fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i_corrected"], mode='lines', name='Red Baseline Line', line=dict(color='black', width=3, dash='dot'), showlegend=False), row=2, col=1)
            # Two selected points (markers)
            baseline_x = [res["red_v"][left], res["red_v"][right]]
            baseline_y = [res["red_i"][left], res["red_i"][right]]
            fig.add_trace(go.Scatter(x=baseline_x, y=baseline_y, mode='markers', name='Red Baseline Pts', marker=dict(color='black', size=8), showlegend=False), row=2, col=1)
    
    # --- Plot 4: Subtracted Signal ---
    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i_subtracted"], mode='lines', name='Ox Subtracted', line=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i_subtracted"], mode='lines', name='Red Subtracted', line=dict(color='blue')), row=2, col=2)
    # Highlight the resulting I_peak
    if res["ox_raw_peak_idx"] is not None:
        idx = res["ox_raw_peak_idx"]
        fig.add_trace(go.Scatter(x=[res["ox_v"][idx]], y=[res["ox_i_subtracted"][idx]], mode='markers', name='Ox Sub Peak', marker=dict(color='darkred', size=10)), row=2, col=2)
    if res["red_raw_peak_idx"] is not None:
        idx = res["red_raw_peak_idx"]
        fig.add_trace(go.Scatter(x=[res["red_v"][idx]], y=[res["red_i_subtracted"][idx]], mode='markers', name='Red Sub Peak', marker=dict(color='darkblue', size=10)), row=2, col=2)
    
    # Update layout and axes
    fig.update_layout(height=800, showlegend=True, title_text=f"CV Analysis for: {res['filename']} - {res['cycle_label']} (Scan Rate: {res['scan_rate']} V/s)")
    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_xaxes(title_text="Voltage (V)", row=r, col=c)
            fig.update_yaxes(title_text="Current (mA)", row=r, col=c)
    
    # Manually adjust y-axis label for d2I/dV2 plot since it's not strictly current
    fig.update_yaxes(title_text="d²I/dV² (mA/V²)", row=1, col=2)
    
    return fig

# -----------------------------------------------------------------------------
# REMAINDER OF STREAMLIT UI (Unchanged)
# -----------------------------------------------------------------------------

def create_interactive_summary_plots(df):
    """Creates interactive Plotly graphs for peaks vs scan rate and sqrt(scan rate)."""
    # Sort by scan rate to ensure lines are drawn in the correct order
    df = df.sort_values('Scan Rate (V/s)')

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('All Peaks vs Scan Rate', 'All Peaks vs sqrt(Scan Rate)', 'Log(I_peak) vs Log(Scan Rate)'),
        specs=[[{}, {}], [{'colspan': 2}, None]],
        horizontal_spacing=0.1, vertical_spacing=0.15
    )

    avg_df = df.groupby('Scan Rate (V/s)').agg({'Ox Subtracted Peak': 'mean', 'Red Subtracted Peak': 'mean'}).reset_index()
    avg_df['sqrt(Scan Rate)'] = np.sqrt(avg_df['Scan Rate (V/s)'])
    df['sqrt(Scan Rate)'] = np.sqrt(df['Scan Rate (V/s)'])
    
    epsilon = 1e-12 
    avg_df['log_scan_rate'] = np.log10(avg_df['Scan Rate (V/s)'])
    avg_df['log_ox_peak'] = np.log10(np.abs(avg_df['Ox Subtracted Peak']) + epsilon)
    avg_df['log_red_peak'] = np.log10(np.abs(avg_df['Red Subtracted Peak']) + epsilon)

    ox_label, red_label = "log(Avg Ox Peak)", "log(Avg Red Peak)"
    if len(avg_df) > 1:
        finite_ox_mask = np.isfinite(avg_df['log_scan_rate']) & np.isfinite(avg_df['log_ox_peak'])
        finite_red_mask = np.isfinite(avg_df['log_scan_rate']) & np.isfinite(avg_df['log_red_peak'])

        if finite_ox_mask.sum() > 1:
            ox_slope = stats.linregress(avg_df.loc[finite_ox_mask, 'log_scan_rate'], avg_df.loc[finite_ox_mask, 'log_ox_peak']).slope
            ox_label = f'log(Avg Ox Peak) | Slope: {ox_slope:.3f}'
        if finite_red_mask.sum() > 1:
            red_slope = stats.linregress(avg_df.loc[finite_red_mask, 'log_scan_rate'], avg_df.loc[finite_red_mask, 'log_red_peak']).slope
            red_label = f'log(Avg Red Peak) | Slope: {red_slope:.3f}'

    fig.add_trace(go.Scatter(x=df["Scan Rate (V/s)"], y=df["Ox Subtracted Peak"], mode='lines+markers', name='Ox Peaks', marker=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Scan Rate (V/s)"], y=df["Red Subtracted Peak"], mode='lines+markers', name='Red Peaks', marker=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["sqrt(Scan Rate)"], y=df["Ox Subtracted Peak"], mode='lines+markers', name='Ox Peaks', marker=dict(color='red'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["sqrt(Scan Rate)"], y=df["Red Subtracted Peak"], mode='lines+markers', name='Red Peaks', marker=dict(color='blue'), showlegend=False), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=avg_df["log_scan_rate"], y=avg_df["log_ox_peak"], mode='lines+markers', name=ox_label, line=dict(color='red', dash='dot'), marker=dict(symbol='star', color='darkred', size=8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=avg_df["log_scan_rate"], y=avg_df["log_red_peak"], mode='lines+markers', name=red_label, line=dict(color='blue', dash='dot'), marker=dict(symbol='star', color='darkblue', size=8)), row=2, col=1)

    fig.update_layout(title_text="Peak Current Summary Across All Files", height=900, showlegend=True)
    fig.update_xaxes(title_text="Scan Rate (V/s)", row=1, col=1); fig.update_yaxes(title_text="Peak Current (mA)", row=1, col=1)
    fig.update_xaxes(title_text="√Scan Rate (V/s)⁰⁵", row=1, col=2); fig.update_yaxes(title_text="Peak Current (mA)", row=1, col=2)
    fig.update_xaxes(title_text="log(Scan Rate)", row=2, col=1); fig.update_yaxes(title_text="log(|Peak Current|)", row=2, col=1)
    return fig

# -----------------------------------------------------------------------------
# STREAMLIT USER INTERFACE (Unchanged)
# -----------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🔬 Interactive Cyclic Voltammetry Analyzer")
st.markdown("Upload your CSV files, adjust parameters for each file, and generate a summary analysis.")

if 'fixed_params' not in st.session_state:
    st.session_state.fixed_params = {}
if 'manual_params' not in st.session_state:
    st.session_state.manual_params = {}
if 'confirmed_files' not in st.session_state:
    st.session_state.confirmed_files = set()


st.sidebar.header("Controls")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files", type=['csv'], accept_multiple_files=True
)

if uploaded_files:
    file_dict = {f.name: f for f in uploaded_files}
    file_names = sorted(file_dict.keys())
    
    selected_filename = st.sidebar.selectbox("Select a file to analyze:", file_names)
    
    # Initialize/Load parameters
    fixed_params_for_file = st.session_state.fixed_params.get(selected_filename, {}).copy()
    manual_params_for_file = st.session_state.manual_params.get(selected_filename, {}).copy()

    st.sidebar.subheader("Analysis Parameters (Auto)")

    default_pattern = fixed_params_for_file.get('segment_pattern', 'A')
    default_index = 0 if default_pattern == 'A' else 1
    segment_pattern = st.sidebar.radio(
        "Segmentation Pattern", ("A", "B"), 
        index=default_index,
        help="Select the data point segmentation pattern."
    )

    smooth_window = st.sidebar.slider('Smooth Window', 3, 21, fixed_params_for_file.get('smooth_window', 11), 2)
    slope_window = st.sidebar.slider('Slope Window', 0.01, 0.5, fixed_params_for_file.get('slope_window', 0.1), 0.01)
    min_distance_between_minima = st.sidebar.slider('Min Peak Distance (V)', 0.01, 0.2, fixed_params_for_file.get('min_distance_between_minima', 0.05), 0.01)
    window_length_ox = st.sidebar.slider('Auto Ox Window Length', 0.1, 1.0, fixed_params_for_file.get('window_length_ox', 0.6), 0.01)
    window_length_red = st.sidebar.slider('Auto Red Window Length', 0.1, 1.0, fixed_params_for_file.get('window_length_red', 0.4), 0.01)

    # --- NEW MANUAL PARAMETER SELECTION UI ---
    st.sidebar.subheader("Manual Overrides")

    # 1. Allow the user to select the peak (Oxidation)
    manual_ox_peak_v = st.sidebar.number_input('Oxidation Peak Voltage (V)', 
                                               value=manual_params_for_file.get('manual_ox_peak_v', None), 
                                               format="%.4f", step=0.01, 
                                               help="Enter a voltage value to manually select the oxidation peak.")
    
    # 1. Allow the user to select the peak (Reduction)
    manual_red_peak_v = st.sidebar.number_input('Reduction Peak Voltage (V)', 
                                                value=manual_params_for_file.get('manual_red_peak_v', None), 
                                                format="%.4f", step=0.01, 
                                                help="Enter a voltage value to manually select the reduction peak.")

    # 2. Allow the user to select the starting point of the baseline (Oxidation - using two points for a line)
    st.sidebar.markdown('**Oxidation Baseline Points (V)**')
    manual_ox_baseline_v_saved = manual_params_for_file.get('manual_ox_baseline_v', [None, None])
    manual_ox_baseline_v1 = st.sidebar.number_input('Ox Baseline Point 1 V', 
                                                    value=manual_ox_baseline_v_saved[0],
                                                    format="%.4f", step=0.01)
    manual_ox_baseline_v2 = st.sidebar.number_input('Ox Baseline Point 2 V', 
                                                    value=manual_ox_baseline_v_saved[1],
                                                    format="%.4f", step=0.01)
    
    # 2. Allow the user to select the starting point of the baseline (Reduction - using two points for a line)
    st.sidebar.markdown('**Reduction Baseline Points (V)**')
    manual_red_baseline_v_saved = manual_params_for_file.get('manual_red_baseline_v', [None, None])
    manual_red_baseline_v1 = st.sidebar.number_input('Red Baseline Point 1 V', 
                                                     value=manual_red_baseline_v_saved[0],
                                                     format="%.4f", step=0.01)
    manual_red_baseline_v2 = st.sidebar.number_input('Red Baseline Point 2 V', 
                                                     value=manual_red_baseline_v_saved[1],
                                                     format="%.4f", step=0.01)
    
    # Clean up manual inputs for passing to the function
    manual_ox_baseline_v = [manual_ox_baseline_v1, manual_ox_baseline_v2] if manual_ox_baseline_v1 is not None and manual_ox_baseline_v2 is not None else None
    manual_red_baseline_v = [manual_red_baseline_v1, manual_red_baseline_v2] if manual_red_baseline_v1 is not None and manual_red_baseline_v2 is not None else None

    # Compile current parameters
    current_params = {
        'segment_pattern': segment_pattern, 
        'smooth_window': smooth_window, 'slope_window': slope_window,
        'num_minima': fixed_params_for_file.get('num_minima', 100), 'num_maxima': fixed_params_for_file.get('num_maxima', 100),
        'threshold_close': fixed_params_for_file.get('threshold_close', 0.06), 'min_distance_between_minima': min_distance_between_minima,
        'window_length_ox': window_length_ox, 'window_length_red': window_length_red,
        'num_peaks_ox': 1, 'num_peaks_red': 1
    }

    current_manual_params = {
        'manual_ox_peak_v': manual_ox_peak_v,
        'manual_red_peak_v': manual_red_peak_v,
        'manual_ox_baseline_v': manual_ox_baseline_v,
        'manual_red_baseline_v': manual_red_baseline_v,
    }

    # Save fixed parameters
    if st.sidebar.button("📌 Fix Automatic Parameters for this File"):
        st.session_state.fixed_params[selected_filename] = current_params
        st.sidebar.success(f"Automatic parameters fixed for {selected_filename}!")

    # Save manual overrides
    if st.sidebar.button("💾 Save Manual Overrides for this File"):
        st.session_state.manual_params[selected_filename] = current_manual_params
        st.sidebar.success(f"Manual overrides saved for {selected_filename}!")
    
    # 3. Ask the user to confirm
    is_confirmed = selected_filename in st.session_state.confirmed_files
    
    st.sidebar.markdown("---")
    if is_confirmed:
        st.sidebar.success(f"✅ Analysis for `{selected_filename}` is **CONFIRMED**.")
        if st.sidebar.button("🔄 Un-confirm Analysis"):
            st.session_state.confirmed_files.remove(selected_filename)
            st.rerun()
    else:
        if st.sidebar.button("👉 Confirm Analysis for Full Summary"):
            st.session_state.confirmed_files.add(selected_filename)
            st.rerun()
        st.sidebar.info("Confirm to include this file's results in the 'Full Data Summary' plot.")
    st.sidebar.markdown("---")
    
    # Merge all parameters for the current run
    run_params = current_params.copy()
    run_params.update(current_manual_params)


    tab1, tab2 = st.tabs(["Single File Analysis", "Full Data Summary"])

    with tab1:
        st.header(f"Analysis for: `{selected_filename}`")
        selected_file_content = file_dict[selected_filename].getvalue().decode("utf-8")
        
        # 4. Proceed with the analysis and plotting
        analysis_results = analyze_file(selected_file_content, selected_filename, **run_params)
        fig_single = create_interactive_plot(analysis_results, **run_params)
        st.plotly_chart(fig_single, use_container_width=True)

    with tab2:
        st.header("Summary of All Confirmed Files")
        if not st.session_state.confirmed_files:
             st.warning("No files confirmed yet. Please confirm a file in the sidebar to include it in the summary.")

        if st.button("📊 Generate Full Summary Plot (Confirmed Files Only)"):
            all_summary_data = []
            confirmed_file_names = [f for f in file_names if f in st.session_state.confirmed_files]
            
            if not confirmed_file_names:
                st.warning("No files confirmed. Please confirm at least one file to generate the summary.")
            else:
                progress_bar = st.progress(0, text="Analyzing confirmed files...")
                for i, filename in enumerate(confirmed_file_names):
                    file_content = file_dict[filename].getvalue().decode("utf-8")
                    
                    # Use the fixed/saved parameters for this file
                    params = st.session_state.fixed_params.get(filename, current_params)
                    manual_p = st.session_state.manual_params.get(filename, {})
                    final_params = params.copy()
                    final_params.update(manual_p)
                    
                    res_list = analyze_file(file_content, filename, **final_params)
                    
                    if res_list:
                        res = res_list[0]
                        all_summary_data.append({
                            "File": res["filename"], "Cycle": res["cycle_label"],
                            "Scan Rate (V/s)": res["scan_rate"],
                            "Ox Subtracted Peak": res["ox_subtracted_peak"],
                            "Red Subtracted Peak": res["red_subtracted_peak"]
                        })
                    progress_bar.progress((i + 1) / len(confirmed_file_names), text=f"Analyzing {filename}...")
                
                progress_bar.empty()
                if all_summary_data:
                    summary_df = pd.DataFrame(all_summary_data)
                    fig_summary = create_interactive_summary_plots(summary_df)
                    st.plotly_chart(fig_summary, use_container_width=True)
                    st.subheader("Summary Data Table")
                    st.dataframe(summary_df)
                else:
                    st.warning("Could not generate summary data for confirmed files. Check files or parameters.")
else:
    st.info("Please upload one or more CSV files to begin.")
