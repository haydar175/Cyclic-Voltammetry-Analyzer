import streamlit as st
import os  # <-- THIS IMPORT IS ADDED
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from io import StringIO
import re


# Ignore common warnings for a cleaner output
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# MODIFIED HELPER FUNCTION
# -----------------------------------------------------------------------------

def extract_scan_rate_from_filename(filename):
    """
    Tries to find the scan rate from a filename based on a specific list.
    
    Checks if one of the last 4 parts of the filename (split by _, -, or space)
    is a match in a predefined list of scan rates.
    """
    
    # 1. Define the set of valid scan rate strings, exactly as requested.
    SCAN_RATE_CANDIDATES = {
        '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.9', 
        '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0'
    }
    
    try:
        # 2. Get filename parts (remove extension first)
        filename_without_ext, _ = os.path.splitext(filename)
        
        # 3. Split by underscore, hyphen, or space
        parts = re.split(r'[_ -]', filename_without_ext)
        
        # 4. Get the last 4 parts
        last_four_parts = parts[-4:]
        
        # 5. Check for matches, starting from the end
        for part in reversed(last_four_parts):
            if part in SCAN_RATE_CANDIDATES:
                return float(part) # Found it
                
    except Exception:
        pass # If any part of this fails, just fall through to the default
    
    # 6. If no match is found, return 0.0
    return 0.0

# -----------------------------------------------------------------------------
# CORE ANALYSIS FUNCTIONS (Unchanged logic, modified signature)
# -----------------------------------------------------------------------------

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
    """Computes the second derivative (d²I/dV²)."""
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
    """Finds the side points for baseline correction."""
    if peak_raw_idx is None or not (0 <= peak_raw_idx < len(v_raw)):
        return None, None
    
    v0, i0 = v_raw[peak_raw_idx], i_raw[peak_raw_idx]
    v_left, v_right = v0 - window_length / 2, v0 + window_length / 2

    mask_left = (v_raw >= v_left) & (v_raw < v0)
    mask_right = (v_raw > v0) & (v_raw <= v_right)

    if not (np.any(mask_left) and np.any(mask_right)):
        return None, None
    
    left_idx = np.where(mask_left)[0][np.argmax(np.abs(i_raw[mask_left] - i0))]
    right_idx = np.where(mask_right)[0][np.argmax(np.abs(i_raw[mask_right] - i0))]
    return left_idx, right_idx

def apply_baseline_correction(v_segment, i_segment, left_idx, right_idx):
    """Applies linear baseline correction and returns the composite signal."""
    corrected_i = i_segment.copy()
    
    if left_idx is None or right_idx is None:
        return corrected_i

    x_left, x_right = v_segment[left_idx], v_segment[right_idx]
    y_left, y_right = i_segment[left_idx], i_segment[right_idx]
    
    if x_right == x_left:
        return corrected_i
    
    slope = (y_right - y_left) / (x_right - x_left)
    intercept = y_left - slope * x_left
    
    start_v, end_v = min(x_left, x_right), max(x_left, x_right)
    mask = (v_segment >= start_v) & (v_segment <= end_v)
    
    corrected_i[mask] = slope * v_segment[mask] + intercept
        
    return corrected_i

# -----------------------------------------------------------------------------
# MODIFIED ANALYSIS FUNCTION
# -----------------------------------------------------------------------------
@st.cache_data
def analyze_file(file_content, filename,
                 segment_pattern="A",
                 smooth_window=7, slope_window=0.05,
                 num_minima=5, num_maxima=5, threshold_close=0.02,
                 min_distance_between_minima=0.03,
                 num_peaks_ox=1, num_peaks_red=1,
                 window_length_ox=0.2, window_length_red=0.2):
    
    data = pd.read_csv(StringIO(file_content), engine='python').dropna()
    
    # --- MODIFIED LOGIC: Use new helper function ---
    # This replaces the old `float(filename.split('_')[7])`
    extracted_value = extract_scan_rate_from_filename(filename)
    # --- END MODIFIED LOGIC ---

    cycle_info = {'voltage': 10, 'current': 11, 'cycle_label': 'Cycle 6'}
    results = []

    voltage = data.iloc[:, cycle_info['voltage']].astype(float).values
    current = data.iloc[:, cycle_info['current']].astype(float).values
    
    if segment_pattern == "A":
        segments = [(0, 109), (110, 210), (211, 317), (318, 399)] ### Pattern A
    elif segment_pattern == "B":
        segments = [(0, 99), (100, 199), (200, 299), (300, 399)] ### Pattern B
    else:
        st.warning(f"Invalid pattern '{segment_pattern}'. Defaulting to 'A'.")
        segments = [(0, 109), (110, 210), (211, 317), (318, 399)] ### Pattern A

    ox_v = np.concatenate([voltage[s:e+1] for s, e in [segments[3], segments[0]]])
    ox_i = np.concatenate([current[s:e+1] for s, e in [segments[3], segments[0]]])
    red_v = np.concatenate([voltage[s:e+1] for s, e in [segments[1], segments[2]]])
    red_i = np.concatenate([current[s:e+1] for s, e in [segments[1], segments[2]]])

    current_window_ox = window_length_ox + (extracted_value * 0.1)
    current_window_red = window_length_red + (extracted_value * 0.1)
    
    # (The rest of the analysis logic is unchanged)
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
    
    ox_raw_peak_idx, red_raw_peak_idx = None, None
    if ox_final_selected:
        raw_idxs = [np.argmin(np.abs(ox_v - ox_v_der2[idx])) for idx in ox_final_selected]
        ox_raw_peak_idx = raw_idxs[np.argmax([abs(ox_i[r]) for r in raw_idxs])]
    if red_final_selected:
        raw_idxs = [np.argmin(np.abs(red_v - red_v_der2[idx])) for idx in red_final_selected]
        red_raw_peak_idx = raw_idxs[np.argmax([abs(red_i[r]) for r in raw_idxs])]
    
    ox_side_raw = find_side_points_raw(ox_v, ox_i, ox_raw_peak_idx, current_window_ox)
    red_side_raw = find_side_points_raw(red_v, red_i, red_raw_peak_idx, current_window_red)
    
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
        "window_length_ox": current_window_ox, "window_length_red": current_window_red
    })
    return results

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS (Unchanged)
# -----------------------------------------------------------------------------

def create_interactive_plot(all_cycles_res, **kwargs):
    if not all_cycles_res:
        return go.Figure().update_layout(title_text="No cycles found or analysis failed.")
    
    res = all_cycles_res[0]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Raw Data', 'Second Derivative', 'Peak Detection & Baseline',
                        'Baseline Corrected', 'Subtracted Signal', 'Summary Overlay'),
        vertical_spacing=0.1, horizontal_spacing=0.1
    )

    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i"], mode='lines', name='Oxidation', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i"], mode='lines', name='Reduction', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=res["ox_v_der2"], y=res["ox_deriv2_smooth"], mode='lines', name='Ox d2', line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Scatter(x=res["red_v_der2"], y=res["red_deriv2_smooth"], mode='lines', name='Red d2', line=dict(color='blue')), row=1, col=2)
    if res["ox_final_selected"]:
        fig.add_trace(go.Scatter(x=[res["ox_v_der2"][i] for i in res["ox_final_selected"]], y=[res["ox_deriv2_smooth"][i] for i in res["ox_final_selected"]], mode='markers', name='Ox Peaks d2', marker=dict(color='darkred', size=8)), row=1, col=2)
    if res["red_final_selected"]:
        fig.add_trace(go.Scatter(x=[res["red_v_der2"][i] for i in res["red_final_selected"]], y=[res["red_deriv2_smooth"][i] for i in res["red_final_selected"]], mode='markers', name='Red Peaks d2', marker=dict(color='navy', size=8)), row=1, col=2)
    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i"], mode='lines', name='Ox raw', line=dict(color='red', width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i"], mode='lines', name='Red raw', line=dict(color='blue', width=1), showlegend=False), row=2, col=1)
    if res["ox_raw_peak_idx"] is not None:
        idx = res["ox_raw_peak_idx"]
        fig.add_trace(go.Scatter(x=[res["ox_v"][idx]], y=[res["ox_i"][idx]], mode='markers', name='Ox Peak', marker=dict(color='red', size=10)), row=2, col=1)
        left, right = find_side_points_raw(res["ox_v"], res["ox_i"], idx, res["window_length_ox"])
        if left is not None and right is not None:
            fig.add_trace(go.Scatter(x=[res["ox_v"][left], res["ox_v"][right]], y=[res["ox_i"][left], res["ox_i"][right]], mode='lines+markers', name='Ox Baseline Pts', line=dict(color='black', width=3), marker=dict(size=8)), row=2, col=1)
    if res["red_raw_peak_idx"] is not None:
        idx = res["red_raw_peak_idx"]
        fig.add_trace(go.Scatter(x=[res["red_v"][idx]], y=[res["red_i"][idx]], mode='markers', name='Red Peak', marker=dict(color='blue', size=10)), row=2, col=1)
        left, right = find_side_points_raw(res["red_v"], res["red_i"], idx, res["window_length_red"])
        if left is not None and right is not None:
            fig.add_trace(go.Scatter(x=[res["red_v"][left], res["red_v"][right]], y=[res["red_i"][left], res["red_i"][right]], mode='lines+markers', name='Red Baseline Pts', line=dict(color='black', width=3), marker=dict(size=8)), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i_corrected"], mode='lines', name='Ox Corrected', line=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i_corrected"], mode='lines', name='Red Corrected', line=dict(color='blue')), row=2, col=2)

    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i_subtracted"], mode='lines', name='Ox Subtracted', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i_subtracted"], mode='lines', name='Red Subtracted', line=dict(color='blue')), row=3, col=1)
    if res["ox_raw_peak_idx"] is not None:
        idx = res["ox_raw_peak_idx"]
        fig.add_trace(go.Scatter(x=[res["ox_v"][idx]], y=[res["ox_i_subtracted"][idx]], mode='markers', name='Ox Sub Peak', marker=dict(color='darkred', size=10)), row=3, col=1)
    if res["red_raw_peak_idx"] is not None:
        idx = res["red_raw_peak_idx"]
        fig.add_trace(go.Scatter(x=[res["red_v"][idx]], y=[res["red_i_subtracted"][idx]], mode='markers', name='Red Sub Peak', marker=dict(color='darkblue', size=10)), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i"], mode='lines', name='Original Ox', line=dict(color='red', dash='dash')), row=3, col=2)
    fig.add_trace(go.Scatter(x=res["ox_v"], y=res["ox_i_subtracted"], mode='lines', name='Subtracted Ox', line=dict(color='red')), row=3, col=2)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i"], mode='lines', name='Original Red', line=dict(color='blue', dash='dash')), row=3, col=2)
    fig.add_trace(go.Scatter(x=res["red_v"], y=res["red_i_subtracted"], mode='lines', name='Subtracted Red', line=dict(color='blue')), row=3, col=2)
    
    fig.update_layout(height=1200, showlegend=True, title_text=f"{res['filename']} - {res['cycle_label']} (Scan Rate: {res['scan_rate']} V/s)")
    for r in range(1, 4):
        for c in range(1, 3):
            fig.update_xaxes(title_text="Voltage (V)", row=r, col=c)
            fig.update_yaxes(title_text="Current (uA)", row=r, col=c)
    return fig

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
        # Filter out inf/-inf before linregress
        finite_ox_mask = np.isfinite(avg_df['log_scan_rate']) & np.isfinite(avg_df['log_ox_peak'])
        finite_red_mask = np.isfinite(avg_df['log_scan_rate']) & np.isfinite(avg_df['log_red_peak'])

        if finite_ox_mask.sum() > 1:
            ox_slope = stats.linregress(avg_df.loc[finite_ox_mask, 'log_scan_rate'], avg_df.loc[finite_ox_mask, 'log_ox_peak']).slope
            ox_label = f'log(Avg Ox Peak) | Slope: {ox_slope:.3f}'
        if finite_red_mask.sum() > 1:
            red_slope = stats.linregress(avg_df.loc[finite_red_mask, 'log_scan_rate'], avg_df.loc[finite_red_mask, 'log_red_peak']).slope
            red_label = f'log(Avg Red Peak) | Slope: {red_slope:.3f}'

    # Changed mode from 'markers' to 'lines+markers'
    fig.add_trace(go.Scatter(x=df["Scan Rate (V/s)"], y=df["Ox Subtracted Peak"], mode='lines+markers', name='Ox Peaks', marker=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Scan Rate (V/s)"], y=df["Red Subtracted Peak"], mode='lines+markers', name='Red Peaks', marker=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["sqrt(Scan Rate)"], y=df["Ox Subtracted Peak"], mode='lines+markers', name='Ox Peaks', marker=dict(color='red'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["sqrt(Scan Rate)"], y=df["Red Subtracted Peak"], mode='lines+markers', name='Red Peaks', marker=dict(color='blue'), showlegend=False), row=1, col=2)
    
    # This plot already had lines, so it is unchanged
    fig.add_trace(go.Scatter(x=avg_df["log_scan_rate"], y=avg_df["log_ox_peak"], mode='lines+markers', name=ox_label, line=dict(color='red', dash='dot'), marker=dict(symbol='star', color='darkred', size=8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=avg_df["log_scan_rate"], y=avg_df["log_red_peak"], mode='lines+markers', name=red_label, line=dict(color='blue', dash='dot'), marker=dict(symbol='star', color='darkblue', size=8)), row=2, col=1)

    fig.update_layout(title_text="Peak Current Summary Across All Files", height=900, showlegend=True)
    fig.update_xaxes(title_text="Scan Rate (V/s)", row=1, col=1); fig.update_yaxes(title_text="Peak Current (uA)", row=1, col=1)
    fig.update_xaxes(title_text="√Scan Rate (V/s)⁰⁵", row=1, col=2); fig.update_yaxes(title_text="Peak Current (uA)", row=1, col=2)
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

st.sidebar.header("Controls")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files", type=['csv'], accept_multiple_files=True
)

if uploaded_files:
    file_dict = {f.name: f for f in uploaded_files}
    file_names = sorted(file_dict.keys())
    
    selected_filename = st.sidebar.selectbox("Select a file to analyze:", file_names)
    
    # Load the fixed parameters for the selected file, or use an empty dict
    params_for_file = st.session_state.fixed_params.get(selected_filename, {}).copy()

    st.sidebar.subheader("Analysis Parameters")

    # --- NEW UI ELEMENT: Pattern Selection ---
    # Get the default pattern from saved params, defaulting to 'A'
    default_pattern = params_for_file.get('segment_pattern', 'A')
    default_index = 0 if default_pattern == 'A' else 1
    segment_pattern = st.sidebar.radio(
        "Segmentation Pattern", ("A", "B"), 
        index=default_index,
        help="Select the data point segmentation pattern.\n- 'A': (0-109, 110-210, ...)\n- 'B': (0-99, 100-199, ...)"
    )
    # --- END NEW UI ELEMENT ---

    smooth_window = st.sidebar.slider('Smooth Window', 3, 21, params_for_file.get('smooth_window', 11), 2)
    slope_window = st.sidebar.slider('Slope Window', 0.01, 0.5, params_for_file.get('slope_window', 0.1), 0.01)
    num_minima = st.sidebar.slider('Num Minima Candidates', 1, 200, params_for_file.get('num_minima', 100), 1)
    num_maxima = st.sidebar.slider('Num Maxima Candidates', 1, 200, params_for_file.get('num_maxima', 100), 1)
    threshold_close = st.sidebar.slider('Extrema Proximity Threshold', 0.01, 0.2, params_for_file.get('threshold_close', 0.06), 0.01)
    min_distance_between_minima = st.sidebar.slider('Min Peak Distance (V)', 0.01, 0.2, params_for_file.get('min_distance_between_minima', 0.05), 0.01)
    
    st.sidebar.subheader("Baseline Correction")
    window_length_ox = st.sidebar.slider('Oxidation Window Length', 0.1, 1.0, params_for_file.get('window_length_ox', 0.6), 0.01)
    window_length_red = st.sidebar.slider('Reduction Window Length', 0.1, 1.0, params_for_file.get('window_length_red', 1.0), 0.01)

    # Add the new segment_pattern to the dictionary of current parameters
    current_params = {
        'segment_pattern': segment_pattern, # <-- Added
        'smooth_window': smooth_window, 'slope_window': slope_window,
        'num_minima': num_minima, 'num_maxima': num_maxima,
        'threshold_close': threshold_close, 'min_distance_between_minima': min_distance_between_minima,
        'window_length_ox': window_length_ox, 'window_length_red': window_length_red,
        'num_peaks_ox': 1, 'num_peaks_red': 1
    }

    if st.sidebar.button("📌 Fix Parameters for this File"):
        # This will now save the segment_pattern along with all other params
        st.session_state.fixed_params[selected_filename] = current_params
        st.sidebar.success(f"Parameters fixed for {selected_filename}!")

    tab1, tab2 = st.tabs(["Single File Analysis", "Full Data Summary"])

    with tab1:
        st.header(f"Analysis for: `{selected_filename}`")
        selected_file_content = file_dict[selected_filename].getvalue().decode("utf-8")
        # Pass all current params (including the pattern) to the analysis function
        analysis_results = analyze_file(selected_file_content, selected_filename, **current_params)
        fig_single = create_interactive_plot(analysis_results, **current_params)
        st.plotly_chart(fig_single, use_container_width=True)

    with tab2:
        st.header("Summary of All Uploaded Files")
        if st.button("📊 Generate Full Summary Plot"):
            all_summary_data = []
            progress_bar = st.progress(0, text="Analyzing files...")
            for i, filename in enumerate(file_names):
                file_content = file_dict[filename].getvalue().decode("utf-8")
                # Get fixed params for the file, or use the currently selected params as default
                params = st.session_state.fixed_params.get(filename, current_params)
                
                # Pass the correct set of params (including the pattern) for *this specific file*
                res_list = analyze_file(file_content, filename, **params)
                
                if res_list:
                    res = res_list[0]
                    all_summary_data.append({
                        "File": res["filename"], "Cycle": res["cycle_label"],
                        "Scan Rate (V/s)": res["scan_rate"],
                        "Ox Subtracted Peak": res["ox_subtracted_peak"],
                        "Red Subtracted Peak": res["red_subtracted_peak"]
                    })
                progress_bar.progress((i + 1) / len(file_names), text=f"Analyzing {filename}...")
            
            progress_bar.empty()
            if all_summary_data:
                summary_df = pd.DataFrame(all_summary_data)
                fig_summary = create_interactive_summary_plots(summary_df)
                st.plotly_chart(fig_summary, use_container_width=True)
                st.subheader("Summary Data Table")
                st.dataframe(summary_df)
            else:
                st.warning("Could not generate summary data. Check files or parameters.")
else:
    st.info("Please upload one or more CSV files to begin.")


