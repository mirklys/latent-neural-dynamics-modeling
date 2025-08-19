# %% [markdown]
# # 1. Data Loading and Exploration (Unwrapped Script)
#
# This script demonstrates how to load and explore the multi-modal data from the Dareplane project.
# It follows the same logic as `1_data_loading_and_exploration.ipynb` but **unwraps all functions**
# into a sequential, linear script where each logical block is a cell.
# This allows for step-by-step execution and inspection of intermediate variables.

# %%
# =============================================================================
# Cell 1: Imports and Setup
# =============================================================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pyxdf
import plotly.express as px
import plotly.graph_objects as go

print("Cell 1: Imports and setup complete.")

# %%
# =============================================================================
# Cell 2: Define Constants and File Paths
# =============================================================================
# This logic was originally part of the setup in the notebook and in `analysis_scripts/load_xdf.py`.

# Define the root directory for the data
DATA_ROOT = Path.cwd().parent.joinpath("data")

# This map is used to rename the raw channel names from the .mat file
# to more human-readable names like 'LFP_1', 'ECOG_1', etc.
CHANNEL_MAP = {
    "CECOG_HF_1___01___Array_1___01": "LFP_1",
    "CECOG_HF_1___02___Array_1___02": "LFP_2",
    "CECOG_HF_1___03___Array_1___03": "LFP_3",
    "CECOG_HF_1___04___Array_1___04": "LFP_4",
    "CECOG_HF_1___05___Array_1___05": "LFP_5",
    "CECOG_HF_1___06___Array_1___06": "LFP_6",
    "CECOG_HF_1___07___Array_1___07": "LFP_7",
    "CECOG_HF_1___08___Array_1___08": "LFP_8",
    "CECOG_HF_1___09___Array_2___09": "LFP_9",
    "CECOG_HF_1___10___Array_2___10": "LFP_10",
    "CECOG_HF_1___11___Array_2___11": "LFP_11",
    "CECOG_HF_1___12___Array_2___12": "LFP_12",
    "CECOG_HF_1___13___Array_2___13": "LFP_13",
    "CECOG_HF_1___14___Array_2___14": "LFP_14",
    "CECOG_HF_1___15___Array_2___15": "LFP_15",
    "CECOG_HF_1___16___Array_2___16": "LFP_16",
    "CECOG_HF_2___01___Array_3___01": "ECOG_1",
    "CECOG_HF_2___02___Array_3___02": "ECOG_2",
    "CECOG_HF_2___03___Array_3___03": "ECOG_3",
    "CECOG_HF_2___04___Array_3___04": "ECOG_4",
    "CECOG_HF_2___05___Array_3___05": "EOG_1",
    "CECOG_HF_2___06___Array_3___06": "EOG_2",
    "CECOG_HF_2___07___Array_3___07": "EOG_3",
    "CECOG_HF_2___08___Array_3___08": "EOG_4",
}

# We will work with a single example file to demonstrate the process.
# This corresponds to the file selected in the original notebook.
session = "day4"
try:
    fname_xdf = list(
        DATA_ROOT.joinpath(f"sub-p001_ses-{session}", "lsl").glob(
            "*block1_copydraw_off.xdf"
        )
    )[0]
    print(f"Cell 2: Selected XDF file for analysis:\n{fname_xdf}")
except IndexError:
    raise FileNotFoundError(
        f"Could not find example file in {DATA_ROOT}. Please ensure data is downloaded."
    )

# %%
# =============================================================================
# Cell 3: Find and Load the corresponding .mat file
# =============================================================================
# This logic is unwrapped from the `get_AO_file_data` function.

print("Cell 3: Finding and loading the corresponding .mat file.")

# Get the file stem (e.g., 'block1_copydraw_off') from the xdf path
file_stem = fname_xdf.stem

# The .mat files are in a sibling directory named 'AO'.
ao_directory = fname_xdf.parents[1].joinpath("AO")

# Find all .mat files in the AO directory that match the stem.
mat_files = list(ao_directory.glob(f"{file_stem}*mat"))

if not mat_files:
    raise FileNotFoundError(
        f"No corresponding .mat file found for {fname_xdf.name} in {ao_directory}"
    )

mat_file_path = mat_files[0]
print(f"Found .mat file: {mat_file_path}")

# Load the .mat file. `loadmat` returns a dictionary where keys are variable
# names from the MATLAB workspace and values are the data.
mat_data = loadmat(mat_file_path)

print(f"Successfully loaded .mat file. Found {len(mat_data)} variables.")
print("Example keys:", list(mat_data.keys())[:5], "...")

# %%
# =============================================================================
# Cell 4: Process Neural Data from .mat file into a DataFrame
# =============================================================================
# This logic is unwrapped from the `get_AO_data` function.

print("\nCell 4: Processing neural data from the .mat file.")

# Extract a reference channel to get timing information.
# The [0] indexing is because loadmat can create nested arrays.
ref_channel_name = "CECOG_HF_2___01___Array_3___01"
ref_channel_data = mat_data[ref_channel_name][0]

# The sampling rate is stored in a variable like '..._KHz'.
# It's given in KHz, so we multiply by 1000.
sfreq = mat_data[f"{ref_channel_name}_KHz"][0][0] * 1000
print(f"Sampling frequency: {sfreq} Hz")

# Create a time vector in seconds.
num_samples = len(ref_channel_data)
duration_sec = num_samples / sfreq
time_vector = np.linspace(0, duration_sec, num_samples, endpoint=False)

# Create the initial DataFrame for the AlphaOmega (AO) neural data.
df_ao = pd.DataFrame({"time": time_vector, "src": "AO"})

# Add all the channels defined in CHANNEL_MAP to the DataFrame.
channel_data_dict = {v: mat_data[k][0] for k, v in CHANNEL_MAP.items()}
df_ao = df_ao.assign(**channel_data_dict)

print("Created DataFrame for AO (neural) data.")
df_ao.info()

# %%
# =============================================================================
# Cell 5: Process Hardware Markers (CPORT)
# =============================================================================
# This logic is also from `get_AO_data` (when with_cport_marker=True).

print("\nCell 5: Processing hardware markers (CPORT).")

# The CPORT stream contains hardware markers sent via a parallel port.
# It's a 2xN array: row 0 is timestamps, row 1 is marker values.
cport_data = mat_data["CPORT__1"]
cport_timestamps = cport_data[0]
cport_values = cport_data[1]

# The timestamps need to be converted to sample indices in our main data array.
cport_khz = mat_data["CPORT__1_KHz"][0][0]
t_start_recording = mat_data[f"{ref_channel_name}_TimeBegin"][0][0]

# Convert CPORT timestamps to seconds and align with the recording start time.
cport_time_sec = (cport_timestamps / (cport_khz * 1000)) - t_start_recording

# Convert aligned time in seconds to sample indices.
cport_indices = (cport_time_sec * sfreq).astype(int).flatten()

# Ignore markers with a value of 0 (hardware reset signals).
valid_marker_mask = cport_values != 0
valid_indices = cport_indices[valid_marker_mask]
valid_values = cport_values[valid_marker_mask]

# Add them to a new 'marker' column in the DataFrame.
valid_indices_in_bounds = valid_indices[
    (valid_indices >= 0) & (valid_indices < len(df_ao))
]
valid_values_in_bounds = valid_values[
    (valid_indices >= 0) & (valid_indices < len(df_ao))
]
df_ao["marker"] = np.nan  # Initialize column
df_ao.loc[valid_indices_in_bounds, "marker"] = valid_values_in_bounds

print(f"Added {len(valid_indices_in_bounds)} hardware markers to the 'marker' column.")
print("Marker summary:\n", df_ao["marker"].value_counts())

# %%
# =============================================================================
# Cell 6: Load Behavioral (Stylus) and LSL Marker Data from XDF
# =============================================================================
# This logic is unwrapped from the `get_xdf_data` function.

print("\nCell 6: Loading behavioral (stylus) and LSL marker data from the .xdf file.")

xdf_streams, _ = pyxdf.load_xdf(fname_xdf)

# --- Extract Stylus Data ---
stylus_stream = next((s for s in xdf_streams if s["info"]["name"][0] == "Stylus"), None)
if stylus_stream is None:
    raise ValueError("Could not find 'Stylus' stream in the XDF file.")

df_stylus = pd.DataFrame(stylus_stream["time_series"], columns=["x", "y", "force"])
df_stylus["time"] = stylus_stream["time_stamps"]
df_stylus["src"] = "LSL_Stylus"
print("Created DataFrame for stylus data.")

# --- Extract LSL Marker Data ---
marker_stream = next(
    (s for s in xdf_streams if s["info"]["name"][0] == "CopyDrawParadigmMarkerStream"),
    None,
)
if marker_stream is None:
    raise ValueError("Could not find 'CopyDrawParadigmMarkerStream' in the XDF file.")

# Create a clean DataFrame for just the LSL markers and their precise timestamps
df_markers = pd.DataFrame(
    {"time": marker_stream["time_stamps"], "marker": marker_stream["time_series"][:, 0]}
)
print(f"Found {len(df_markers)} LSL markers.")


# --- Extract LSL Neural Data (for alignment purposes) ---
# This is the neural data as streamed over LSL, which may have gaps or jitter.
# We use it to find the time offset relative to the high-quality AO data.
lsl_neural_stream = next(
    (s for s in xdf_streams if s["info"]["name"][0] == "AODataStream"), None
)
if lsl_neural_stream is None:
    raise ValueError("Could not find 'AODataStream' in the XDF file.")

# Use the same reference channel as in the AO data for comparison (ECOG_1 is channel 16)
df_lsl = (
    pd.DataFrame(
        {
            "time": lsl_neural_stream["time_stamps"],
            "data": lsl_neural_stream["time_series"][:, 16],
            "src": "LSL",
        }
    )
    .sort_values("time")
    .reset_index(drop=True)
)

# --- Normalize Timestamps ---
# To make comparisons easier, we set the start time of both LSL dataframes to 0.
# The absolute timestamps are preserved in the original variables.
t_min_lsl = df_lsl.time.iloc[0]
df_lsl["time"] -= t_min_lsl
df_markers["time"] -= t_min_lsl

print("Created DataFrame for LSL neural data (for alignment).")
print("Normalized timestamps for LSL dataframes.")

# %%
# =============================================================================
# Cell 7: Align Data Streams
# =============================================================================
# This cell unwraps the logic from `align_on_markers` and `align_on_stim_artifact`.
# It first tries to align using hardware markers, and if that fails, it falls
# back to using the large stimulation artifact.

print("\nCell 7: Aligning AO and LSL data streams.")
aligned_df = None
time_offset = None

# --- Method 1: Align on Hardware Markers (unwrapped from `align_on_markers`) ---
try:
    print("Attempting alignment using hardware markers...")
    # Get the timestamps of non-null markers from both sources
    ao_marker_times = df_ao[df_ao.marker.notna()].time
    lsl_marker_times = df_markers[df_markers.marker.notna()].time

    # Calculate the time difference (delta) between consecutive markers
    dt_ao = np.diff(ao_marker_times)
    dt_lsl = np.diff(lsl_marker_times)

    # Find a matching sequence of marker intervals. This is needed because
    # the AO recording might have extra hardware triggers at the start.
    # We look for a sequence in AO deltas that matches the first LSL delta.
    match_found = False
    ao_start_index = -1
    for i in range(min(3, len(dt_ao))):  # Check first 3 AO deltas
        # Allow for a 10% tolerance in timing
        if np.abs(dt_ao[i] - dt_lsl[0]) / dt_lsl[0] < 0.1:
            ao_start_index = i
            match_found = True
            break

    if not match_found:
        raise KeyError("No matching marker time differences found.")

    print(
        f"Found matching marker sequence starting at AO marker index {ao_start_index}."
    )

    # --- Transfer LSL markers to the AO timeline ---
    # Get the time of the first matched AO marker
    t0_ao = ao_marker_times.iloc[ao_start_index]

    # The times of all subsequent LSL markers relative to the first one
    lsl_marker_relative_times = lsl_marker_times - lsl_marker_times.iloc[0]

    # Calculate the absolute timestamps for LSL markers in the AO time domain
    aligned_lsl_marker_times = t0_ao + lsl_marker_relative_times

    # Find the closest sample index in the AO dataframe for each aligned marker time
    aligned_indices = [np.searchsorted(df_ao.time, t) for t in aligned_lsl_marker_times]

    # Create the new 'lsl_marker' column in the AO dataframe
    aligned_df = df_ao.copy()
    aligned_df["lsl_marker"] = np.nan
    aligned_df.loc[aligned_indices, "lsl_marker"] = df_markers[
        df_markers.marker.notna()
    ].marker.values

    # For visualization, calculate the time offset
    time_offset = ao_marker_times.iloc[ao_start_index] - lsl_marker_times.iloc[0]

    print(
        f"Success: Aligned data using hardware markers. Time offset: {time_offset:.4f}s"
    )

# --- Method 2: Align on Stimulation Artifact (unwrapped from `align_on_stim_artifact`) ---
except (KeyError, IndexError) as e:
    print(f"Marker alignment failed ({e}). Falling back to stim artifact alignment.")

    threshold = 2000  # uV, for detecting the large stimulation artifact

    # Find the index of the first major peak (the stim artifact) in both streams
    # We start searching after 1s to avoid startup artifacts.
    ix_lsl_first_peak = df_lsl[(df_lsl.data > threshold) & (df_lsl.time > 1)].index[0]
    ix_ao_first_peak = df_ao[(df_ao.data > threshold) & (df_ao.time > 1)].index[0]

    # Calculate the coarse time offset between the two streams
    coarse_time_offset = (
        df_ao.time.iloc[ix_ao_first_peak] - df_lsl.time.iloc[ix_lsl_first_peak]
    )
    print(f"Coarse offset from stim artifact: {coarse_time_offset:.4f}s")

    # --- Fine-tune the offset with a small cross-correlation ---
    # This corrects for small misalignments by matching the signal shape.
    chk_idx_range = 10_000  # Search window in samples (in AO data)
    chk_len = 400  # Length of the signal chunk to match (in LSL data)

    # We'll match the signal shape around the first LSL marker
    idx_lsl_first_marker = df_lsl[df_lsl.marker.notna()].index[0]
    t_lsl_first_marker = df_lsl.time.iloc[idx_lsl_first_marker]

    # Estimate where this marker should be in the AO data using the coarse offset
    ao_test_idx = np.searchsorted(df_ao.time, t_lsl_first_marker + coarse_time_offset)

    # Extract the chunk of LSL data to be used as a template
    dlsl_chk = df_lsl.iloc[idx_lsl_first_marker : idx_lsl_first_marker + chk_len]

    # Extract a larger search window from the AO data
    dao_chk = df_ao.iloc[ao_test_idx - chk_idx_range : ao_test_idx + chk_idx_range]

    # Slide the LSL chunk across the AO window and find the best match
    differences = np.asarray(
        [
            np.abs(
                dao_chk.iloc[i : i + chk_len].ECOG_1.values - dlsl_chk.data.values
            ).mean()
            for i in range(len(dao_chk) - chk_len)
        ]
    )

    # The shift that minimizes the difference
    idx_shift = np.argmin(differences) - chk_idx_range
    fine_tuned_offset = coarse_time_offset + idx_shift / sfreq
    time_offset = fine_tuned_offset  # Store for visualization
    print(f"Fine-tuned offset after cross-correlation: {fine_tuned_offset:.4f}s")

    # --- Transfer LSL markers to the AO timeline using the final offset ---
    dm = df_markers[df_markers.marker.notna()]
    aligned_indices = [
        np.searchsorted(df_ao.time, t + fine_tuned_offset) for t in dm.time
    ]

    aligned_df = df_ao.copy()
    aligned_df["lsl_marker"] = np.nan
    aligned_df.loc[aligned_indices, "lsl_marker"] = dm.marker.values

    print("Success: Aligned data using stimulation artifact.")

# %%
# =============================================================================
# Cell 8: Final Inspection of Aligned Data
# =============================================================================
print("\nCell 8: Inspecting the final aligned DataFrame.")

if aligned_df is not None:
    print(f"Alignment complete. Result is a DataFrame with shape {aligned_df.shape}")
    print(
        f"Found {aligned_df.lsl_marker.notna().sum()} LSL markers now aligned to the AO data."
    )
    print("\nValue counts of aligned markers:")
    print(aligned_df.lsl_marker.value_counts())
    print("\nHead of the DataFrame with a marker:")
    print(aligned_df[aligned_df.lsl_marker.notna()].head())
else:
    print("Alignment failed. `aligned_df` is None.")

# %%
# =============================================================================
# Cell 9: Visualization - Check Alignment
# =============================================================================
# This is a crucial step to visually confirm the alignment worked. We plot
# the LSL signal (shifted by the calculated offset) on top of the AO signal.
# They should overlap almost perfectly.

print("\nCell 9: Visualizing the alignment quality.")

if time_offset is not None and aligned_df is not None:
    # Create a new DataFrame for plotting LSL data with the aligned time
    df_lsl_aligned = df_lsl.copy()
    df_lsl_aligned["time"] += time_offset

    # Concatenate for easier plotting with Plotly Express
    plot_df = pd.concat(
        [
            df_ao[["time", "ECOG_1", "src"]],
            df_lsl_aligned[["time", "data", "src"]].rename(columns={"data": "ECOG_1"}),
        ]
    )

    # Select a small time window to see the alignment clearly.
    # Let's find the first marker to center the plot around.
    first_marker_time = aligned_df[aligned_df.lsl_marker.notna()].time.iloc[0]
    plot_window_start = first_marker_time - 0.5
    plot_window_end = first_marker_time + 0.5

    plot_df_window = plot_df[
        (plot_df["time"] >= plot_window_start) & (plot_df["time"] <= plot_window_end)
    ]

    fig = px.line(
        plot_df_window,
        x="time",
        y="ECOG_1",
        color="src",
        title="Alignment Check: AO vs LSL (shifted)",
        labels={"time": "Time (s)", "ECOG_1": "Amplitude (uV)"},
    )
    fig.show()
else:
    print("Skipping alignment visualization because alignment failed.")

# %%
# =============================================================================
# Cell 10: Visualization - Stylus Trace
# =============================================================================
# Finally, let's visualize the behavioral data: the path traced by the stylus.

print("\nCell 10: Visualizing the stylus trace.")

# The stylus data is in the `df_stylus` DataFrame created in Cell 6.
fig = px.line(df_stylus, x="x", y="y", title="Stylus Trace for the Block")

# Screen coordinates often have the origin at the top-left, so we invert the y-axis.
fig.update_yaxes(autorange="reversed")
fig.update_layout(
    xaxis=dict(scaleanchor="y", scaleratio=1)
)  # Ensure aspect ratio is 1:1
fig.show()
