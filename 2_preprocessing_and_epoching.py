# %%
# 2. Preprocessing and Epoching
# 
# This notebook continues from the data loading step and guides you through the preprocessing pipeline described in the project overview. The goal is to clean the raw data and structure it for model training.
# 
# The steps include:
# - DBS artifact removal (conceptual outline)
# - Downsampling
# - Filtering (band-pass and notch)
# - Common Average Referencing (CAR)
# - Source Power Comodulation (SPoC) for spatial filtering
# - Epoching into 1-second segments
# - Calculating tracing speed from stylus data

# %%
# 2.1 Setup and Data Loading
# 
# We start by importing the necessary libraries and loading the data, just as in the first notebook.

# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import mne
import pyxdf

# Add the analysis_scripts directory to the Python path
sys.path.append(str(Path.cwd().parent.parent / "analysis_scripts"))

from load_xdf import get_xdf_files, create_raws_from_mat_and_xdf

# Load the data again
session_id = "S003"  # Example session
session_folder = f"./data/data_p00{session_id[-1]}/"
xdf_files = get_xdf_files(session_folder)
raws = create_raws_from_mat_and_xdf(xdf_files, day="day3")
raw = mne.concatenate_raws(raws)

# We'll work with a copy to keep the original data intact
raw_processed = raw.copy()

# %%
# 2.2 DBS Artifact Removal (Conceptual)
# 
# The project description mentions using a template subtraction method for removing DBS artifacts in ON-state recordings. This is a crucial but complex step that requires a precise template of the stimulation artifact.
# 
# The process generally involves:
# 1.  Identifying stimulation pulse timings.
# 2.  Creating an artifact template.
# 3.  Subtracting the template.
# 
# Implementing this is highly specific to the data and stimulation parameters. Below is a conceptual function.
# You would need to replace it with a proper implementation.

# %%
def remove_dbs_artifact(raw, stim_freq):
    """Placeholder function for DBS artifact removal."""
    print("Applying conceptual DBS artifact removal...")
    # In a real implementation, you would find stim events,
    # create a template, and subtract it.
    # For now, this function does nothing to the data.
    return raw

# The function call is omitted because it is a local function.
# To apply the artifact removal, you might call:
# raw_processed = remove_dbs_artifact(raw_processed, stim_freq=130)  # Assuming a 130Hz stimulation

# %%
# 2.3 Downsampling
# 
# The original data is sampled at 22 kHz, which is very high. We downsample it to 1000 Hz to make it more manageable and to match the frequency range of interest for most neural signals.

# %%
print(f"Original sampling frequency: {raw_processed.info['sfreq']} Hz")
raw_processed.resample(1000, npad="auto")
print(f"New sampling frequency: {raw_processed.info['sfreq']} Hz")

# %%
# 2.4 Filtering
# 
# We apply two types of filters:
# 1.  Band-pass filter (3-250 Hz): Removes very low-frequency drifts and high-frequency noise.
# 2.  Notch filter (50 Hz and harmonics): Removes power-line noise.

# %%
# Apply band-pass filter
raw_processed.filter(l_freq=3.0, h_freq=250.0, fir_design="firwin")

# Apply notch filter for power-line noise
notch_freqs = np.arange(50, 251, 50)
raw_processed.notch_filter(freqs=notch_freqs, fir_design="firwin")

# %%
# 2.5 Common Average Reference (CAR)
# 
# CAR is a re-referencing technique that helps reduce common-mode noise across all channels. It works by subtracting the average signal of all electrodes from each individual electrode.

# %%
# Apply CAR. We'll apply it to ECOG and LFP channels separately.
raw_processed.set_eeg_reference("average", projection=True, ch_type="ecog")
raw_processed.set_eeg_reference("average", projection=True, ch_type="dbs")  # 'dbs' for LFP
raw_processed.apply_proj()

# %%
# 2.6 Source Power Comodulation (SPoC)
# 
# SPoC is a spatial filtering technique used to find components whose power comodulates with a target variable.
# It requires a target variable y to be fitted.
# Here, we outline the steps with a placeholder target variable.

# %%
from mne.decoding import SPoC

# 1. Define a target variable `y`.
y = np.random.randn(len(raw_processed.times))

# 2. Initialize the SPoC transformer
spoc = SPoC(n_components=4, reg="ledoit_wolf", rank="full")

# 3. Fit SPoC to the data (conceptual; actual call is omitted)
print("Fitting SPoC... (this may take a while)")
# The following line is commented out because it is a conceptual placeholder:
# spoc.fit(raw_processed.get_data(), y)
print("Conceptual SPoC fitting complete. In a real scenario, you would now use spoc.transform()")

# %%
# 2.7 Epoching
# 
# Now we segment the continuous, preprocessed data into 1-second epochs.
# This is a common step before feeding data into a machine learning model.

# %%
# Get events from annotations
events, event_id = mne.events_from_annotations(raw_processed)

# Create 1-second epochs, starting from the event onset
epochs = mne.Epochs(
    raw_processed,
    events,
    event_id,
    tmin=0.0,
    tmax=1.0,
    baseline=None,
    preload=True,
    reject=None,
)

print(epochs)

# %%
# 2.8 Calculate Tracing Speed
# 
# Finally, calculate the tracing speed from the stylus coordinates.
# Speed is a useful behavioral variable that can be correlated with neural activity or used as a target for decoding models.

# %%
# Load the stylus data again (as in the first notebook)
xdf_file_path = xdf_files[0]
streams, header = pyxdf.load_xdf(xdf_file_path)
stylus_stream = next((s for s in streams if "Mouse" in s["info"]["name"][0]), None)

if stylus_stream:
    df_stylus = pd.DataFrame(stylus_stream["time_series"], columns=["x", "y"])
    df_stylus["time"] = stylus_stream["time_stamps"]

    # Calculate differences in position and time
    df_stylus["dx"] = df_stylus["x"].diff()
    df_stylus["dy"] = df_stylus["y"].diff()
    df_stylus["dt"] = df_stylus["time"].diff()

    # Calculate speed
    df_stylus["speed"] = np.sqrt(df_stylus["dx"] ** 2 + df_stylus["dy"] ** 2) / df_stylus["dt"]
    df_stylus = df_stylus.dropna()

    # Plot the speed over time
    fig = px.line(df_stylus, x="time", y="speed", title="Stylus Tracing Speed")
    fig.show()
else:
    print("Could not find stylus stream to calculate speed.")