# In-Depth Overview of the Data Processing Workflow

This document provides a detailed walkthrough of the data processing pipeline for the Dareplane project dataset. It expands upon the steps implemented in the accompanying Jupyter notebooks (`1_data_loading_and_exploration.ipynb` and `2_preprocessing_and_epoching.ipynb`), explaining what each step does and why it is important for preparing the data for machine learning analysis.

---

## Phase 1: Data Loading and Exploration

The primary goal of this phase is to load the raw data from its source format into a structured environment and perform initial quality checks. This ensures that we are starting with a solid foundation for the more complex preprocessing steps to follow.

### 1.1. Loading Multi-Modal Data

- **What was done:** We used the `pyxdf` library to parse the `.xdf` files, which are containers for multiple time-synchronized data streams. The core neural data (LFP, ECoG, EOG) was then loaded into an `mne.Raw` object using custom functions from `analysis_scripts/load_xdf.py`. The `mne` library is the de-facto standard for neurophysiological data analysis in Python, providing robust data structures and a vast array of tools.
- **Why it's important:** Raw data comes in many formats. This step standardizes it into a single, well-defined structure (`mne.Raw`) that all subsequent tools can work with. It also handles the critical task of aligning data from different sources (the Neuro Omega amplifier and the behavioral recording system) using timestamps and marker events.

### 1.2. Inspecting Data Streams and Metadata

- **What was done:** The notebooks demonstrate how to inspect the `raw.info` attribute, which contains metadata like channel names, channel types (e.g., `ecog`, `dbs`), and the sampling frequency.
- **Why it's important:** This is a crucial sanity check. It verifies that the data was loaded correctly, all expected channels are present, and the sampling rate is what we expect (originally 22 kHz). Misinterpreting a channel or using an incorrect sampling rate can invalidate the entire analysis.

### 1.3. Visualizing Raw Signals and Behavior

- **What was done:** We plotted the raw time-series data for a few selected channels and also plotted the stylus (X-Y) coordinates to visualize the drawing trace.
- **Why it's important:** Visualization provides an intuitive understanding of the data that summary statistics cannot. It allows us to spot gross abnormalities immediately, such as dead channels (flat lines), major electrical noise, or unexpected patterns in the behavioral data. It also gives us a first look at the event markers that define the experiment's structure.

---

## Phase 2: Preprocessing and Signal Cleaning

This is the most intensive phase. Its purpose is to remove noise and artifacts—unwanted components of the signal that are not of neural origin—to improve the signal-to-noise ratio (SNR) of the underlying brain activity.

### 2.1. DBS Artifact Removal via Template Subtraction

This step is critical for analyzing data from the DBS-ON sessions.

- **The Problem:** Deep Brain Stimulation involves delivering high-frequency electrical pulses directly to the brain. These pulses create massive electrical artifacts in the recorded neural signals (LFP, ECoG). The artifact's amplitude can be orders of magnitude larger than the actual brain signals, completely masking them.

- **The Solution: Template Subtraction and Rebuilding:** This is a widely-used and effective technique to remove stereotyped artifacts. The process is as follows:
    1.  **Pulse Detection:** The first step is to find the exact time of each stimulation pulse. This can be done using a dedicated synchronization channel or by detecting the large, sharp peaks in the recorded data.
    2.  **Epoching around Pulses:** The continuous signal is cut into small segments (epochs), each centered on one stimulation pulse.
    3.  **Template Creation (Simple Averaging):** A basic template is created by averaging all of these epochs together. The assumption is that the underlying neural activity is random and will average out to zero, while the stimulation artifact is time-locked and consistent, so it will remain. The result is a clean waveform of the "average" artifact.
    4.  **Template Subtraction:** This average template is then subtracted from the raw signal at each pulse location, effectively removing the artifact and leaving the neural signal behind.

- **Advanced Method: Interpolation-Based Template Rebuilding:** The project description mentions a more advanced variant. A simple static template assumes the artifact's shape is identical for every pulse. In reality, it can vary slightly due to physiological changes or hardware factors. **Template rebuilding** addresses this by creating a more dynamic template. Instead of using a single average for all pulses, it might, for example:
    - Use a **moving average:** The template for a given pulse is the average of the N preceding and N succeeding pulses.
    - Use **interpolation:** It can create a unique template for each pulse by interpolating the shapes of the adjacent artifacts.

    This ensures that slow variations in the artifact shape are captured and removed more accurately, leading to a cleaner final signal. The placeholder function in the notebook marks where this complex but vital step should be implemented.

### 2.2. Downsampling

- **What was done:** The data was downsampled from 22,000 Hz to 1,000 Hz using `raw.resample()`.
- **Why it's important:** The original sampling rate is extremely high, capturing frequencies far beyond what is typically analyzed in LFP/ECoG studies (which rarely exceeds a few hundred Hz). Downsampling makes the data files smaller, reduces memory usage, and dramatically speeds up all subsequent processing steps without losing relevant neural information.

### 2.3. Filtering

- **What was done:** A 3-250 Hz band-pass filter and a 50 Hz (plus harmonics) notch filter were applied.
- **Why it's important:**
    - **Band-pass:** The low-cut (3 Hz) removes very slow signal drifts that are physiological but not typically part of the signal of interest. The high-cut (250 Hz) removes high-frequency noise. This focuses the analysis on the relevant neural frequency bands (from delta up to high-gamma).
    - **Notch:** Electrical equipment in the recording environment introduces noise at the power-line frequency (50 Hz in Europe). This filter specifically removes that frequency and its harmonics (100, 150, 200, 250 Hz), which are common and powerful contaminants.

### 2.4. Common Average Referencing (CAR)

- **What was done:** The average signal across all ECoG channels was subtracted from each individual ECoG channel (and likewise for LFP channels).
- **Why it's important:** It is a spatial filter that helps to remove noise that is present on all channels simultaneously, such as noise from distant electrical sources or movement-related artifacts. This enhances the signals that are unique to each electrode's local area.

---

## Phase 3: Feature Engineering and Structuring for Modeling

The final phase prepares the cleaned data for machine learning.

### 3.1. Source Power Comodulation (SPoC)

- **What was done:** The notebook outlines the setup for SPoC, a supervised spatial filtering method.
- **Why it's important:** Unlike CAR, which is an unsupervised method, SPoC uses knowledge about the task (a target variable, `y`) to find the best spatial filters. It identifies patterns of brain activity across multiple channels whose signal *power* is maximally correlated with the target variable (e.g., tracing speed). Applying SPoC can produce a small number of "virtual channels" that have a much higher signal-to-noise ratio for the specific behavior you want to decode.

### 3.2. Epoching

- **What was done:** The continuous preprocessed data was segmented into 1-second chunks, time-locked to task events.
- **Why it's important:** Machine learning models typically work with fixed-size inputs. Epoching creates these inputs. Each epoch represents a single data point (an "X" sample) for the model. The choice of epoch length (1 second) and timing (e.g., starting at a trial cue) are critical design decisions.

### 3.3. Calculating Behavioral Features (Tracing Speed)

- **What was done:** We used the raw stylus coordinates and timestamps to calculate the instantaneous tracing speed.
- **Why it's important:** This creates a continuous behavioral variable that can be used as the prediction target (`y`) for a decoding model. The goal of the model would be to learn the mapping from an epoch of brain data (X) to the corresponding tracing speed (y). This is a classic regression problem in brain-computer interfacing.

By following these three phases, you transform complex, noisy, raw data into a clean, well-structured dataset ready for the final step of model training and evaluation.
