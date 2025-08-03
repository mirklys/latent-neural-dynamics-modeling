# Data and script collection used for the Dareplane Paper

This repository contains the data and analysis scripts used for the paper "Dareplane: A modular open-source software platform for BCI research with application in closed-loop deep brain stimulation", which is currently under review at the Journal of Neural Engineering. The paper is available as a preprint on [arxiv](https://arxiv.org/abs/2408.01242).

# Data
This data set comprises three types of recordings for which all recording modalities are provided in the Dareplane paper.

## 1. Benchtop evaluations of closed-loop setups running Dareplane modules. 
These recordings used the AWG of a Picoscope to generate a sinus wave, which was recorded with different neurostimulators (BIC EvalKit, CorTec and Neuro Omega, Alpha Omega) or read from a single oscilloscope's channel. The signal was then processed in a chain of Dareplane modules handing over data in LSL streams and finally triggering a single stimulation peak on the neurostimulators or an Arduino Uno within the [0, 2pi] phase of the reference signal.
XDF files are available for each of the three recording types and can be found under:
  - `/data/AO_test.xdf`
  - `/data/CT_pico_loop_with_1ms_sleep.xdf`
  - `/data/Arduino_test.xdf`


## 2. Electrophysiological recordings during the CopyDraw
These recordings comprise local field potentials (LFP - 16 channels), electrocorticography (ECoG - 4 channels), electrooculography (EOG - 4 channels), heart rate (HR), respiratory rate (RR) and galvanic skin response (GSR) for a single PD patient, collected in a subacute phase after implantation of DBS electrodes. LFP, ECoG and EOG were collected at using the Neuro Omega (Alpha Omega, Israel) neurostimulator. HR, RR, GSR were recorded using a BrainAmp ExG (Brain Products, Germany). Additionally behavioral data from a Wacom stylus was collected multiple blocks of the CopyDraw task. Additionally, continuous LSL streams were recorded as XDF, containing LFP, ECoG, EOG and behavioral markers. 

Data of three subacute sessions can be found in the folders /data/sub-p001_ses-day2, /data/sub-p001_ses-day3, /data/sub-p001/ses-04, with the session suffix referring to the days after surgery.

### LFP, ECoG, EOG
This data is found in the subfolders /data/sub-p001_ses-day2/ao/ (etc) and is provided in "mat" files. File names indicate the CopyDraw block number and stimulation (DBS stimulation condition), the data was recorded in.

Loading the `*.mat` files will result in key-values pairs for the various recorded channels and additional meta information, such as arrival times of the stimulation pulses. The following mapping (see `./analysis_scripts/load_xdf.py`) identifies the channels with the short names referred to in the `./analysis_scripts/`:


```python
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
```

- `LFP_1` to `LFP_8` refer to the channels of the Boston Scientific Vercise electrode in the left STN, with `LFP_1` being the tip electrode (most ventral) and `LFP_8` being the most dorsal contact.
- `LFP_9` to `LFP_16` refer to the channels of the Boston Scientific Vercise electrode in the right STN, with `LFP_9` being the tip electrode (most ventral) and `LFP_16` being the most dorsal contact.
- `ECOG_1` to `ECOG_4` refer to the 4 contact Ad-Tech electrodes over M1, with `ECOG_1` being the most caudal electrode and `ECOG_4` being closest to the left borehole (most rostral).
- `EOG_1` was placed above the right eye-brow, `EOG_2` below the right eye, `EOG_3` horizontal left of the left eye and `EOG_4` horizontal right of the right eye.


### HR, RR, GSR
This data is found in the subfolders `./data/sub-p001_ses-*/bv/` (only for day2 and day3) and is provided in ".eeg", ".vhdr", ".vmrk" files.

### Behavioral data
The behavioral data is stored in the subfolders `./data/sub-p001_ses-day2/behavioral/` (etc) within individual folders for each block, containing data as ".yaml" files for each trial. These files contain the coordinates of the target trace as well as the stylus drawing coordinates in screen pixels as (x,y) tuples, and additional meta data.

- `trace_let`: (x, y) coordinates of the drawn trace 
- `trace_pix`: same as `trace_let`
- `cursor_t`: time stamps of the (x, y) samples, relative to the start of the trial
- `template`: (x, y) coordinates of the target trace in relative coordinates
- `template_pix`: (x, y) coordinates of the target trace in pixels
- `theBoxPix`: Corner (x, y) coordinates of the box which is to be activate to signal readiness. (Marker `50`)


### XDF
The LSL files located at /data/sub-p001_ses-day4/lsl/ (etc) contain the marker streams for the behavioral protocol as well as closed-loop processing data for the day 4 closed-loop aDBS measuremens (see the Dareplane paper for details). The XDF files contain various streams, the relevant streams being the `AODataStream`, containing the 24 channels for LFP, ECoG and EOG data, and the `CopyDrawParadigmMarkerStream`, containing the markers of the CopyDraw task. 

- `AODataStream`: LFP, ECoG, EOG (see section on `*.mat` files for the order and of the channels). _Note_: The `nominal_sampling` rate will always show 2000Hz, which was incorrectly specified within the Dareplane module for streaming the Alpha Omega data. The actual sampling frequencies were either `22kHz` (CopyDraw) or `5.5kHz` (benchtop). The actual rate can easily be confirmed by computing the average sample count in the first minutes of recording (using the time stamps of the recordings).

- `CopyDrawParadigmMarkerStream`:
Markers `10` to `21` refer to the start of copy-drawing a trace of the CopyDraw task, while the according `110` to `121` refer to the end of a single trial (~9s later). The marker value `50` refers to a self paced readiness confirmation by the patient - see https://pubmed.ncbi.nlm.nih.gov/31536010/ for details on the CopyDraw task.

The other streams where not used for evaluation but contain the following data:
- `MousePosition`: The mouse position in screen pixels
- `BrainVision RDA Markers`: Markes from the BrainVision RDA stream, redundant to `CopyDrawParadigmMarkerStream`, but less accurate 
- `Keyboard`: Keyboard events
- `AOFctCallStream`: empty
- `MouseButtons`: Mouse button events
- `BrainVision RDA`: The BrainVision RDA stream, containing the HR, RR, GSR data
- `RestParadigmMarkerStream`: empty



## 3. Electrophysiological recordings during a c-VEP speller experiment
These recordings comprise EEG and behavioral of a c-VEP speller experiment as XDF files. The EEG data was recorded from 8 channels (Fz, T7, T8, POz, O1, Oz, O2, Iz)
placed according to the 10-10 system, using a BioSemi ActiveTwo amplifier recording at 512 Hz. This collection contains data from three healthy participants using the c-VEP speller in three runs each. The xdf files are located in the subfolders `./data/data_p001`, `./data/data_p002`, `./data/data_p003`. Each folder contains 4 XDF files with the calibration data and three online runs, suffixed by "a", "b", and "c". The file without suffix contains the calibration data. Additionally, each folder contains a "dp-cvep" folder with the fitted decoding model and meta data.

The xdf files contain two (for training), or three (for online decoding) LSL streams:
- `cvep-speller-stream`: Markers about the paradigm including `start_trial`, `{<prediction_result_dictionary with "prediction_key">}`, `start_feedback`, `stop_feedback`, `start_iti`, `stop_iti`.
- `cvep-decoder-stream`: stream of integers containing decoder predictions, e.g., `[28]` for predicting the 28th characters. 
- `BioSemi`: The streaming data from the BioSemi amplifier, containing 89 channel, but only the above mentioned 8 channels were connected.

## Other files
The `./data` folder contains additional files which are intermediate results derived from the raw data described above. They are generated with the scripts found under /analysis_scripts and are the basis for most of the plots generated for the paper.


# Scripts
The `./analysis_scripts` folder contains pythons scripts used to create the key figures and tables for the Dareplane paper (https://arxiv.org/pdf/2408.01242). File names clearly reference the script to single figures and or tables. All files without "fig" or "table" in the name contain auxiliary functions, e.g. for creating the intermediate results. The scripts were used in the parent directory of the `./data` folder and are configured to work from there. We provide the scripts in separate folder now for better packaging. If you download all folders, simply copy the scripts to the parent directory (e.g., `cp  ./analysis_scripts/* .`); or create a symbolic link in the `./analysis_scripts` directory (e.g., `ln -s ../data`) and run from within the `./analysis_folder`.
Additional evaluation functions for the CopyDraw data are found in `./analysis_scripts/behavior` which are used within the `./analysis_scripts/create_copydraw_scores.py`.


### Scripts with requirements and output

###### chrono_split
Leave out blocks of chronologically sorted pairs of cross validation. Provides
the split functionality used in `./behavior/post_processing/projection.py` and `ecog_decoding.py`.

- `./analysis_scripts/chrono_split.py`

###### create_copydraw_scores
Post processing the behavioral features to CopyDraw scores

- `./analysis_scripts/create_copydraw_scores.py`
  - requires:
    - `./data/sub-p001_ses-*/behavioral/block_01/*.yaml`
  - creates:
    - `./data/sub-p001_ses-*/behavioral/projection_results`
    - `./data/behavioral_data_closed_loop.hdf`
    - `./data/lda_cross_val_*.hdf`

###### ecog_decoding
Decoding CopyDraw scores from ECoG data
- `./analysis_scripts/ecog_decoding.py`
  - requires:
    - `./data/sub-p001_ses-*/ao/*.mat`
    - `./data/sub-p001_ses-*/lsl/*.xdf`
  - creates:
    - `./data/wip_raw_bandpassed_*.pkl`
    - `./data/wip_epochs_bandpassed_*.pkl`
    - `./data/wip_epo_bandpassed_*.pkl`
    - `./data/wip_*_ridge.hdf`
    - `./data/model_*.joblib`
    - `./data/bootstrap_n*_mean_r_*_model_*.npy`


###### load_xdf
This file contains the loading utility functions used for processing xdf and mat files (used in `eco_decoding.py`)
- `./analysis_scripts/load_xdf.py`


###### paper_plots_aux
This files contains auxiliary functions used for creating the plots. Imported in the `./analysis_scripts/plot_*.py` scripts.

- `./analysis_scripts/paper_plots_aux.py`

###### plot_all_chunks_fig9_table4
This script includes the evaluation logic and plotting for the chunking evaluation of the benchtop recordings.

- `./analysis_scripts/plot_all_chunks_fig9_table4.py`
  - requires:
    - `./data/AO_test.xdf`
    - `./data/CT_pico_loop_with_1ms_sleep.xdf`
    - `./data/Arduino_test.xdf`

###### plot_benchmarking_results_fig8_table3
This script inclused the evaluation logic ($\Delta$ calculation) for the benchtop experiments as well as the plotting routines for figure 8 and table 3.

- `./analysis_scripts/plot_benchmarking_results_fig8_table3.py`

  - requires:
    - `./data/AO_test.xdf`
    - `./data/CT_pico_loop_with_1ms_sleep.xdf`
    - `./data/Arduino_test.xdf`

  - creates (also used as intermediate result):
    - `./data/AO_test.p`
    - `./data/CT_pico_loop_with_1ms_sleep.p`
    - `./data/Arduino_test.p`
    

###### plot_ct_stim_artifact_when_full_speed
The code in this scrip was used to evalute an artifact in the benchtop recordings
with the CT EvalKit. The artifact was caused by a code segment in the Dareplane
module for running a while loop at full speed, limiting resources of the same
process for sending LSL data packages. The plots where used in discussion with
the second reviewer of the paper.

- `./analysis_scripts/plot_ct_stim_artifact_when_full_speed.py`
  - requires:
    - `./data/CT_pico_loop_with_1ms_sleep.xdf`
    - `./data/CT_pico_loop_full_speed.xdf`


###### plot_fig11_fig13_behavior_and_decoding
This script contains the code for the evaluation of the CopyDraw (behavioral)
results and the decoding from ECoG data.

- `./analysis_scripts/plot_fig11_fig13_behavior_and_decoding.py`
  - requires:
    - `./data/wip_*_ridge.hdf`
    - `./data/model_*.joblib`
    - `./data/bootstrap_n*_mean_r_*_model_*.npy`
    - `./data/lda_cross_val_*.hdf`

###### plot_fig12_example_trace_plot
This script contains the plotting routines used for the aDBS example trace.

- `./analysis_scripts/plot_fig12_example_trace_plot.py`
  - requires:
    - `./data/sub-p001_ses-day4/lsl/block7_clcopydraw_on.xdf`

###### plot_stim_artifact_extrapolation_fig10
Plot to explain the de-jittering to approximately calculate when the stim pulse arrives at the tissue.

- `./analysis_scripts/plot_stim_artifact_extrapolation_fig10.py`
  - requires:
    - `./data/AO_test.xdf`
