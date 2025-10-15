import streamlit as st
import polars as pl
import numpy as np
import re
from pathlib import Path
import sys
from collections import defaultdict

import os

# Add project root to the Python path
project_root = Path(os.path.dirname(__file__)).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.file_handling import get_child_subchilds_tuples
from utils.plots import (
    plot_tracing_speed,
    plot_trial_channel,
    plot_psd_heatmap,
    plot_average_psd,
    plot_trial_coordinates,
)

st.set_page_config(layout="wide")

st.title("iEEG Data Analysis Dashboard")

# --- Data Loading and Caching ---
DATA_PATH = project_root / "resampled_recordings"
PARTICIPANTS_PATH = DATA_PATH / "participants"


def natural_sort_key(s):
    """Sort strings in a natural order (e.g., 'LFP_1', 'LFP_2', 'LFP_10')."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


@st.cache_data
def get_participant_sessions():
    """Gets a dictionary mapping participants to their available sessions."""
    if not PARTICIPANTS_PATH.exists():
        st.error(f"Data directory not found at: {PARTICIPANTS_PATH}")
        return {}

    session_tuples = get_child_subchilds_tuples(PARTICIPANTS_PATH)
    participant_sessions = defaultdict(list)
    for p, s in session_tuples:
        p_id = p.split("=")[1]
        s_id = s.split("=")[1]
        participant_sessions[p_id].append(s_id)

    for p_id in participant_sessions:
        participant_sessions[p_id].sort(key=natural_sort_key)

    return dict(sorted(participant_sessions.items()))


@st.cache_data
def load_participant_session_data(participant_id: str, session: str):
    """Loads the session data for a given participant and session."""
    st.info(f"Loading data for P{participant_id}, Session {session}...")

    p_partition = f"participant_id={participant_id}"
    s_partition = f"session={session}"

    p_partition_path = PARTICIPANTS_PATH / p_partition / s_partition / "*"

    try:
        df = pl.read_parquet(p_partition_path)
        if df.is_empty():
            st.warning("No data found for the selected session.")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None


# --- Sidebar for Session Selection ---
st.sidebar.header("Session Selection")
participant_sessions = get_participant_sessions()

if not participant_sessions:
    st.sidebar.warning("No participant data found.")
else:
    selected_participant_id = st.sidebar.selectbox(
        "Select Participant", options=list(participant_sessions.keys())
    )

    if selected_participant_id:
        available_sessions = participant_sessions[selected_participant_id]
        selected_session = st.sidebar.selectbox(
            "Select Session", options=available_sessions
        )

        if st.sidebar.button("Load Data"):
            data = load_participant_session_data(
                selected_participant_id, selected_session
            )
            if data is not None and not data.is_empty():
                st.session_state["session_data"] = data
                st.session_state["participant_id"] = selected_participant_id
                st.session_state["session"] = selected_session
                st.sidebar.success(
                    f"Data loaded for Participant {selected_participant_id}, Session {selected_session}"
                )
            else:
                st.sidebar.error("Failed to load data or data is empty.")
                if "session_data" in st.session_state:
                    del st.session_state["session_data"]

# --- Main Panel for Analysis ---
if "session_data" not in st.session_state:
    st.info(
        "Select a participant and session from the sidebar and click 'Load Data' to begin."
    )
else:
    session_data = st.session_state["session_data"]

    # --- Block and Trial Selection ---
    st.header("Data Selection")
    blocks = sorted(session_data["block"].unique().to_list())
    selected_block = st.selectbox("Select a Block to Analyze", options=blocks)

    trial_data = None
    if selected_block:
        block_data = session_data.filter(pl.col("block") == selected_block)
        trials_in_block = sorted(block_data["trial"].unique().to_list())
        selected_trial = st.selectbox(
            "Select a Trial to Render", options=trials_in_block
        )

        if selected_trial:
            trial_data = block_data.filter(pl.col("trial") == selected_trial)

    lfp_channels = sorted(
        [
            col
            for col in session_data.columns
            if col.lower().startswith("lfp") and ("psd" not in col and "epochs" not in col)
        ],
        key=natural_sort_key,
    )
    ecog_channels = sorted(
        [
            col
            for col in session_data.columns
            if col.lower().startswith("ecog") and ("psd" not in col and "epochs" not in col)
        ],
        key=natural_sort_key,
    )

    tab1, tab2 = st.tabs(["Time-Series Analysis", "Frequency (PSD) Analysis"])

    with tab1:
        st.header("Time-Series Analysis")
        if trial_data is not None:
            if trial_data.is_empty():
                st.warning(f"No data for the selected trial.")
            else:
                dbs_on = trial_data["dbs_stim"][0]
                st.subheader(
                    f"Analyzing Trial {trial_data['trial'][0]} (DBS: {'ON' if dbs_on else 'OFF'})"
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Channel Time Series")
                    selected_lfp = st.selectbox(
                        "Select LFP Channel",
                        lfp_channels,
                        key=f"lfp_ts_{trial_data['trial'][0]}",
                    )
                    if selected_lfp:
                        df_exploded = trial_data.select(
                            "time",
                            "chunk_margin",
                            "margined_duration",
                            "dbs_stim",
                            "participant_id",
                            "trial",
                            selected_lfp,
                        ).explode("time", selected_lfp)
                        st.plotly_chart(
                            plot_trial_channel(df_exploded, selected_lfp),
                            use_container_width=True,
                        )

                    selected_ecog = st.selectbox(
                        "Select ECoG Channel",
                        ecog_channels,
                        key=f"ecog_ts_{trial_data['trial'][0]}",
                    )
                    if selected_ecog:
                        df_exploded = trial_data.select(
                            "time",
                            "chunk_margin",
                            "margined_duration",
                            "dbs_stim",
                            "participant_id",
                            "trial",
                            selected_ecog,
                        ).explode("time", selected_ecog)
                        st.plotly_chart(
                            plot_trial_channel(df_exploded, selected_ecog),
                            use_container_width=True,
                        )
                with col2:
                    st.markdown("#### Coordinates and Speed")
                    coords_data = trial_data.select(
                        "motion_time",
                        "x",
                        "y",
                        "tracing_speed",
                        "participant_id",
                        "session",
                        "trial",
                    ).explode("motion_time", "x", "y", "tracing_speed")
                    st.plotly_chart(
                        plot_trial_coordinates(
                            coords_data, time="motion_time", plot_over_time=True
                        ),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        plot_trial_coordinates(coords_data, time="motion_time"),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        plot_tracing_speed(coords_data, time="motion_time"),
                        use_container_width=True,
                    )
        else:
            st.info("Select a block and trial to view time-series data.")

    with tab2:
        st.header("Power Spectral Density (PSD) Analysis")

        st.subheader("PSD Heatmap for a Specific Trial")
        if trial_data is not None:
            if not trial_data.is_empty():
                selected_trial_psd = trial_data["trial"][0]
                col1_psd, col2_psd = st.columns(2)
                with col1_psd:
                    selected_lfp_psd = st.selectbox(
                        "LFP Channel Heatmap", lfp_channels, key="lfp_psd_heatmap"
                    )
                    if selected_lfp_psd:
                        freqs = np.array(trial_data[f"{selected_lfp_psd}_psd_freq"][0])
                        psd_values = np.array(
                            trial_data[f"{selected_lfp_psd}_psd_values"][0]
                        )
                        fig = plot_psd_heatmap(
                            freqs,
                            psd_values,
                            title=f"LFP PSD Heatmap ({selected_lfp_psd}, Trial {selected_trial_psd})",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                with col2_psd:
                    selected_ecog_psd = st.selectbox(
                        "ECoG Channel Heatmap", ecog_channels, key="ecog_psd_heatmap"
                    )
                    if selected_ecog_psd:
                        freqs = np.array(trial_data[f"{selected_ecog_psd}_psd_freq"][0])
                        psd_values = np.array(
                            trial_data[f"{selected_ecog_psd}_psd_values"][0]
                        )
                        fig = plot_psd_heatmap(
                            freqs,
                            psd_values,
                            title=f"ECoG PSD Heatmap ({selected_ecog_psd}, Trial {selected_trial_psd})",
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a block and trial to view its PSD heatmap.")

        st.markdown("---")
        st.subheader("Average PSD (DBS ON vs. OFF) Across Session")
        dbs_on_data = session_data.filter(pl.col("dbs_stim") == True)
        dbs_off_data = session_data.filter(pl.col("dbs_stim") == False)

        col1_avg, col2_avg = st.columns(2)
        with col1_avg:
            st.markdown("#### Average LFP PSD")
            selected_lfp_avg = st.multiselect(
                "Select LFP Channels", lfp_channels, key="lfp_psd_avg"
            )
            if selected_lfp_avg:
                psd_data = {}
                freqs = None
                for ch in selected_lfp_avg:
                    if freqs is None:
                        freqs = np.array(session_data[f"{ch}_psd_freq"][0])
                    psds_on = (
                        np.vstack(dbs_on_data[f"{ch}_psd_values"].to_list())
                        if not dbs_on_data.is_empty()
                        else np.array([])
                    )
                    psds_off = (
                        np.vstack(dbs_off_data[f"{ch}_psd_values"].to_list())
                        if not dbs_off_data.is_empty()
                        else np.array([])
                    )
                    psd_data[ch] = {"on": psds_on, "off": psds_off}

                if freqs is not None:
                    fig = plot_average_psd(freqs, psd_data, title="Average LFP PSD")
                    st.plotly_chart(fig, use_container_width=True)

        with col2_avg:
            st.markdown("#### Average ECoG PSD")
            selected_ecog_avg = st.multiselect(
                "Select ECoG Channels", ecog_channels, key="ecog_psd_avg"
            )
            if selected_ecog_avg:
                psd_data = {}
                freqs = None
                for ch in selected_ecog_avg:
                    if freqs is None:
                        freqs = np.array(session_data[f"{ch}_psd_freq"][0])
                    psds_on = (
                        np.vstack(dbs_on_data[f"{ch}_psd_values"].to_list())
                        if not dbs_on_data.is_empty()
                        else np.array([])
                    )
                    psds_off = (
                        np.vstack(dbs_off_data[f"{ch}_psd_values"].to_list())
                        if not dbs_off_data.is_empty()
                        else np.array([])
                    )
                    psd_data[ch] = {"on": psds_on, "off": psds_off}

                if freqs is not None:
                    fig = plot_average_psd(freqs, psd_data, title="Average ECoG PSD")
                    st.plotly_chart(fig, use_container_width=True)
