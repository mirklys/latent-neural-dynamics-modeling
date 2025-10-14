import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

project_root = "/home/bobby/repos/latent-neural-dynamics-modeling"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.file_handling import get_child_subchilds_tuples

# from utils.polars import (
#     # load_participant_session_data,  # This will be defined within the dashboard for now
#     get_trial,
# )
from utils.plots import (
    plot_tracing_speed,
    plot_trial_channel,
    plot_psd_heatmap,
    plot_average_psd,
    plot_trial_coordinates,
)

st.set_page_config(layout="wide")

st.title("iEEG Data Analysis Dashboard")

ROOT_PATH = Path(project_root)
DATA_PATH = ROOT_PATH / "resampled_recordings"
PARTICIPANTS_PATH = DATA_PATH / "participants"


@st.cache_data
def get_available_sessions():
    """Gets a list of available participant and session tuples."""
    if not DATA_PATH.exists():
        st.error(f"Data directory not found at: {DATA_PATH}. Please create it.")
        return []
    return get_child_subchilds_tuples(PARTICIPANTS_PATH)


@st.cache_data
def load_participant_session_data(participant: str, session: str):
    """
    Loads the session data for a given participant and session.
    This is a placeholder and returns a dummy DataFrame.
    """
    st.info(f"Loading data for {participant}, {session}...")

    p_partition_path = PARTICIPANTS_PATH / participant / session / "*"

    return pl.read_parquet(p_partition_path)


st.sidebar.header("Session Selection")

available_sessions = get_available_sessions()

if not available_sessions:
    st.sidebar.warning("No sessions found. Check the `preprocessed_data` directory.")
else:
    session_options = [
        f"{p.split('=')[1]} - {s.split('=')[1]}" for p, s in available_sessions
    ]
    selected_option = st.sidebar.selectbox(
        "Select Participant and Session", options=session_options
    )

    if selected_option:
        p_id_str, session_str = selected_option.split(" - ")
        selected_tuple = next(
            (
                (p, s)
                for p, s in available_sessions
                if p.endswith(p_id_str) and s.endswith(session_str)
            ),
            None,
        )

        if selected_tuple and st.sidebar.button("Load Data"):
            participant_id, session = selected_tuple
            session_data = load_participant_session_data(participant_id, session)

            if not session_data.is_empty():
                st.session_state["session_data"] = session_data
                st.session_state["participant_id"] = participant_id
                st.session_state["session"] = session
                st.sidebar.success(f"Data loaded for {p_id_str}, Session {session_str}")
            else:
                st.sidebar.error("Failed to load data.")

st.header("Trial Analysis")

if "session_data" in st.session_state:
    session_data = st.session_state["session_data"]

    trials = sorted(session_data["trial"].unique().to_list())

    lfp_channels = sorted([
        col for col in session_data.columns
        if col.lower().startswith("lfp") and col.split("_")[-1].isdigit()
    ])
    ecog_channels = sorted([
        col for col in session_data.columns
        if col.lower().startswith("ecog") and col.split("_")[-1].isdigit()
    ])

    for trial_num in trials:
        trial_data = session_data.filter(pl.col("trial") == trial_num)
        dbs_on = trial_data["dbs_stim"][0]

        with st.expander(
            f"Trial {trial_num} (DBS: {'ON' if dbs_on else 'OFF'})",
            expanded=trial_num == trials[0],
        ):
            st.markdown(f"### Details for Trial {trial_num}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Channel Time Series")
                selected_lfp = st.selectbox(
                    f"LFP Channel (Trial {trial_num})",
                    lfp_channels,
                    key=f"lfp_{trial_num}",
                )
                selected_ecog = st.selectbox(
                    f"ECoG Channel (Trial {trial_num})",
                    ecog_channels,
                    key=f"ecog_{trial_num}",
                )
                # print selected columns
                st.write(f"Selected LFP: {selected_lfp}, Selected ECoG: {selected_ecog}")
                st.markdown("**Note:** Time series plots may take a moment to render.")
                st.markdown("---")


                if selected_lfp:
                    channel_data = trial_data.select(
                        "time",
                        "chunk_margin",
                        "margined_duration",
                        "dbs_stim",
                        "participant_id",
                        "trial",
                        selected_lfp,
                    ).explode("time", selected_lfp)

                    fig_lfp = plot_trial_channel(channel_data, selected_lfp)
                    st.plotly_chart(fig_lfp, use_container_width=True)

                if selected_ecog:
                    channel_data = trial_data.select(
                        "time",
                        "chunk_margin",
                        "margined_duration",
                        "dbs_stim",
                        "participant_id",
                        "trial",
                        selected_ecog,
                    ).explode("time", selected_ecog)
                    fig_ecog = plot_trial_channel(channel_data, selected_ecog)
                    st.plotly_chart(fig_ecog, use_container_width=True)

            with col2:
                st.subheader("Coordinates and Speed")
                coordinates_data = trial_data.select(
                    "motion_time",
                    "x",
                    "y",
                    "tracing_speed",
                    "chunk_margin",
                    "margined_duration",
                    "dbs_stim",
                    "participant_id",
                    "trial",
                    "session",
                    "block",
                ).explode("motion_time", "x", "y")

                if not coordinates_data.is_empty():
                    fig_coords_time = plot_trial_coordinates(
                        coordinates_data,
                        time="motion_time",
                        plot_over_time=True,
                    )
                    st.plotly_chart(fig_coords_time, use_container_width=True)
                    fig_coords = plot_trial_coordinates(
                        coordinates_data,
                        time="motion_time",
                    )
                    st.plotly_chart(fig_coords, use_container_width=True)

                    fig_speed = plot_tracing_speed(
                        coordinates_data,
                        tracing_speed="tracing_speed",
                        time="motion_time",
                    )
                    st.plotly_chart(fig_speed, use_container_width=True)
                else:
                    st.warning("No coordinate data available for this trial.")

            st.markdown("---")

    # --- PSD Analysis Section ---
    st.header("Power Spectral Density (PSD) Analysis")

    # --- PSD Heatmaps for a selected trial ---
    st.subheader("PSD Heatmap for a Specific Trial")
    selected_trial_psd = st.selectbox(
        "Select Trial for PSD Heatmap", trials, key="psd_trial_selector"
    )
    trial_data_psd = session_data.filter(pl.col("trial") == selected_trial_psd)

    col1_psd, col2_psd = st.columns(2)
    with col1_psd:
        selected_lfp_psd = st.selectbox(
            "Select LFP Channel for PSD Heatmap", lfp_channels, key="lfp_psd_heatmap"
        )
        if selected_lfp_psd and not trial_data_psd.is_empty():
            freqs = np.array(trial_data_psd[f"{selected_lfp_psd}_psd_freq"][0])
            psd_values = np.array(
                [np.array(arr) for arr in trial_data_psd[f"{selected_lfp_psd}_psd_values"][0]]
            )
            fig = plot_psd_heatmap(
                freqs,
                psd_values,
                title=f"LFP PSD Heatmap for {selected_lfp_psd}, Trial {selected_trial_psd}",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2_psd:
        selected_ecog_psd = st.selectbox(
            "Select ECoG Channel for PSD Heatmap", ecog_channels, key="ecog_psd_heatmap"
        )
        if selected_ecog_psd and not trial_data_psd.is_empty():
            freqs = np.array(trial_data_psd[f"{selected_ecog_psd}_psd_freq"][0])
            psd_values = np.array(
                [np.array(arr) for arr in trial_data_psd[f"{selected_ecog_psd}_psd_values"][0]]
            )
            fig = plot_psd_heatmap(
                freqs,
                psd_values,
                title=f"ECoG PSD Heatmap for {selected_ecog_psd}, Trial {selected_trial_psd}",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Average PSD Plots ---
    st.subheader("Average PSD Analysis (DBS ON vs. OFF)")

    dbs_on_data = session_data.filter(pl.col("dbs_stim") == True)
    dbs_off_data = session_data.filter(pl.col("dbs_stim") == False)

    col1_avg, col2_avg = st.columns(2)
    with col1_avg:
        st.markdown("#### Average LFP PSD")
        selected_lfp_avg = st.selectbox(
            "Select LFP Channel for Average PSD", lfp_channels, key="lfp_psd_avg"
        )
        if selected_lfp_avg:
            freqs = np.array(session_data[f"{selected_lfp_avg}_psd_freq"][0])
            psds_on = np.vstack(dbs_on_data[f"{selected_lfp_avg}_psd_values"].to_list())
            psds_off = np.vstack(dbs_off_data[f"{selected_lfp_avg}_psd_values"].to_list())

            fig = plot_average_psd(
                freqs, psds_on, psds_off, title=f"Average LFP PSD for {selected_lfp_avg}"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2_avg:
        st.markdown("#### Average ECoG PSD")
        selected_ecog_avg = st.selectbox(
            "Select ECoG Channel for Average PSD", ecog_channels, key="ecog_psd_avg"
        )
        if selected_ecog_avg:
            freqs = np.array(session_data[f"{selected_ecog_avg}_psd_freq"][0])
            psds_on = np.vstack(dbs_on_data[f"{selected_ecog_avg}_psd_values"].to_list())
            psds_off = np.vstack(dbs_off_data[f"{selected_ecog_avg}_psd_values"].to_list())

            fig = plot_average_psd(
                freqs, psds_on, psds_off, title=f"Average ECoG PSD for {selected_ecog_avg}"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        "Select a participant and session from the sidebar and click 'Load Data' to begin."
    )
