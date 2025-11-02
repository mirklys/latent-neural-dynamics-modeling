import streamlit as st
import polars as pl
import numpy as np
import re
from pathlib import Path
import sys
from collections import defaultdict

import os

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

st.title("iEEG & Motion Analysis Dashboard")

DATA_PATH = project_root / "resampled_recordings"
PARTICIPANTS_PATH = DATA_PATH / "participants_2"


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


@st.cache_data
def get_participant_sessions():
    if not PARTICIPANTS_PATH.exists():
        st.error(f"Data directory not found at: {PARTICIPANTS_PATH}")
        return {}

    session_tuples = get_child_subchilds_tuples(PARTICIPANTS_PATH)
    participant_sessions = defaultdict(lambda: defaultdict(list))
    for _1, p, s, b in session_tuples:
        p_id = p.split("=")[1]
        s_id = s.split("=")[1]
        b_id = b.split("=")[1]
        if b_id not in participant_sessions[p_id][s_id]:
            participant_sessions[p_id][s_id].append(b_id)

    for p_id in participant_sessions:
        for s_id in participant_sessions[p_id]:
            participant_sessions[p_id][s_id].sort(key=natural_sort_key)
        participant_sessions[p_id] = dict(
            sorted(
                participant_sessions[p_id].items(),
                key=lambda kv: natural_sort_key(kv[0]),
            )
        )

    return dict(sorted(participant_sessions.items()))


@st.cache_data
def load_participant_block_data(participant_id: str, session: str, block: str):
    block_msg = f", Block {block}"
    st.info(f"Loading data for P{participant_id}, Session {session}{block_msg}...")

    p_partition = f"participant_id={participant_id}"
    s_partition = f"session={session}"
    b_partition = f"block={block}"

    p_partition_path = PARTICIPANTS_PATH / p_partition / s_partition / b_partition / "*"

    print(f"Loading data from: {p_partition_path}")
    df = pl.read_parquet(p_partition_path)
    return df


st.sidebar.header("Selection")
participant_sessions = get_participant_sessions()

selected_participant_id = st.sidebar.selectbox(
    "Participant", options=list(participant_sessions.keys())
)

sessions_dict = participant_sessions[selected_participant_id]
selected_session = st.sidebar.selectbox("Session", options=list(sessions_dict.keys()))
blocks = sessions_dict[selected_session]
selected_block = st.sidebar.selectbox("Block", options=blocks)

if st.sidebar.button("Load Data") and selected_session:
    data = load_participant_block_data(
        selected_participant_id, selected_session, selected_block
    )
    st.session_state["block_data"] = data
    st.session_state["participant_id"] = selected_participant_id
    st.session_state["session"] = selected_session
    st.session_state["block"] = selected_block
    st.sidebar.success(
        f"Loaded: P{selected_participant_id} | S{selected_session} | B{selected_block}"
    )

if "block_data" not in st.session_state:
    st.info(
        "Select a participant and session from the sidebar and click 'Load Data' to begin."
    )
else:
    block_data = st.session_state["block_data"]
    stim_col = "stim"

    st.header("Data Selection")
    selected_block = st.session_state.get("block")

    trials_in_block = sorted(block_data["trial"].unique().to_list())
    selected_trial = st.selectbox(
        "Select a Trial to Render",
        options=trials_in_block,
    )

    trial_data = block_data.filter(pl.col("trial") == selected_trial)

    lfp_channels = sorted(
        [
            col
            for col in block_data.columns
            if col.lower().startswith("lfp")
            and ("psd" not in col and "epochs" not in col)
        ],
        key=natural_sort_key,
    )
    ecog_channels = sorted(
        [
            col
            for col in block_data.columns
            if col.lower().startswith("ecog")
            and ("psd" not in col and "epochs" not in col)
        ],
        key=natural_sort_key,
    )

    tab1, tab2 = st.tabs(["Time-Series Analysis", "Frequency (PSD) Analysis"])

    with tab1:
        st.header("Time-Series Analysis")
        if trial_data is not None:
            stim_on = trial_data["stim"][0]
            st.subheader(
                f"""Participant {st.session_state.get('participant_id')} Session {st.session_state.get('session')} 
                Block{selected_block} Trial{trial_data['trial'][0]} Stim {stim_on}"""
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
                        "stim",
                        "participant_id",
                        "session",
                        "block",
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
                        "stim",
                        "participant_id",
                        "session",
                        "block",
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
                    "block",
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
                show_rel_axis = st.checkbox(
                    "Show relative time (top axis)",
                    value=True,
                    help="Uses trial onset as zero.",
                )
                tvec = np.array(trial_data["time_original"][0])
                onset = float(trial_data["onset"][0])
                n_epochs = len(trial_data[f"{lfp_channels[0]}_psd_values"][0])
                times_abs = np.linspace(tvec.min(), tvec.max(), n_epochs)

                with col1_psd:
                    selected_lfp_psd = st.selectbox(
                        "LFP Channel Heatmap", lfp_channels, key="lfp_psd_heatmap"
                    )
                    if selected_lfp_psd:
                        freqs = np.array(trial_data[f"{selected_lfp_psd}_psd_freq"][0])
                        psd_values = np.array(
                            trial_data[f"{selected_lfp_psd}_psd_values"][0]
                        )
                        heat_title = f"""Participant {st.session_state.get('participant_id')}
                            Session {st.session_state.get('session')} Block{selected_block} 
                            Trial {selected_trial_psd} {selected_lfp_psd}"""
                        fig = plot_psd_heatmap(
                            freqs,
                            psd_values,
                            title=heat_title,
                            times_abs=times_abs,
                            add_rel_axis=show_rel_axis,
                            rel_offset=onset,
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
                        heat_title = f"""Participant {st.session_state.get('participant_id')}
                            Session {st.session_state.get('session')} Block{selected_block} 
                            Trial {selected_trial_psd} {selected_ecog_psd}"""
                        fig = plot_psd_heatmap(
                            freqs,
                            psd_values,
                            title=heat_title,
                            times_abs=times_abs,
                            add_rel_axis=show_rel_axis,
                            rel_offset=onset,
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a trial to view its PSD heatmap.")

        st.markdown("---")
        st.subheader("Average PSD (Stim ON vs. OFF) Across Session")
        dbs_on_data = block_data.filter(pl.col(stim_col) == "on")
        dbs_off_data = block_data.filter(pl.col(stim_col) == "off")

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
                    if freqs is None and not block_data.is_empty():
                        freqs = np.array(block_data[f"{ch}_psd_freq"][0])

                    def _per_trial_means(df, col):
                        means = []
                        if df.is_empty():
                            return means
                        for trial_psds in df[col].to_list():
                            arr = np.array(trial_psds)
                            if arr.ndim == 1:
                                means.append(arr)
                            elif arr.ndim >= 2 and arr.size > 0:
                                means.append(arr.mean(axis=0))
                        return means

                    on_means = _per_trial_means(dbs_on_data, f"{ch}_psd_values")
                    off_means = _per_trial_means(dbs_off_data, f"{ch}_psd_values")

                    psds_on = np.vstack(on_means) if len(on_means) > 0 else np.array([])
                    psds_off = (
                        np.vstack(off_means) if len(off_means) > 0 else np.array([])
                    )

                    psd_data[ch] = {"on": psds_on, "off": psds_off}

                if freqs is not None:
                    title_avg = f"""Average LFP PSD 
                         Participant {st.session_state.get('participant_id')}
                        Session {st.session_state.get('session')} Block{selected_block} 
                        Trial {selected_trial_psd}"""
                    fig = plot_average_psd(freqs, psd_data, title=title_avg)
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
                    if freqs is None and not block_data.is_empty():
                        freqs = np.array(block_data[f"{ch}_psd_freq"][0])

                    def _per_trial_means(df, col):
                        means = []
                        if df.is_empty():
                            return means
                        for trial_psds in df[col].to_list():
                            arr = np.array(trial_psds)
                            if arr.ndim == 1:
                                means.append(arr)
                            elif arr.ndim >= 2 and arr.size > 0:
                                means.append(arr.mean(axis=0))
                        return means

                    on_means = _per_trial_means(dbs_on_data, f"{ch}_psd_values")
                    off_means = _per_trial_means(dbs_off_data, f"{ch}_psd_values")

                    psds_on = np.vstack(on_means) if len(on_means) > 0 else np.array([])
                    psds_off = (
                        np.vstack(off_means) if len(off_means) > 0 else np.array([])
                    )

                    psd_data[ch] = {"on": psds_on, "off": psds_off}

                if freqs is not None:
                    title_avg = f"""Average ECoG PSD 
                         Participant {st.session_state.get('participant_id')}
                        Session {st.session_state.get('session')} Block{selected_block} 
                        Trial {selected_trial_psd}"""
                    fig = plot_average_psd(freqs, psd_data, title=title_avg)
                    st.plotly_chart(fig, use_container_width=True)
