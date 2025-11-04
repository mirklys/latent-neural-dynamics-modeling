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
from typing import Optional

from utils.file_handling import get_child_subchilds_tuples
from utils.plots import (
    plot_tracing_speed,
    plot_trial_channel,
    plot_psd_heatmap,
    plot_average_psd,
    plot_trial_coordinates,
)

from utils.logger import setup_logger

logger = setup_logger("dashboard_logs", name=__name__)

st.set_page_config(layout="wide")

st.title("iEEG & Motion Analysis Dashboard")

# Additional imports for predictions tab
import plotly.graph_objects as go
from training.components.tester import Tester
from utils.config import get_config

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

    tab1, tab2, tab3 = st.tabs(
        ["Time-Series Analysis", "Frequency (PSD) Analysis", "Model Predictions"]
    )

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

    with tab3:
        st.header("Model Predictions")

        RESULTS_ROOT = project_root / "results"

        def _list_variants(results_root: Path):
            if not results_root.exists():
                return []
            return sorted([p.name for p in results_root.iterdir() if p.is_dir()])

        def _list_run_timestamps(variant_dir: Path):
            ts = set()
            for p in variant_dir.glob("val_results_*"):
                name = p.name
                if name.startswith("val_results_"):
                    ts.add(name.replace("val_results_", ""))
            for p in variant_dir.glob("model_*.pkl"):
                name = p.name
                if name.startswith("model_") and name.endswith(".pkl"):
                    ts.add(name.replace("model_", "").replace(".pkl", ""))
            return sorted(list(ts))

        def _config_for_variant(variant_name: str) -> Optional[Path]:
            cfg = project_root / "training" / "setups" / f"{variant_name}.yaml"
            return cfg if cfg.exists() else None

        variants = _list_variants(RESULTS_ROOT)
        if len(variants) == 0:
            st.info("No result variants found under results/.")
        else:
            variant = st.selectbox(
                "Model variant", options=variants, key="pred_variant"
            )
            variant_dir = RESULTS_ROOT / variant
            runs = _list_run_timestamps(variant_dir)
            if len(runs) == 0:
                st.info("No runs found for this variant yet. Train a model first.")
            else:
                run_ts = st.selectbox("Run timestamp", options=runs, key="pred_run")
                cfg_path = _config_for_variant(variant)
                if cfg_path is None:
                    st.error(
                        f"Config not found for variant '{variant}'. Expected at training/setups/{variant}.yaml"
                    )
                else:

                    @st.cache_resource(show_spinner=True)
                    def _cached_predictions(config_path: str, run_timestamp: str):
                        tester = Tester.from_config_file(
                            config_path, run_timestamp=run_timestamp
                        )
                        return tester.run_predictions()

                    if st.button("Run predictions", key="btn_run_predictions"):
                        st.session_state["predictions_key"] = (str(cfg_path), run_ts)

                    pred_key = st.session_state.get("predictions_key")
                    if (
                        pred_key
                        and pred_key[0] == str(cfg_path)
                        and pred_key[1] == run_ts
                    ):
                        with st.spinner("Running predictions..."):
                            try:
                                pred_results = _cached_predictions(
                                    str(cfg_path), run_ts
                                )
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                                pred_results = None
                        if pred_results is not None:
                            split = st.selectbox(
                                "Split",
                                options=["train", "val", "test"],
                                key="pred_split",
                            )
                            split_res = pred_results.get(split)
                            if not split_res:
                                st.info("No results for selected split.")
                            else:
                                Y_true = split_res["Y"]
                                Yp = split_res["Yp"]
                                Zp = split_res["Zp"]
                                Xp = split_res["Xp"]
                                pearson_tr = split_res["pearson_per_channel"]
                                pearson_mean = split_res["pearson_mean"]

                                n_trials = len(Y_true)
                                trial_indices = list(range(n_trials))
                                trial_idx = st.selectbox(
                                    "Trial", options=trial_indices, key="pred_trial"
                                )

                                y_t = np.array(Y_true[trial_idx])
                                y_p = np.array(Yp[trial_idx])
                                z_p = (
                                    None
                                    if Zp[trial_idx] is None
                                    else np.array(Zp[trial_idx])
                                )
                                x_p = np.array(Xp[trial_idx])

                                r_list = pearson_tr[trial_idx] if pearson_tr else []
                                r_mean = (
                                    pearson_mean[trial_idx] if pearson_mean else np.nan
                                )
                                mean_str = (
                                    f"{r_mean:.4f}" if not np.isnan(r_mean) else "nan"
                                )
                                st.markdown(
                                    f"Pearson per channel: {r_list} | Mean: {mean_str}"
                                )

                                # Metadata and time vector
                                meta_time = split_res.get("time", [])
                                offsets = split_res.get("offset", [])
                                t_abs = (
                                    meta_time[trial_idx]
                                    if meta_time and len(meta_time) > trial_idx
                                    else None
                                )
                                cm_list = split_res.get("chunk_margin", [])
                                md_list = split_res.get("margined_duration", [])
                                stim_list = split_res.get("stim", [])
                                pid_list = split_res.get("participant_id", [])
                                ses_list = split_res.get("session", [])
                                blk_list = split_res.get("block", [])
                                tri_list = split_res.get("trial", [])
                                chan_names = split_res.get("input_channels", [])

                                # Header info
                                hdr_pid = (
                                    pid_list[trial_idx]
                                    if pid_list
                                    else st.session_state.get("participant_id")
                                )
                                hdr_ses = (
                                    ses_list[trial_idx]
                                    if ses_list
                                    else st.session_state.get("session")
                                )
                                hdr_blk = (
                                    blk_list[trial_idx]
                                    if blk_list
                                    else st.session_state.get("block")
                                )
                                hdr_tri = tri_list[trial_idx] if tri_list else trial_idx
                                st.subheader(
                                    f"Participant {hdr_pid} | Session {hdr_ses} | Block {hdr_blk} | Trial {hdr_tri}"
                                )

                                n_samples = y_t.shape[0]
                                if t_abs is None or (
                                    hasattr(t_abs, "__len__")
                                    and len(t_abs) != n_samples
                                ):
                                    dur = md_list[trial_idx] if md_list else None
                                    if dur is not None:
                                        t_abs = np.linspace(0.0, float(dur), n_samples)
                                    else:
                                        t_abs = np.arange(n_samples)
                                else:
                                    t_abs = np.array(t_abs)
                                t_offset = (
                                    float(offsets[trial_idx])
                                    if offsets and len(offsets) > trial_idx and offsets[trial_idx] is not None
                                    else 0.0
                                )
                                t_abs = t_abs + t_offset

                                n_chan = y_t.shape[1] if y_t.ndim == 2 else 1
                                if (
                                    chan_names
                                    and isinstance(chan_names, list)
                                    and len(chan_names) == n_chan
                                ):
                                    channel_options = chan_names
                                else:
                                    channel_options = [f"ch{i}" for i in range(n_chan)]
                                selected_name = st.selectbox(
                                    "Channel for Y/Yp plot",
                                    options=channel_options,
                                    index=0,
                                    key="pred_chan",
                                )
                                c = (
                                    channel_options.index(selected_name)
                                    if n_chan > 1
                                    else 0
                                )

                                if y_t.ndim == 2 and y_t.shape[0] != len(t_abs) and y_t.shape[1] == len(t_abs):
                                    y_t = y_t.T
                                if y_p is not None and y_p.ndim == 2 and y_p.shape[0] != len(t_abs) and y_p.shape[1] == len(t_abs):
                                    y_p = y_p.T

                                y_true_c = y_t.squeeze() if n_chan == 1 else y_t[:, c]
                                y_pred_c = (
                                    None
                                    if y_p is None
                                    else (y_p.squeeze() if n_chan == 1 else y_p[:, c])
                                )
                                inp_mean = split_res.get("input_mean")
                                inp_std = split_res.get("input_std")

                                if inp_mean is not None and inp_std is not None:
                                    mu = np.array(inp_mean).squeeze()
                                    sd = np.array(inp_std).squeeze()
                                    mu_c = mu if np.ndim(mu) == 0 or n_chan == 1 else mu[c]
                                    sd_c = sd if np.ndim(sd) == 0 or n_chan == 1 else sd[c]
                                    if y_true_c is not None:
                                        y_true_c = y_true_c * sd_c + mu_c
                                    if y_pred_c is not None:
                                        y_pred_c = y_pred_c * sd_c + mu_c

                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=t_abs, y=y_true_c, name="Y_true (µV)", mode="lines"
                                    )
                                )
                                if y_pred_c is not None:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=t_abs,
                                            y=y_pred_c,
                                            name="Y_pred (µV)",
                                            mode="lines",
                                        )
                                    )

                                cm = cm_list[trial_idx] if cm_list else None
                                dur = md_list[trial_idx] if md_list else None
                                if dur is not None:
                                    event_start = t_offset + float(cm) if cm is not None else t_abs[0]
                                    event_end = t_offset + float(dur) - (float(cm) if cm is not None else 0.0)
                                    fig.add_vrect(
                                        x0=event_start,
                                        x1=event_end,
                                        fillcolor="rgba(0, 100, 0, 0.1)",
                                        layer="below",
                                        line_width=0,
                                    )
                                    fig.add_vline(
                                        x=event_start,
                                        line_dash="dash",
                                        line_color="green",
                                    )
                                    fig.add_vline(
                                        x=event_end, line_dash="dash", line_color="red"
                                    )

                                r_ch = (
                                    r_list[c] if r_list and c < len(r_list) else np.nan
                                )
                                chan_name = selected_name
                                fig.update_layout(
                                    title=f"Y and Y_p — {chan_name} (r={r_ch:.3f})",
                                    xaxis_title="Time (s)",
                                    yaxis_title="Amplitude (µV)",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                if x_p is not None:
                                    figx = go.Figure()
                                    t_x = (
                                        np.linspace(t_abs[0], t_abs[-1], x_p.shape[0])
                                        if len(t_abs) != x_p.shape[0]
                                        else t_abs
                                    )
                                    nx = x_p.shape[1] if x_p.ndim == 2 else 1
                                    x_min = float(np.nanmin(x_p))
                                    x_max = float(np.nanmax(x_p))
                                    for d in range(nx):
                                        series = x_p[:, d] if nx > 1 else x_p.squeeze()
                                        figx.add_trace(
                                            go.Scatter(
                                                x=t_x,
                                                y=series,
                                                name=f"X_p[{d}]",
                                                mode="lines",
                                            )
                                        )
                                    if dur is not None:
                                        event_start = t_offset + float(cm) if cm is not None else t_x[0]
                                        event_end = t_offset + float(dur) - (float(cm) if cm is not None else 0.0)
                                        figx.add_vrect(
                                            x0=event_start,
                                            x1=event_end,
                                            fillcolor="rgba(0, 100, 0, 0.1)",
                                            layer="below",
                                            line_width=0,
                                        )
                                        figx.add_vline(x=event_start, line_dash="dash", line_color="green")
                                        figx.add_vline(x=event_end, line_dash="dash", line_color="red")
                                    figx.update_layout(
                                        title=f"Latent states X_p — Trial {trial_idx}",
                                        xaxis_title="Time (s)",
                                        yaxis_title="Raw value",
                                        xaxis_range=[t_x[0], t_x[-1]],
                                        yaxis_range=[x_min, x_max],
                                    )
                                    st.plotly_chart(figx, use_container_width=True)

                                if z_p is not None:
                                    figz = go.Figure()
                                    t_z = (
                                        np.linspace(t_abs[0], t_abs[-1], z_p.shape[0])
                                        if len(t_abs) != z_p.shape[0]
                                        else t_abs
                                    )
                                    nz = z_p.shape[1] if z_p.ndim == 2 else 1
                                    z_min = float(np.nanmin(z_p))
                                    z_max = float(np.nanmax(z_p))
                                    for d in range(nz):
                                        series = z_p[:, d] if nz > 1 else z_p.squeeze()
                                        figz.add_trace(
                                            go.Scatter(
                                                x=t_z,
                                                y=series,
                                                name=f"Z_p[{d}]",
                                                mode="lines",
                                            )
                                        )
                                    if dur is not None:
                                        event_start = t_offset + float(cm) if cm is not None else t_z[0]
                                        event_end = t_offset + float(dur) - (float(cm) if cm is not None else 0.0)
                                        figz.add_vrect(
                                            x0=event_start,
                                            x1=event_end,
                                            fillcolor="rgba(0, 100, 0, 0.1)",
                                            layer="below",
                                            line_width=0,
                                        )
                                        figz.add_vline(x=event_start, line_dash="dash", line_color="green")
                                        figz.add_vline(x=event_end, line_dash="dash", line_color="red")
                                    figz.update_layout(
                                        title=f"Aux predictions Z_p — Trial {trial_idx}",
                                        xaxis_title="Time (s)",
                                        yaxis_title="Value",
                                        xaxis_range=[t_z[0], t_z[-1]],
                                        yaxis_range=[z_min, z_max],
                                    )
                                    st.plotly_chart(figz, use_container_width=True)
