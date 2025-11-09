import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional

from training.components.tester import Tester


def model_predictions_tab(project_root):
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
        variant = st.selectbox("Model variant", options=variants, key="pred_variant")
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
                    tester.run_predictions()
                    return tester.results

                if st.button("Run predictions", key="btn_run_predictions"):
                    st.session_state["predictions_key"] = (str(cfg_path), run_ts)

                pred_key = st.session_state.get("predictions_key")
                if pred_key and pred_key[0] == str(cfg_path) and pred_key[1] == run_ts:
                    with st.spinner("Running predictions..."):
                        try:
                            pred_results = _cached_predictions(str(cfg_path), run_ts)
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
                            r_mean = pearson_mean[trial_idx] if pearson_mean else np.nan
                            mean_str = (
                                f"{r_mean:.4f}" if not np.isnan(r_mean) else "nan"
                            )
                            st.markdown(
                                f"Pearson per channel: {r_list} | Mean: {mean_str}"
                            )

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
                            meta_time_margined = split_res.get("time_margined", [])
                            cm_samples_list = split_res.get("chunk_margin_samples_list", [])

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
                                hasattr(t_abs, "__len__") and len(t_abs) != n_samples
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
                                if offsets
                                and len(offsets) > trial_idx
                                and offsets[trial_idx] is not None
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

                            if (
                                y_t.ndim == 2
                                and y_t.shape[0] != len(t_abs)
                                and y_t.shape[1] == len(t_abs)
                            ):
                                y_t = y_t.T
                            if (
                                y_p is not None
                                and y_p.ndim == 2
                                and y_p.shape[0] != len(t_abs)
                                and y_p.shape[1] == len(t_abs)
                            ):
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
                                    x=t_abs,
                                    y=y_true_c,
                                    name="Y_true (µV)",
                                    mode="lines",
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

                            r_ch = r_list[c] if r_list and c < len(r_list) else np.nan
                            chan_name = selected_name
                            fig.update_layout(
                                title=f"Y and Y_p — {chan_name} (r={r_ch:.3f})",
                                xaxis_title="Time (s)",
                                yaxis_title="Amplitude (µV)",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Forecast visualization, if available
                            f_res = split_res.get("forecast")
                            if f_res is not None:
                                try:
                                    m = int(f_res.get("m", 0))
                                    margin_samples = int(f_res.get("margin_samples", 0))
                                    t_abs_unmargined = np.array(meta_time_margined[trial_idx])
                                    m_samples = cm_samples_list[trial_idx]
                                    y_concat = f_res.get("Y_concat_for_plot", [None])[
                                        trial_idx
                                    ]
                                    y_future_true = f_res.get("Y_future_true", [None])[
                                        trial_idx
                                    ]
                                    y_future_pred = f_res.get("Y_future_pred", [None])[
                                        trial_idx
                                    ]
                                    r_fore_list = f_res.get(
                                        "pearson_per_channel", [None]
                                    )[trial_idx]

                                    if (
                                        y_concat is not None
                                        and y_future_true is not None
                                        and y_future_pred is not None
                                        and m > 0
                                    ):
                                        y_concat = np.array(y_concat)
                                        y_future_true = np.array(y_future_true)
                                        y_future_pred = np.array(y_future_pred)

                                        if (
                                            y_concat.ndim == 2
                                            and y_concat.shape[0] != len(t_abs_unmargined)
                                            and y_concat.shape[1] == len(t_abs_unmargined)
                                        ):
                                            y_concat = y_concat.T
                                        if (
                                            y_future_true.ndim == 2
                                            and y_future_true.shape[1]
                                            != (1 if n_chan == 1 else n_chan)
                                        ):
                                            y_future_true = y_future_true
                                        if (
                                            y_future_pred.ndim == 2
                                            and y_future_pred.shape[1]
                                            != (1 if n_chan == 1 else n_chan)
                                        ):
                                            y_future_pred = y_future_pred

                                        # Select channel
                                        y_concat_c = (
                                            y_concat.squeeze()
                                            if n_chan == 1
                                            else y_concat[:, c]
                                        )
                                        y_ft_c = (
                                            y_future_true.squeeze()
                                            if n_chan == 1
                                            else y_future_true[:, c]
                                        )
                                        y_fp_c = (
                                            y_future_pred.squeeze()
                                            if n_chan == 1
                                            else y_future_pred[:, c]
                                        )

                                        # Denormalize if stats available
                                        if inp_mean is not None and inp_std is not None:
                                            mu = np.array(inp_mean).squeeze()
                                            sd = np.array(inp_std).squeeze()
                                            mu_c = (
                                                mu
                                                if np.ndim(mu) == 0 or n_chan == 1
                                                else mu[c]
                                            )
                                            sd_c = (
                                                sd
                                                if np.ndim(sd) == 0 or n_chan == 1
                                                else sd[c]
                                            )
                                            y_concat_c = y_concat_c * sd_c + mu_c
                                            y_ft_c = y_ft_c * sd_c + mu_c
                                            y_fp_c = y_fp_c * sd_c + mu_c

                                        T = len(y_concat_c)
                                        Tpast = max(0, T - m_samples)
                                        t_past = t_abs_margined[:Tpast]
                                        t_future = t_abs_margined[Tpast:T]

                                        figf = go.Figure()
                                        # Past true segment
                                        if Tpast > 0:
                                            figf.add_trace(
                                                go.Scatter(
                                                    x=t_past,
                                                    y=y_concat_c[:Tpast],
                                                    name="Y true (past)",
                                                    mode="lines",
                                                    line=dict(color="#1f77b4"),
                                                )
                                            )
                                        # True future segment
                                        figf.add_trace(
                                            go.Scatter(
                                                x=t_future,
                                                y=y_ft_c,
                                                name="Y true (future)",
                                                mode="lines",
                                                line=dict(color="#2ca02c"),
                                            )
                                        )
                                        # Predicted future
                                        figf.add_trace(
                                            go.Scatter(
                                                x=t_future,
                                                y=y_fp_c,
                                                name="Y forecast",
                                                mode="lines",
                                                line=dict(color="#d62728"),
                                            )
                                        )

                                        r_fore_ch = np.nan
                                        if (
                                            r_fore_list is not None
                                            and isinstance(r_fore_list, (list, tuple))
                                            and len(r_fore_list) > c
                                        ):
                                            r_fore_ch = r_fore_list[c]

                                        figf.update_layout(
                                            title=f"Forecast (m={m}) — {selected_name} (r_future={r_fore_ch:.3f})",
                                            xaxis_title="Time (s)",
                                            yaxis_title="Amplitude (µV)",
                                        )
                                        st.plotly_chart(figf, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not render forecast plot: {e}")

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
                                    event_start = (
                                        t_offset + float(cm)
                                        if cm is not None
                                        else t_x[0]
                                    )
                                    event_end = (
                                        t_offset
                                        + float(dur)
                                        - (float(cm) if cm is not None else 0.0)
                                    )
                                    figx.add_vrect(
                                        x0=event_start,
                                        x1=event_end,
                                        fillcolor="rgba(0, 100, 0, 0.1)",
                                        layer="below",
                                        line_width=0,
                                    )
                                    figx.add_vline(
                                        x=event_start,
                                        line_dash="dash",
                                        line_color="green",
                                    )
                                    figx.add_vline(
                                        x=event_end,
                                        line_dash="dash",
                                        line_color="red",
                                    )
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
                                    event_start = (
                                        t_offset + float(cm)
                                        if cm is not None
                                        else t_z[0]
                                    )
                                    event_end = (
                                        t_offset
                                        + float(dur)
                                        - (float(cm) if cm is not None else 0.0)
                                    )
                                    figz.add_vrect(
                                        x0=event_start,
                                        x1=event_end,
                                        fillcolor="rgba(0, 100, 0, 0.1)",
                                        layer="below",
                                        line_width=0,
                                    )
                                    figz.add_vline(
                                        x=event_start,
                                        line_dash="dash",
                                        line_color="green",
                                    )
                                    figz.add_vline(
                                        x=event_end,
                                        line_dash="dash",
                                        line_color="red",
                                    )
                                figz.update_layout(
                                    title=f"Aux predictions Z_p — Trial {trial_idx}",
                                    xaxis_title="Time (s)",
                                    yaxis_title="Value",
                                    xaxis_range=[t_z[0], t_z[-1]],
                                    yaxis_range=[z_min, z_max],
                                )
                                st.plotly_chart(figz, use_container_width=True)
