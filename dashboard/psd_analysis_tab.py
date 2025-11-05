import streamlit as st
import polars as pl
import numpy as np

from utils.plots import (
    plot_psd_heatmap,
    plot_average_psd,
)
from utils.data_loader import natural_sort_key

def psd_analysis_tab(block_data, lfp_channels, ecog_channels):
    st.header("Power Spectral Density (PSD) Analysis")

    stim_col = "stim"
    trials_in_block = sorted(block_data["trial"].unique().to_list())
    selected_trial_psd = st.selectbox(
        "Select a Trial for PSD Heatmap",
        options=trials_in_block,
    )

    trial_data = block_data.filter(pl.col("trial") == selected_trial_psd)

    st.subheader("PSD Heatmap for a Specific Trial")
    if trial_data is not None and not trial_data.is_empty():
        selected_block = st.session_state.get("block")
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
