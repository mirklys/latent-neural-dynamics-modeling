import streamlit as st
import polars as pl

from utils.plots import (
    plot_trial_channel,
    plot_trial_coordinates,
    plot_tracing_speed,
)
from utils.data_loader import natural_sort_key


def time_series_tab(block_data):
    st.header("Time-Series Analysis")
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

    if trial_data is not None:
        stim_on = trial_data["stim"][0]
        st.subheader(
            f"""Participant {st.session_state.get('participant_id')} Session {st.session_state.get('session')}
            Block{selected_block} Trial{trial_data['trial'][0]} Stim {stim_on}"""
        )

        use_absolute_time = st.toggle("Use Absolute Time", value=False)

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
                    plot_trial_channel(df_exploded, selected_lfp, use_absolute_time),
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
                    plot_trial_channel(df_exploded, selected_ecog, use_absolute_time),
                    use_container_width=True,
                )
        with col2:
            st.markdown("#### Coordinates and Speed")

            # Check which motion columns exist and have data
            motion_cols = []
            if "motion_time" in trial_data.columns:
                motion_cols.append("motion_time")
            if "x" in trial_data.columns:
                motion_cols.append("x")
            if "y" in trial_data.columns:
                motion_cols.append("y")
            if "tracing_speed" in trial_data.columns:
                motion_cols.append("tracing_speed")

            if motion_cols:
                try:
                    # Try to explode all columns together
                    coords_data = trial_data.select(
                        *motion_cols,
                        "participant_id",
                        "session",
                        "block",
                        "trial",
                    ).explode(*motion_cols)
                except (pl.exceptions.ShapeError, Exception):
                    # If columns have different lengths, explode separately and align
                    st.warning(
                        "Motion data columns have mismatched lengths. Aligning to shortest column."
                    )

                    # Find the minimum length among all motion columns
                    lengths = {}
                    for col in motion_cols:
                        col_data = trial_data[col][0]
                        if col_data is not None:
                            lengths[col] = (
                                len(col_data)
                                if isinstance(col_data, (list, pl.Series))
                                else 1
                            )

                    if lengths:
                        min_length = min(lengths.values())

                        # Truncate all columns to minimum length
                        truncated_data = trial_data.select(
                            "participant_id", "session", "block", "trial"
                        )

                        for col in motion_cols:
                            col_data = trial_data[col][0]
                            if col_data is not None:
                                truncated = col_data[:min_length]
                                truncated_data = truncated_data.with_columns(
                                    pl.Series([truncated]).alias(col)
                                )

                        coords_data = truncated_data.explode(*motion_cols)
                    else:
                        coords_data = None

                if coords_data is not None:
                    if "x" in coords_data.columns and "y" in coords_data.columns:
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

                    if "tracing_speed" in coords_data.columns:
                        st.plotly_chart(
                            plot_tracing_speed(coords_data, time="motion_time"),
                            use_container_width=True,
                        )
                else:
                    st.info("Motion data could not be loaded for this trial.")
            else:
                st.info("No motion data available for this trial.")
    else:
        st.info("Select a block and trial to view time-series data.")
