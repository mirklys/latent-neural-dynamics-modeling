import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def _create_base_figure(title: str, x_axis_title: str, y_axis_title: str) -> go.Figure:
    """Creates a base Plotly figure with a standardized 'academic' layout."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, family="Times New Roman")),
        xaxis_title=dict(
            text=x_axis_title, font=dict(size=14, family="Times New Roman")
        ),
        yaxis_title=dict(
            text=y_axis_title, font=dict(size=14, family="Times New Roman")
        ),
        template="plotly_white",
        font=dict(family="Times New Roman", size=12, color="black"),
        legend=dict(font=dict(size=12, family="Times New Roman")),
        showlegend=True,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


def plot_trial_channel(trial_df: pl.DataFrame, channel: str) -> go.Figure:
    """Plots a single iEEG channel for a given trial with improved styling."""
    if trial_df.is_empty():
        return go.Figure().update_layout(title_text=f"No data for {channel}")

    participant_id = trial_df["participant_id"][0]
    trial = trial_df["trial"][0]
    dbs_state = "ON" if trial_df["dbs_stim"][0] else "OFF"

    title = f"{channel.replace('_', ' ')} Signal (P{participant_id}, Trial {trial}, DBS {dbs_state})"
    fig = _create_base_figure(title, "Time (s)", "Amplitude (µV)")

    time_start = trial_df["time"].min()
    trial_df = trial_df.with_columns(
        (pl.col("time") - time_start).alias("relative_time")
    )

    fig.add_trace(
        go.Scatter(
            x=trial_df["relative_time"],
            y=trial_df[channel],
            mode="lines",
            name=channel,
            line=dict(color="black", width=1),
        )
    )

    # Add annotations for event start/end
    chunk_margin = trial_df["chunk_margin"][0]
    original_duration = trial_df["margined_duration"][0] - 2 * chunk_margin
    event_start = chunk_margin
    event_end = event_start + original_duration

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
        annotation_text="Event Start",
    )
    fig.add_vline(
        x=event_end, line_dash="dash", line_color="red", annotation_text="Event End"
    )

    fig.update_layout(showlegend=False)
    return fig


def plot_trial_coordinates(
    trial_df: pl.DataFrame, time: str, plot_over_time: bool = False
) -> go.Figure:
    """Plots trial coordinates with improved styling."""
    if trial_df.is_empty() or trial_df["x"].is_null().all():
        return go.Figure().update_layout(title_text="No coordinate data available")

    p_id = trial_df["participant_id"][0]
    session = trial_df["session"][0]
    trial = trial_df["trial"][0]

    if plot_over_time:
        title = f"Coordinates vs. Time (P{p_id}, S{session}, T{trial})"
        fig = _create_base_figure(title, "Time (s)", "Position")
        fig.add_trace(
            go.Scatter(x=trial_df[time], y=trial_df["x"], mode="lines", name="X-coord")
        )
        fig.add_trace(
            go.Scatter(x=trial_df[time], y=trial_df["y"], mode="lines", name="Y-coord")
        )
    else:
        title = f"2D Trajectory (P{p_id}, S{session}, T{trial})"
        fig = _create_base_figure(title, "X Coordinate", "Y Coordinate")
        fig.add_trace(
            go.Scatter(
                x=trial_df["x"], y=trial_df["y"], mode="lines+markers", name="Path"
            )
        )
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        fig.add_trace(
            go.Scatter(
                x=trial_df.head(1)["x"],
                y=trial_df.head(1)["y"],
                mode="markers",
                marker=dict(color="green", size=10),
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trial_df.tail(1)["x"],
                y=trial_df.tail(1)["y"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="End",
            )
        )

    return fig


def plot_tracing_speed(trial_df: pl.DataFrame, time: str) -> go.Figure:
    """Plots tracing speed with improved styling."""
    if trial_df.is_empty() or trial_df["tracing_speed"].is_null().all():
        return go.Figure().update_layout(title_text="No speed data available")

    p_id = trial_df["participant_id"][0]
    session = trial_df["session"][0]
    trial = trial_df["trial"][0]

    title = f"Tracing Speed (P{p_id}, S{session}, T{trial})"
    fig = _create_base_figure(title, "Time (s)", "Speed (pixels/s)")
    fig.add_trace(
        go.Scatter(
            x=trial_df[time], y=trial_df["tracing_speed"], mode="lines", name="Speed"
        )
    )
    fig.update_layout(showlegend=False)

    return fig


def plot_psd_heatmap(freqs: np.ndarray, psds: np.ndarray, title: str) -> go.Figure:
    """Plots a PSD heatmap with improved styling."""
    if freqs.size == 0 or psds.size == 0:
        return go.Figure().update_layout(title_text="No PSD data for heatmap")

    fig = _create_base_figure(title, "Time Epoch", "Frequency (Hz)")
    # Ensure psds is a float array for log calculation

    psds_arr = np.asarray([np.array(psd, dtype=float) for psd in psds])
    log_psds = 10 * np.log10(psds_arr.T) + 120

    fig.add_trace(
        go.Heatmap(
            z=log_psds,
            x=np.arange(psds_arr.shape[0]),
            y=freqs,
            colorscale="Viridis",
            colorbar=dict(title="Power/Freq (dB/Hz)"),
        )
    )
    return fig


def plot_average_psd(
    freqs: np.ndarray,
    psd_data: dict,  # {channel: {'on': psds_on, 'off': psds_off}}
    title: str,
) -> go.Figure:
    """
    Plots the average PSD for multiple channels, comparing DBS ON and OFF states.
    """
    fig = _create_base_figure(title, "Frequency (Hz)", "Power/Frequency (dB/Hz)")
    fig.update_layout(legend_title_text="Channel (DBS State)")

    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for channel, data in psd_data.items():
        color_on = colors[color_idx % len(colors)]
        color_off = colors[(color_idx + 1) % len(colors)]

        # Plot DBS ON
        if "on" in data and data["on"].size > 0:
            mean_psd_on = 10 * np.log10(np.mean(data["on"], axis=0)) + 120
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=mean_psd_on,
                    mode="lines",
                    name=f"{channel} (ON)",
                    line=dict(color=color_on),
                )
            )

        # Plot DBS OFF
        if "off" in data and data["off"].size > 0:
            mean_psd_off = 10 * np.log10(np.mean(data["off"], axis=0)) + 120
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=mean_psd_off,
                    mode="lines",
                    name=f"{channel} (OFF)",
                    line=dict(color=color_off, dash="dash"),
                )
            )

        color_idx += 2

    return fig
