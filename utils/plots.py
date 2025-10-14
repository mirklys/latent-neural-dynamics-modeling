import polars as pl
import numpy as np
import plotly.graph_objects as go


def _create_base_figure(title: str, x_axis_title: str, y_axis_title: str) -> go.Figure:
    """Creates a base Plotly figure with a standardized layout."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        template="plotly_white",
        font=dict(family="Times New Roman", size=12, color="black"),
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig


def plot_trial_channel(
    trial_df: pl.DataFrame,
    channel: str,
):
    """
    Plots a single iEEG channel for a given trial.

    Args:
        trial_df: A DataFrame for a single trial with required columns exploded.
        channel: The name of the channel column to plot.
    """
    if trial_df.is_empty():
        print("Input trial DataFrame is empty. Cannot plot.")
        return

    try:
        participant_id = trial_df.select(pl.col("participant_id").first()).item()
        trial = trial_df.select(pl.col("trial").first()).item()
    except pl.exceptions.ColumnNotFoundError:
        print("Warning: Could not find participant_id/trial columns for plot title.")
        return

    time_end_absolute = trial_df.select(pl.col("time").max()).item()
    time_start = trial_df.select(pl.col("time").min()).item()
    trial_df = trial_df.with_columns(time_original=(pl.col("time") - time_start))

    chunk_margin = trial_df.select(pl.col("chunk_margin").first()).item()
    original_duration = (
        trial_df.select(pl.col("margined_duration").first()).item() - 2 * chunk_margin
    )
    dbs_stim_val = trial_df.select(pl.col("dbs_stim").first()).item()
    dbs_state = "ON" if dbs_stim_val == 1 else "OFF"

    event_start_relative = chunk_margin
    event_end_relative = event_start_relative + original_duration

    title_text = (
        f"{channel.replace('_', ' ')} Signal for Participant {participant_id}, "
        f"Trial {trial} (DBS {dbs_state})"
    )
    fig = _create_base_figure(title_text, "Relative Time (s)", "Signal Amplitude (µV)")
    fig.update_layout(showlegend=False)

    fig.add_trace(
        go.Scatter(
            x=trial_df["time_original"],
            y=trial_df[channel],
            mode="lines",
            name=channel,
            line=dict(color="black", width=1.5),
        )
    )

    fig.add_vline(
        x=event_start_relative,
        line_width=1.5,
        line_dash="dash",
        line_color="darkgreen",
        annotation_text="Event Start",
        annotation_position="top left",
    )
    fig.add_vline(
        x=event_end_relative,
        line_width=1.5,
        line_dash="dash",
        line_color="darkred",
        annotation_text="Event End",
        annotation_position="top right",
    )
    fig.add_vrect(
        x0=event_start_relative,
        x1=event_end_relative,
        fillcolor="rgba(0, 100, 0, 0.1)",
        layer="below",
        line_width=0,
    )

    fig.update_layout(
        xaxis2=dict(
            title="Absolute Time (s)",
            side="top",
            overlaying="x",
            range=[time_start, time_end_absolute],
            visible=True,
            showticklabels=True,
        ),
    )

    return fig


def plot_trial_coordinates(
    trial_df: pl.DataFrame,
    x: str = "x",
    y: str = "y",
    time: str = None,
    plot_over_time: bool = False,
):
    """
    Plots the x, y coordinates for a specific trial in two ways:
    1. A 2D plot of y vs. x.
    2. A plot of x and y coordinates over time.

    Args:
        trial_df: A DataFrame for a single trial, with 'x', 'y', and 'time_original' columns exploded.
        plot_over_time: If True, plots coordinates over time. Otherwise, plots 2D trajectory.
    """
    trial_df = trial_df.sort(by=time) if time else trial_df
    if trial_df.is_empty():
        print("Input trial DataFrame is empty. Cannot plot.")
        return

    # Extract metadata for title from the DataFrame
    try:
        participant_id = trial_df.select(pl.col("participant_id").first()).item()
        session = trial_df.select(pl.col("session").first()).item()
        block = trial_df.select(pl.col("block").first()).item()
        trial = trial_df.select(pl.col("trial").first()).item()
    except pl.exceptions.ColumnNotFoundError:
        print(
            "Warning: Could not find participant/session/block/trial columns for plot title."
        )
        participant_id, session, block, trial = "N/A", "N/A", "N/A", "N/A"

    if plot_over_time:
        title = f"Coordinates over Time for P{participant_id}, S{session}, B{block}, T{trial}"
        fig = _create_base_figure(title, "Time (s)", "Coordinate Value")
        fig.add_trace(
            go.Scatter(x=trial_df[time], y=trial_df[x], mode="lines", name="X")
        )
        fig.add_trace(
            go.Scatter(x=trial_df[time], y=trial_df[y], mode="lines", name="Y")
        )
    else:
        title = f"2D Trajectory for P{participant_id}, S{session}, B{block}, T{trial}"
        fig = _create_base_figure(title, "X Coordinate", "Y Coordinate")
        fig.add_trace(
            go.Scatter(x=trial_df[x], y=trial_df[y], mode="markers", name="Trajectory")
        )
        fig.add_trace(
            go.Scatter(
                x=trial_df.head(1)[x],
                y=trial_df.head(1)[y],
                mode="markers",
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trial_df.tail(1)[x], y=trial_df.tail(1)[y], mode="markers", name="End"
            )
        )

    return fig


def plot_tracing_speed(
    trial_df: pl.DataFrame,
    tracing_speed: str = "tracing_speed",
    time: str = "time_original",
):
    """
    Plots the tracing speed for a single trial.

    Args:
        trial_df: A DataFrame for a single trial with 'time_original' and 'tracing_speed' columns exploded.
    """
    if trial_df.is_empty():
        print("Input trial DataFrame is empty. Cannot plot.")
        return

    # Extract metadata for title from the DataFrame
    try:
        participant_id = trial_df.select(pl.col("participant_id").first()).item()
        session = trial_df.select(pl.col("session").first()).item()
        block = trial_df.select(pl.col("block").first()).item()
        trial = trial_df.select(pl.col("trial").first()).item()
    except pl.exceptions.ColumnNotFoundError:
        print(
            "Warning: Could not find participant/session/block/trial columns for plot title."
        )
        participant_id, session, block, trial = "N/A", "N/A", "N/A", "N/A"

    title = f"Tracing Speed for P{participant_id}, S{session}, B{block}, T{trial}"
    fig = _create_base_figure(title, "Time (s)", "Tracing Speed (pixels/s)")

    fig.add_trace(
        go.Scatter(
            x=trial_df[time],
            y=trial_df[tracing_speed],
            mode="lines",
            name="Speed",
        )
    )

    return fig


def plot_psd_heatmap(
    freqs: np.ndarray, psds: np.ndarray, title: str = "PSD Heatmap per Trial"
):
    """
    Plots a heatmap of the Power Spectral Density (PSD) for a single trial.

    The heatmap shows frequency on the y-axis, time (as epochs) on the x-axis,
    and power as the color intensity.

    Args:
        freqs: A 1D numpy array of frequency values.
        psds: A 2D numpy array of PSD values with shape (n_epochs, n_freqs).
        title: The title for the plot.
    """
    fig = _create_base_figure(
        title=title, x_axis_title="Epoch Number", y_axis_title="Frequency (Hz)"
    )

    # Transpose PSDs so that shape is (n_freqs, n_epochs) for the heatmap
    # Use log power for better color contrast
    log_psds = 10 * np.log10(psds.T) + 120

    fig.add_trace(
        go.Heatmap(
            z=log_psds,
            x=np.arange(psds.shape[0]),  # Epoch numbers
            y=freqs,
            colorscale="Viridis",
            colorbar=dict(title="Power/Frequency (dB/Hz)"),
        )
    )

    return fig


def plot_avg_psd(
    freqs: np.ndarray, psds: np.ndarray, title: str = "Average PSD"  # Corrected title
):
    fig = _create_base_figure(
        title=title,
        x_axis_title="Frequency (Hz)",
        y_axis_title="Power/Frequency (dB/Hz)",
    )

    mean_psd_linear = np.mean(psds, axis=0)  # Average across epochs (axis 0)

    mean_psd_db = 10 * np.log10(mean_psd_linear) + 120

    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=mean_psd_db,
        )
    )

    return fig
