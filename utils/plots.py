import polars as pl
import plotly.graph_objects as go
from utils.polars import get_trial


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


def plot_trial_coordinates(
    participants: pl.DataFrame,
    participant_id: str,
    session: int,
    block: int,
    trial: int,
    plot_over_time: bool = False,
):
    """
    Plots the x, y coordinates for a specific trial in two ways:
    1. A 2D plot of y vs. x.
    2. A plot of x and y coordinates over time.
    """
    trial_df = get_trial(
        participants,
        participant_id=participant_id,
        session=session,
        block=block,
        trial=trial,
        columns=["x", "y", "time"],
        explode=["x", "y", "time"],
    )
    if trial_df.is_empty():
        print(
            f"No data found for participant {participant_id}, session {session}, block {block}, trial {trial}"
        )
        return
    if plot_over_time:
        title = f"Coordinates over Time for P{participant_id}, S{session}, B{block}, T{trial}"
        fig = _create_base_figure(title, "Time (s)", "Coordinate Value")
        fig.add_trace(
            go.Scatter(
                x=trial_df["time"],
                y=trial_df["x"],
                mode="lines",
                name="X coordinate",
                line=dict(color="red"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trial_df["time"],
                y=trial_df["y"],
                mode="lines",
                name="Y coordinate",
                line=dict(color="blue"),
            )
        )
    else:
        title = f"2D Trajectory for P{participant_id}, S{session}, B{block}, T{trial}"
        fig = _create_base_figure(title, "X Coordinate", "Y Coordinate")
        fig.add_trace(
            go.Scatter(
                x=trial_df["x"],
                y=trial_df["y"],
                mode="lines",
                name="Trajectory",
                line=dict(color="blue", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trial_df.head(1)["x"],
                y=trial_df.head(1)["y"],
                mode="markers",
                marker=dict(color="green", size=10, symbol="circle"),
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trial_df.tail(1)["x"],
                y=trial_df.tail(1)["y"],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name="End",
            )
        )
        fig.update_yaxes(autorange="reversed")
    fig.show()


def plot_trial_channel(
    participants: pl.DataFrame,
    channel: str,
    participant_id: str,
    session: int,
    block: int,
    trial: int,
):
    trial_df = get_trial(
        participants,
        participant_id=participant_id,
        session=session,
        block=block,
        trial=trial,
        columns=["time", channel, "chunk_margin", "duration", "dbs_stim"],
        explode=["time", channel],
    )

    if trial_df.is_empty():
        print(
            f"No data found for participant {participant_id}, session {session}, block {block}, trial {trial}"
        )
        return

    time_end_absolute = trial_df.select(pl.col("time").max()).item()
    time_start = trial_df.select(pl.col("time").min()).item()
    trial_df = trial_df.with_columns(time_relative=(pl.col("time") - time_start))

    chunk_margin = trial_df.select(pl.col("chunk_margin").first()).item()
    original_duration = (
        trial_df.select(pl.col("duration").first()).item() - 2 * chunk_margin
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
            x=trial_df["time_relative"],
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

    fig.show()
