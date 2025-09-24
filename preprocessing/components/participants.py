import polars as pl
from pathlib import Path

from utils.polars import (
    read_tsv,
    add_modality_path,
    split_file_path,
    remove_rows_with,
    explode_files,
    keep_rows_with,
    band_pass_resample,
)
from .events import construct_events_table
from .motion import construct_motion_table

from utils.config import Config

iEEG_SCHEMA = pl.Struct(
    [
        pl.Field("LFP_1", pl.List(pl.Float64)),
        pl.Field("LFP_2", pl.List(pl.Float64)),
        pl.Field("LFP_3", pl.List(pl.Float64)),
        pl.Field("LFP_4", pl.List(pl.Float64)),
        pl.Field("LFP_5", pl.List(pl.Float64)),
        pl.Field("LFP_6", pl.List(pl.Float64)),
        pl.Field("LFP_7", pl.List(pl.Float64)),
        pl.Field("LFP_8", pl.List(pl.Float64)),
        pl.Field("LFP_9", pl.List(pl.Float64)),
        pl.Field("LFP_10", pl.List(pl.Float64)),
        pl.Field("LFP_11", pl.List(pl.Float64)),
        pl.Field("LFP_12", pl.List(pl.Float64)),
        pl.Field("LFP_13", pl.List(pl.Float64)),
        pl.Field("LFP_14", pl.List(pl.Float64)),
        pl.Field("LFP_15", pl.List(pl.Float64)),
        pl.Field("LFP_16", pl.List(pl.Float64)),
        pl.Field("ECOG_1", pl.List(pl.Float64)),
        pl.Field("ECOG_2", pl.List(pl.Float64)),
        pl.Field("ECOG_3", pl.List(pl.Float64)),
        pl.Field("ECOG_4", pl.List(pl.Float64)),
        pl.Field("EOG_1", pl.List(pl.Float64)),
        pl.Field("EOG_2", pl.List(pl.Float64)),
        pl.Field("EOG_3", pl.List(pl.Float64)),
        pl.Field("EOG_4", pl.List(pl.Float64)),
        pl.Field("sfreq", pl.Float64),
    ]
)


def construct_participants_table(config: Config):
    data_path = Path(config.data_directory)
    participants = read_tsv(data_path / config.participants_table_name)
    participants = participants.filter(
        pl.col("participant_id").str.starts_with("sub-PD")
    )
    participants = participants.drop("age", "sex", "hand", "weight", "height")

    participants = participants.with_columns(
        pl.concat_str(
            pl.lit(str(data_path)), pl.col("participant_id"), separator="/"
        ).alias("participant_path")
    )
    participants = explode_files(participants, "participant_path", "session_path")

    ieeg_participants = _add_ieeg_data(participants, config)
    motion_participants = construct_motion_table(participants)

    participants = ieeg_participants.join(
        motion_participants, on=["participant_id", "session", "run"], how="left"
    )

    participants = participants.drop(
        "participant_path",
        "session_path",
        "participant_path_right",
        "session_path_right",
        "ieeg_path",
        "ieeg_file",
        "ieeg_headers_file",
        "motion_path",
        "motion_file",
        "type",
        "data_format",
    )

    participants = _chunk_recordings(participants, config.ieeg_process.chunk_margin)
    return participants


def _add_ieeg_data(participants: pl.DataFrame, config: Config) -> pl.DataFrame:
    participants = add_modality_path(participants, "ieeg")

    participants = explode_files(participants, "ieeg_path", "ieeg_file")

    participants = split_file_path(
        participants,
        "ieeg",
        [("session", 1, pl.UInt64), ("task", 2, pl.String), ("run", 3, pl.UInt64)],
    )
    participants = keep_rows_with(participants, task="copydraw").drop("task")
    participants = remove_rows_with(participants, type="channels", data_format="tsv")

    events = construct_events_table(participants)

    participants = participants.join(
        events, on=["participant_id", "session", "run"], how="left"
    )

    participants = remove_rows_with(participants, type="events", data_format="tsv")
    headers = keep_rows_with(participants, type="ieeg", data_format="vhdr").select(
        "participant_id",
        "session",
        "run",
        pl.col("ieeg_file").alias("ieeg_headers_file"),
    )

    participants = participants.join(
        headers, on=["participant_id", "session", "run"], how="left"
    )

    participants = remove_rows_with(participants, type="ieeg", data_format="vhdr")
    participants = remove_rows_with(participants, type="ieeg", data_format="vmkr")

    participants = participants.drop(
        "type", "data_format", "channels_info_right", strict=False
    )

    participants = band_pass_resample(participants, config, iEEG_SCHEMA)

    return participants


def _chunk_recordings(participants: pl.DataFrame, chunk_margin: float) -> pl.DataFrame:

    participants_ = participants.with_columns(
        pl.col("onsets").list.get(pl.col("chunk") - 1).alias("onset"),
        pl.col("durations").list.get(pl.col("chunk") - 1).alias("duration"),
    ).drop("onsets", "durations")

    participants_ = participants_.with_columns(
        (pl.col("onset") - pl.lit(chunk_margin)).alias("onset"),
        (pl.col("duration") + pl.lit(chunk_margin)).alias("duration"),
    )

    participants_ = participants_.with_columns(
        (pl.col("onset") * pl.col("sfreq")).cast(pl.Int64).alias("start_ts"),
        (pl.col("duration") * pl.col("sfreq")).cast(pl.Int64).alias("chunk_length_ts"),
    )

    for ieeg_field in iEEG_SCHEMA.fields:
        if ieeg_field.name == "sfreq":
            continue
        participants_ = participants_.with_columns(
            pl.col(ieeg_field.name).list.slice(
                pl.col("start_ts"), pl.col("chunk_length_ts")
            )
        )

    participants_ = participants_.with_columns(
        pl.lit(chunk_margin).alias("chunk_margin")
    )

    return participants_
