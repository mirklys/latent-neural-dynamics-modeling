import polars as pl
from pathlib import Path

from utils.polars import (
    read_tsv,
    add_modality_path,
    split_file_path,
    remove_rows_with,
    explode_files,
    keep_rows_with,
    dict_to_struct,
)
from utils.ieeg import band_pass_resample
from .events import construct_events_table
from .motion import construct_motion_table

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
    ]
)


def construct_participants_table(config):
    data_path = Path(config.data_directory)
    participants = read_tsv(data_path / config.participants_table_name)

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
    # TODO: transpose the recordings
    # TODO: get only records based on the marker of 9 seconds
    # TODO: get tracing coordinates between the markers, extrapolate the time around 9 seconds, and calculate the speed in each trial (between the markers)
    # TODO: create the trial partition column for between the markers and start and end time in the recording itself
    return participants


def _add_ieeg_data(participants: pl.DataFrame, config) -> pl.DataFrame:
    participants = add_modality_path(participants, "ieeg")

    participants = explode_files(participants, "ieeg_path", "ieeg_file")

    participants = split_file_path(participants, "ieeg", {"run": -2, "session": -4})
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

    participants = participants.with_columns(
        pl.col("ieeg_headers_file")
        .map_elements(
            lambda hd: dict_to_struct(
                band_pass_resample(
                    hd,
                    config.ieeg_process.resampled_freq,
                    config.ieeg_process.low_freq,
                    config.ieeg_process.high_freq,
                    config.ieeg_process.notch_freqs,
                )
            ),
            return_dtype=pl.List(iEEG_SCHEMA),
        )
        .alias("ieeg_raw")
    )
    return participants
