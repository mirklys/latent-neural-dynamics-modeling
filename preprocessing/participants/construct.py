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
    read_motion_data,
)
from utils.ieeg import band_pass_resample
from ..intermediate_tables.events import construct_events_table


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
    motion_participants = _add_motion_data(participants, config)

    participants = ieeg_participants.join(
        motion_participants, on=["participant_id", "session", "run"], how="left"
    )

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
        pl.col("ieeg_headers_file").map_elements(
            lambda hd: dict_to_struct(
                band_pass_resample(
                    hd,
                    config.ieeg_process.resampled_freq,
                    config.ieeg_process.low_freq,
                    config.ieeg_process.high_freq,
                    config.ieeg_process.notch_freqs,
                )
            ),
            return_dtype=pl.List(pl.Struct),
        ).alias("ieeg_raw")
    )
    return participants

def _add_motion_data(participants: pl.DataFrame, config) -> pl.DataFrame:
    motion = add_modality_path(participants, "motion")
    motion = explode_files(motion, "motion_path", "motion_file")
    motion = split_file_path(
        motion, "motion", {"run": -4, "chunk": -3, "session": -6}
    )
    motion = remove_rows_with(motion, type="channels", data_format="tsv")
    motion = remove_rows_with(motion, type="motion", data_format="json")

    motion_schema = pl.List(
        pl.Struct(
            [
                pl.Field("x", pl.List(pl.Float64)),
                pl.Field("y", pl.List(pl.Float64)),
            ]
        )
    )
    motion = motion.with_columns(
        pl.struct(["motion_path", "motion_file"])
        .map_elements(read_motion_data, return_dtype=motion_schema)
        .alias("motion_coordinates"),
    )

    motion = motion.select("participant_id", "session", "run", "motion_coordinates")

    return motion
