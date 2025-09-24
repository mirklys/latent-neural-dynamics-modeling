import polars as pl
from utils.polars import (
    read_tsv_to_dict,
    keep_rows_with,
    add_modality_path,
    explode_files,
    split_file_path,
    keep_rows_with,
)

from utils.file_handling import read_json


def _get_motion_coordinates(motion: pl.DataFrame) -> pl.DataFrame:
    motion_coords = keep_rows_with(motion, type="motion", data_format="tsv")
    motion_coords = motion_coords.with_columns(
        pl.col("motion_file")
        .map_elements(read_tsv_to_dict, return_dtype=pl.Object)
        .alias("motion_coordinates")
    )
    motion_coords = motion_coords.with_columns(
        pl.col("motion_coordinates")
        .map_elements(lambda md: md["x"], pl.List(pl.Int64))
        .alias("x"),
        pl.col("motion_coordinates")
        .map_elements(lambda md: md["y"], pl.List(pl.Int64))
        .alias("y"),
    ).drop("motion_coordinates")

    motion_coords = motion_coords.sort(by=["participant_id", "session", "run", "chunk"])

    return motion_coords


def _get_motion_dbs_cond(motion: pl.DataFrame) -> pl.DataFrame:
    motion_dbs_cond = (
        keep_rows_with(motion, type="motion", data_format="json")
        .filter(pl.col("chunk").is_null())
        .drop("chunk", "type", "data_format")
    )

    motion_dbs_cond = motion_dbs_cond.with_columns(
        pl.col("motion_file")
        .map_elements(lambda mf: read_json(mf)["dbs_stim"], return_dtype=pl.String)
        .alias("dbs_stim")
    ).drop("motion_file")

    return motion_dbs_cond


def construct_motion_table(participants: pl.DataFrame) -> pl.DataFrame:
    motion_ = participants.unique(["participant_id", "session_path"])

    motion_ = add_modality_path(motion_, "motion")
    motion_ = explode_files(motion_, "motion_path", "motion_file")
    motion_ = split_file_path(
        motion_,
        "motion",
        [("session", 1, pl.UInt64), ("run", 3, pl.UInt64), ("chunk", 4, pl.UInt64)],
    )

    motion_coords = _get_motion_coordinates(motion_)
    motion_dbs_cond = _get_motion_dbs_cond(
        motion_.select(
            "participant_id",
            "session",
            "run",
            "chunk",
            "motion_file",
            "type",
            "data_format",
        )
    )

    motion_ = motion_coords.join(
        motion_dbs_cond, on=["participant_id", "session", "run"], how="left"
    )

    return motion_
