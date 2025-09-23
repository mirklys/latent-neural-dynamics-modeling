import polars as pl
from utils.polars import (
    read_tsv_to_struct,
    keep_rows_with,
    add_modality_path,
    explode_files,
    split_file_path,
    keep_rows_with,
)

MOTION_SCHEMA = pl.List(
    pl.Struct(
        [
            pl.Field("x", pl.Float64),
            pl.Field("y", pl.Float64),
        ]
    )
)


def construct_motion_table(participants: pl.DataFrame) -> pl.DataFrame:
    motion_ = participants.unique(["participant_id", "session_path"])

    motion_ = add_modality_path(motion_, "motion")
    motion_ = explode_files(motion_, "motion_path", "motion_file")
    motion_ = split_file_path(
        motion_,
        "motion",
        [("session", -6, pl.UInt64), ("run", -4, pl.UInt64), ("chunk", -3, pl.UInt64)],
    )
    motion_ = keep_rows_with(motion_, type="motion", data_format="tsv")
    motion_ = motion_.with_columns(
        pl.col("motion_file")
        .map_elements(read_tsv_to_struct, return_dtype=MOTION_SCHEMA)
        .alias("motion_coordinates")
    )

    motion_ = (
        motion_.sort(by=["participant_id", "session", "run", "chunk"])
        .group_by(["participant_id", "session", "run", "chunk"], maintain_order=True)
        .agg(pl.col("motion_coordinates"))
    )

    return motion_
