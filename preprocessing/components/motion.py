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
    participants_ = participants.unique(["participant_id", "session_path"])
    participants_ = add_modality_path(participants_, "motion")
    from utils.logger import get_logger

    logger = get_logger()
    logger.info(participants_)
    participants_ = explode_files(participants_, "motion_path", "motion_file")
    participants_ = split_file_path(
        participants_, "motion", {"run": -4, "chunk": -3, "session": -6}
    )
    participants_ = keep_rows_with(participants_, type="motion", data_format="tsv")
    participants_ = participants_.with_columns(
        pl.col("motion_file")
        .map_elements(read_tsv_to_struct, return_dtype=MOTION_SCHEMA)
        .alias("motion_coordinates")
    )
    participants_ = (
        participants_.sort(
            by=[
                pl.col("participant_id"),
                pl.col("session"),
                pl.col("run"),
                pl.col("chunk"),
            ]
        )
        .group_by(["participant_id", "session", "run"])
        .agg(pl.col("motion_coordinates"))
    )

    logger.info(participants_)

    return participants_
