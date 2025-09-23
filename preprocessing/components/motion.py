import polars as pl
from utils.polars import (
    read_tsv_to_struct,
    read_tsv_to_dict,
    keep_rows_with,
    add_modality_path,
    explode_files,
    split_file_path,
    keep_rows_with,
)

MOTION_SCHEMA = pl.List(
    pl.Struct(
        [
            pl.Field("x", pl.Int64),
            pl.Field("y", pl.Int64),
        ]
    )
)


def _get_motion_coordinates(motion: pl.DataFrame) -> pl.DataFrame:
    motion_coords = keep_rows_with(motion, type="motion", data_format="tsv")
    motion_coords = motion_coords.with_columns(
        pl.col("motion_file")
        .map_elements(read_tsv_to_struct, return_dtype=MOTION_SCHEMA)
        .alias("motion_coordinates")
    )

    motion_coords = (
        motion_coords.sort(by=["participant_id", "session", "run", "chunk"])
        .group_by(["participant_id", "session", "run", "chunk"], maintain_order=True)
        .agg(pl.col("motion_coordinates"))
    )

    return motion_coords

def _get_motion_dbs_cond(motion: pl.DataFrame) -> pl.DataFrame:
    motion_dbs_cond = keep_rows_with(motion, type="motion", data_format="json").filter(
        pl.col("chunk").is_null()
    )
    motion_dbs_cond = motion_dbs_cond.with_columns(
        pl.col("motion_file")
        .map_elements(lambda mf: read_tsv_to_dict(mf)["dbs_stim"], return_dtype=pl.String)
        .alias("dbs_stim")
    )
    return NotImplemented

def construct_motion_table(participants: pl.DataFrame) -> pl.DataFrame:
    motion_ = participants.unique(["participant_id", "session_path"])

    motion_ = add_modality_path(motion_, "motion")
    motion_ = explode_files(motion_, "motion_path", "motion_file")
    motion_ = split_file_path(
        motion_,
        "motion",
        [("session", 1, pl.UInt64), ("run", 3, pl.UInt64), ("chunk", 4, pl.UInt64)],
    )

    motion_coords =_get_motion_coordinates(motion_)
    motion_dbs_cond = _get_motion_dbs_cond(motion_)

    motion_ = motion_coords.join(motion_dbs_cond, on=["participant_id", "session", "run"], how="left")

    return motion_
