import polars as pl
from utils.polars import read_tsv_to_struct, keep_rows_with

EVENTS_SCHEMA = pl.List(
    pl.Struct(
        [
            pl.Field("onset", pl.Float64),
            pl.Field("duration", pl.Float64),
            pl.Field("trial_type", pl.Int64),
            pl.Field("value", pl.Int64),
            pl.Field("sample", pl.Int64),
        ]
    )
)


def construct_events_table(participants: pl.DataFrame) -> pl.DataFrame:
    events_ = keep_rows_with(participants, type="events", data_format="tsv")

    events_ = events_.select(
        "participant_id",
        "session",
        "run",
        pl.col("ieeg_file")
        .map_elements(read_tsv_to_struct, return_dtype=EVENTS_SCHEMA)
        .alias("events"),
    )

    events_ = events_.explode("events").filter(
        pl.col("events").struct.field("value") != 25
    )

    events_ = (
        events_.sort(
            by=[
                pl.col("participant_id"),
                pl.col("session"),
                pl.col("run"),
                pl.col("events").struct.field("onset"),
            ]
        )
        .with_columns(pl.col("events").struct.unnest())
        .group_by(["participant_id", "session", "run"], maintain_order=True)
        .agg(
            pl.col("onset"),
            pl.col("duration"),
            pl.col("trial_type"),
            pl.col("value"),
            pl.col("sample"),
        )
    )

    return events_
