import polars as pl
from pathlib import Path
from utils.file_handling import list_files
from .logger import get_logger

def read_tsv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        null_values="n/a",
    )

    return df


def read_tsv_to_struct(path: Path) -> pl.Series:
    df = read_tsv(path)

    return df.to_struct()


def add_modality_path(participants: pl.DataFrame, modality: str) -> pl.DataFrame:

    participants_ = participants.with_columns(
        pl.concat_str(
            [
                pl.col("session_path"),
                pl.lit(modality),
            ],
            separator="/",
        ).alias(f"{modality}_path"),
    )

    return participants_


def explode_files(
    participants: pl.DataFrame, folder: str, out_file_col: str
) -> pl.DataFrame:

    participants_ = participants.with_columns(
        pl.col(folder).map_elements(
            lambda folder: list_files(Path(folder)), return_dtype=pl.List(pl.String)
        ).alias(out_file_col)
    ).explode(pl.col(out_file_col))

    return participants_


def split_file_path(
    participants: pl.DataFrame, modality: str, positions: dict[str, int]
) -> pl.DataFrame:
    logger = get_logger()
    participants_ = participants.with_columns(
        pl.col(f"{modality}_file").str.split(by="/")
        .list.get(-1)
        .str.split("_")
        .alias("split_file")
    )

    participants_ = participants_.with_columns(
        pl.col("split_file").list.get(-1).str.split(".").list.get(0).alias("type"),
        pl.col("split_file")
        .list.get(-1)
        .str.split(".")
        .list.get(-1)
        .alias("data_format"),
    )

    for part, indx in positions.items():
        participants_ = participants_.with_columns(
            pl.col("split_file").list.get(indx).alias(part),
        )

    return participants_.drop("split_file")


def __create_and_polars_filter(**kwargs) -> bool:
    first_key, first_value = next(iter(kwargs.items()))
    filter_ = pl.col(first_key) == first_value

    for key, value in list(kwargs.items())[1:]:
        filter_ = filter_ & (pl.col(key) == value)

    return filter_


def remove_rows_with(table: pl.DataFrame, **kwargs) -> pl.DataFrame:
    filter_ = __create_and_polars_filter(**kwargs)
    table_ = table.filter(~filter_)

    return table_


def keep_rows_with(table: pl.DataFrame, **kwargs) -> pl.DataFrame:
    filter_ = __create_and_polars_filter(**kwargs)
    table_ = table.filter(filter_)

    return table_

def dict_to_struct(data: dict) -> pl.Series:
    return pl.DataFrame(data).to_struct()

def read_motion_data(row: dict) -> pl.Series:
    path_ = Path(row["motion_path"]) / row["motion_file"]
    df = pl.read_csv(
        path_,
        separator="\t",
        null_values="n/a",
        has_header=False,
        new_columns=["x", "y"]
    )
    return df.to_struct()
