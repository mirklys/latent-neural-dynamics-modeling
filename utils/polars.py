import polars as pl
from pathlib import Path
import shutil
import uuid
from utils.file_handling import list_files, load_mat_into_dict
from utils.config import Config
from tqdm import tqdm

from utils.ieeg import preprocess_ieeg

from scipy.io import savemat


def read_tsv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        null_values="n/a",
    ).unique()

    return df


def read_tsv_to_struct(path: Path) -> pl.Series:
    df = read_tsv(path)

    return df.to_struct()


def read_tsv_to_dict(path: Path) -> dict:
    df = read_tsv(path)

    return df.to_dict(as_series=True)


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
        pl.col(folder)
        .map_elements(
            lambda folder: list_files(Path(folder)), return_dtype=pl.List(pl.String)
        )
        .alias(out_file_col)
    ).explode(pl.col(out_file_col))

    return participants_


def split_file_path(
    participants: pl.DataFrame,
    modality: str,
    positions: list[tuple[str, int, pl.DataType]],
) -> pl.DataFrame:

    from utils.logger import get_logger

    logger = get_logger()

    participants_ = participants.with_columns(
        pl.col(f"{modality}_file")
        .str.split(by="/")
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

    for part, indx, dtype in positions:
        logger.info(f"Extracting '{part}' from filename at index {indx} as {dtype}")
        participants_ = participants_.with_columns(
            pl.col("split_file")
            .list.get(indx)
            .str.split("-")
            .list.get(-1)
            .cast(dtype, strict=False)
            .alias(part),
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


def load_parquet_as_dict(parquet_file: str) -> dict:

    df_ = pl.read_parquet(parquet_file)

    return df_.to_dict()


def dict_to_struct(data: dict) -> pl.Series:
    return pl.DataFrame(data).select(pl.all().implode()).to_struct()


def band_pass_resample(
    participants: pl.DataFrame, config: Config, iEEG_SCHEMA: pl.Struct
):
    """CODE IF PARTIAL PREPROCESSING IS NOT DONE
    save_dir = Path(config.ieeg_process.resampled_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    resampled_freq = config.ieeg_process.resampled_freq
    low_freq = config.ieeg_process.low_freq
    high_freq = config.ieeg_process.high_freq
    notch_freqs = config.ieeg_process.notch_freqs

    ieeg_headers_files = participants["ieeg_headers_file"].to_list()

    participants.write_parquet(
        save_dir / "participants_intermediate.parquet",
        partition_by=["participant_id", "session", "run"],
    )
    del participants

    for iegg_hf in ieeg_headers_files:
        base_file = iegg_hf.split("/")[-1].split(".")[0]
        ieeg_dict = preprocess_ieeg(
            iegg_hf, resampled_freq, low_freq, high_freq, notch_freqs
        )

        print(save_dir / f"{base_file}.mat")

        savemat(save_dir / f"{base_file}.mat", ieeg_dict)

    participants_ = pl.read_parquet(save_dir / "participants_intermediate.parquet")

    participants_ = participants_.with_columns(
        pl.col("ieeg_headers_file")
        .str.split("/")
        .list.get(-1)
        .str.split(".")
        .list.get(0)
        .alias("iegg_hf_base_file")
    )
    participants_ = participants_.with_columns(
        pl.concat_str(
            pl.lit(str(save_dir)),
            pl.concat_str(pl.col("iegg_hf_base_file"), pl.lit("mat"), separator="."),
            separator="/",
        ).alias("iegg_mat_file")
    ).drop("iegg_hf_base_file")

    participants_ = participants_.with_columns(
        pl.col("iegg_mat_file")
        .map_elements(load_mat_into_dict, return_dtype=pl.Object)
        .alias("ieeg_raw")
    )

    for ieeg_field in iEEG_SCHEMA.fields:
        participants_ = participants_.with_columns(
            pl.col("ieeg_raw")
            .map_elements(lambda ieeg_raw: ieeg_raw[ieeg_field.name][0], ieeg_field.dtype)
            .alias(ieeg_field.name)
        )

    return participants_.drop("ieeg_raw")
    """
    from utils.logger import get_logger

    logger = get_logger()

    resampled_dir = Path(config.data_directory) / "resampled"
    tmp_dir = Path(config.save_directory) / f"tmp_{uuid.uuid4()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting band-pass resampling process...")
    participants_ = participants.unique()
    participants_ = participants.with_columns(
        pl.col("ieeg_headers_file")
        .str.split("/")
        .list.get(-1)
        .str.split(".")
        .list.get(0)
        .alias("base_ieeg_file")
    )
    participants_ = participants_.with_columns(
        pl.concat_str(
            pl.lit(str(resampled_dir)),
            pl.concat_str(pl.col("base_ieeg_file"), pl.lit("parquet"), separator="."),
            separator="/",
        ).alias("ieeg_parquet")
    ).drop("base_ieeg_file")

    meta_schema = participants_.drop("ieeg_parquet").schema
    # sort it out here and save by session becausewe do within session stuff
    for row in tqdm(
        participants_.iter_rows(named=True), total=len(participants_), desc="Loading Tables"
    ):
        ieeg_parquet_path = row.pop("ieeg_parquet")
        ieeg_df = pl.read_parquet(ieeg_parquet_path)
        ieeg_df_imploded = ieeg_df.select(pl.all().implode())
        meta_df = pl.DataFrame([row], schema=meta_schema)
        combined_df = pl.concat([meta_df, ieeg_df_imploded], how="horizontal")
        combined_df.write_parquet(tmp_dir / f"{uuid.uuid4()}.parquet")

    logger.info("Combining processed files...")
    final_df = pl.read_parquet(tmp_dir / "*.parquet")
    shutil.rmtree(tmp_dir)
    logger.info("Band-pass resampling complete.")

    return final_df
