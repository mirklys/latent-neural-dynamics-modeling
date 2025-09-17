from preprocessing import initialize_preprocessing
from utils.polars import (
    read_tsv,
    add_modality_path,
    split_file_path,
    remove_rows_with,
    explode_files,
    keep_rows_with,
    dict_to_struct,
)
import argparse

from utils.logger import get_logger
from utils.ieeg import band_pass_resample
import polars as pl
from pathlib import Path

from .intermediate_tables.events import construct_events_table


def main(args):
    config = initialize_preprocessing(args.config)
    logger = get_logger()
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content: \n{config}")
    
    data_path = Path(config.data_directory)

    participants = read_tsv(data_path / config.participants_table_name)
    

    participants = participants.with_columns(
        pl.concat_str(
            pl.lit(str(data_path)), pl.col("participant_id"), separator="/"
        ).alias("participant_path")
    )
    participants = explode_files(participants, "participant_path", "session_path")

    

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
    
    logger.info(participants)
    exit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Package recordings based on a configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    main(args)
