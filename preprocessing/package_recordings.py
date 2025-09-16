from utils.config import get_config
from utils.polars import (
    read_tsv,
    add_modality_path,
    split_file_path,
    remove_rows_with,
    explode_files,
    keep_rows_with,
)
import argparse

from utils.logger import setup_logger, get_logger
import polars as pl
from pathlib import Path

from .intermediate_tables.events import construct_events_table


def main():
    parser = argparse.ArgumentParser(
        description="Package recordings based on a configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    config = get_config(args.config)

    print(f"Configuration loaded from: {args.config}")
    print(f"Config content: \n{config}")

    setup_logger(config.logger_directory, name="packaging_recordings")

    logger = get_logger()

    data_path = Path(config.data_directory)

    participants = read_tsv(data_path / config.participants_table_name)
    logger.info(participants)

    participants = participants.with_columns(
        pl.concat_str(
            pl.lit(str(data_path)), pl.col("participant_id"), separator="/"
        ).alias("participant_path")
    )
    participants = explode_files(participants, "participant_path", "session_path")

    logger.info(participants)

    participants = add_modality_path(participants, "ieeg")
    logger.info(participants)

    participants = explode_files(participants, "ieeg_path", "ieeg_file")

    participants = split_file_path(participants, "ieeg", {"run": -2, "session": -4})
    participants = remove_rows_with(participants, type="channels", data_format="tsv")
    logger.info(participants)
    events = construct_events_table(participants)

    participants = participants.join(
        events, on=["participant_id", "session", "run"], how="left"
    )

    participants = remove_rows_with(participants, type="events", data_format="tsv")
    headers = keep_rows_with(participants).select(
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

    logger.info(participants)


if __name__ == "__main__":
    main()
