from preprocessing import initialize_preprocessing
from utils.logger import get_logger
import argparse
from .components.participants import construct_participants_table
from pathlib import Path


def main(args):
    config = initialize_preprocessing(args.config)
    logger = get_logger()
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content: \n{config}")

    participants = construct_participants_table(config)

    save_path = Path(config.save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving participants table to: {save_path}")

    participants.write_parquet(
        save_path / "participants.parquet",
        partition_by=["participant_id", "session", "run"],
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Package recordings based on a configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    main(args)
