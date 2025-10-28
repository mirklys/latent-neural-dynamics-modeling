from utils.config import get_config
from utils.logger import get_logger
import argparse
from training.components.trainer import Trainer


def train(config):
    logger = get_logger()
    logger.info("Initializing training...")
    logger.info(f"Training configuration:\n{config}")

    trainer = Trainer(config)
    trainer.split_data()
    trainer.train()

    logger.info("Training completed successfully!")


def main(args):
    config = get_config(args.config)
    logger = get_logger()
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content:\n{config}")

    train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model using specified configuration."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    main(args)
