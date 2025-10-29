from utils.config import Config
from utils.frameworks import PSIDFramework, DPADFramework
import polars as pl
from pathlib import Path
from utils.split import create_splits
from training.components.data import create_dataloaders
from utils.logger import get_logger


class Trainer:

    def __init__(self, config: Config):
        self.config = config
        self.model_params = config.model
        self.data_params = config.data
        self.results_config = config.results
        self.logger = get_logger()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def split_data(self):
        self.logger.info("Starting data split...")
        self.logger.info(
            f"Data params: participant={self.data_params.participant}, session={self.data_params.session}, "
            f"input_channels={self.data_params.channels.input}, output_channels={self.data_params.channels.output}, "
            f"is_neural_behavioral={self.data_params.channels.is_neural_behavioral}"
        )
        session_path = (
            Path(self.data_params.root)
            / f"participant_id={self.data_params.participant}"
            / f"session={self.data_params.session}"
        )
        combined_cols = list(
            set(self.data_params.channels.input) | set(self.data_params.channels.output)
        )
        combined_cols = [pl.col(f"^{col}.*$") for col in combined_cols]
        session = (
            pl.read_parquet(session_path)
            .select(
                pl.col("participant_id"),
                pl.col("session"),
                pl.col("block"),
                pl.col("trial"),
                *combined_cols,
            )
            .with_columns(pl.col("^.*epochs.*$").list.len().alias("n_epochs"))
            .sort(
                [
                    pl.col("participant_id"),
                    pl.col("session"),
                    pl.col("block"),
                    pl.col("trial"),
                ],
                maintain_order=True,
            )
        ).filter(pl.col("block").is_in(self.data_params.blocks))

        create_splits(session, self.data_params.split, self.results_config)

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.data_params, self.results_config
        )

    def train(self):
        if self.train_loader is None:
            raise ValueError("Data loaders not initialized. Call split_data() first.")

        self.framework_type = self.model_params.name.split("_")[0]
        self.logger.info(f"Selected framework: {self.framework_type}")

        if self.framework_type == "psid":
            self.framework = PSIDFramework(self.config)
        elif self.framework_type == "dpad":
            self.framework = DPADFramework(self.config)
        else:
            raise ValueError(f"Unknown framework type: {self.framework_type}")

        Y_train, Z_train = self.train_loader.get_full_dataset()
        Y_val, Z_val = self.val_loader.get_full_dataset()
        try:
            y_tr_shape = [y.shape for y in Y_train[:3]]
            z_tr_shape = None if Z_train is None else [z.shape for z in Z_train[:3]]
            y_val_shape = [y.shape for y in Y_val[:3]]
            z_val_shape = None if Z_val is None else [z.shape for z in Z_val[:3]]
            self.logger.info(
                f"Data prepared. Train trials={len(Y_train)}, Val trials={len(Y_val)}; "
                f"Y_train (first 3) shapes={y_tr_shape}; Z_train (first 3) shapes={z_tr_shape}; "
                f"Y_val (first 3) shapes={y_val_shape}; Z_val (first 3) shapes={z_val_shape}"
            )
        except Exception:
            pass

        self.logger.info("Beginning training...")
        self.framework._train(Y_train, Z_train)

        self.logger.info("Beginning validation...")
        val_results = self.framework._validate(Y_val, Z_val)
        self.logger.info(f"Validation complete. Results: {val_results}")

        return val_results
