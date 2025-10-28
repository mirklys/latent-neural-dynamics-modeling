from utils.config import Config
from utils.frameworks import PSIDFramework, DPADFramework
import polars as pl
from pathlib import Path
from utils.split import create_splits


class Trainer:

    def __init__(self, config: Config):
        self.model_params = config.model
        self.data_params = config.data
        self.training_params = config.training
        self.results_config = config.results
        self.search = config.search

    def split_data(self):
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
            pl.read_parquet(self.data_params.root)
            .select(
                pl.col("participant_id"),
                pl.col("session"),
                pl.col("block"),
                pl.col("trial"),
                *combined_cols,
                pl.col("onset"),
                pl.col("margined_duration"),
                "time",
                "start_ts",
                pl.col("trial_length_ts"),
                "chunk_margin",
                "dbs_stim",
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
        )

        create_splits(session, self.data_params.split, self.results_config)

    def train(self):
        self.framework_type = self.model_params.name.split("_")[0]

        if self.framework_type == "psid":
            self.framework = PSIDFramework(
                self.model_params,
                self.data_params,
                self.training_params,
                self.results_config,
                self.search,
            )
        elif self.framework_type == "dpad":
            self.framework = DPADFramework(
                self.model_params,
                self.data_params,
                self.training_params,
                self.results_config,
                self.search,
            )
        else:
            raise ValueError(f"Unknown framework type: {self.framework_type}")
