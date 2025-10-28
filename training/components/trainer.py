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

        inputs_cols = [
            pl.col(f"^{inpt_col}.*$") for inpt_col in self.model_params.input
        ]
        output_cols = [
            pl.col(f"^{outpt_col}.*$") for outpt_col in self.model_params.output
        ]

        session = pl.read_parquet(self.data_params.data_path).select(
            pl.col("participant_id"),
            pl.col("session"),
            pl.col("block"),
            pl.col("trial"),
            *inputs_cols,
            *output_cols,
            pl.col("onset"),
            pl.col("duration"),
            "time",
            "start_ts",
            pl.col("chunk_length_ts").alias("trial_length_ts"),
            "chunk_margin",
            "dbs_stim",
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
