import pickle
import json
from datetime import datetime
from utils.config import Config
import polars as pl
from pathlib import Path
from utils.split import create_splits
from training.components.data import create_dataloaders
from utils.logger import get_logger
from utils.miscellaneous import length


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

        epoch_samp = f"{self.data_params.channels.input[0]}_epochs"
        trial = (
            pl.read_parquet(session_path)
            .select(
                pl.col("participant_id"),
                pl.col("session"),
                pl.col("block"),
                pl.col("trial"),
                pl.when(pl.col("time").is_not_null())
                .then(pl.col("time"))
                .otherwise(None)
                .alias("time"),
                pl.col("chunk_margin"),
                pl.col("margined_duration"),
                pl.col("stim"),
                *combined_cols,
                pl.col("onset").alias("offset"),
            )
            .with_columns(pl.col(epoch_samp).list.len().alias("n_epochs"))
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

        if self.data_params.blocks != "all":
            trial = trial.filter(pl.col("block").is_in(self.data_params.blocks))

        dbs_condition = self.data_params.dbs_condition
        if dbs_condition != "both":
            trial = trial.filter(pl.col("stim") == dbs_condition)
            self.logger.info(f"Filtered to {dbs_condition} DBS condition")

        create_splits(trial, self.data_params.split, self.results_config)

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.data_params, self.results_config
        )

    def _slice_data(self, Y_list_margined, Z_list_margined, meta_list):
        (
            _Y,
            _Z,
        ) = (
            [],
            [],
        )
        Z_list_margined = (
            [None] * len(Y_list_margined)
            if Z_list_margined is None
            else Z_list_margined
        )
        for Y, Z, meta in zip(Y_list_margined, Z_list_margined, meta_list):
            chunk_margin = meta["chunk_margin_ts"]

            Y_sliced = Y[chunk_margin:-chunk_margin]

            _Y.append(Y_sliced)
            _Z.append(Z)

        _Z = None if all([_z is None for _z in _Z]) else _Z
        self.logger.info(
            f"Sliced data: Y={length(_Y)}, Z={length(_Z)}, meta={length(meta_list)}"
        )
        return _Y, _Z

    def train(self):
        if self.train_loader is None:
            raise ValueError("Data loaders not initialized. Call split_data() first.")

        self.framework_type = self.model_params.name.split("_")[0]
        self.logger.info(f"Selected framework: {self.framework_type}")

        if self.framework_type == "psid":
            from utils.frameworks import PSIDFramework

            self.framework = PSIDFramework(self.config)
        elif self.framework_type == "dpad":
            from utils.frameworks import DPADFramework

            self.framework = DPADFramework(self.config)
        else:
            raise ValueError(f"Unknown framework type: {self.framework_type}")

        Y_train, Z_train, meta_train = self.train_loader.get_full_dataset()
        Y_val, Z_val, meta_val = self.val_loader.get_full_dataset()

        Y_train, Z_train = self._slice_data(Y_train, Z_train, meta_train)
        Y_val, Z_val = self._slice_data(Y_val, Z_val, meta_val)

        self.logger.info("Beginning training...")
        self.framework._train(Y_train, Z_train)

        self.logger.info("Beginning validation...")
        val_results = self.framework._validate(Y_val)

        self.logger.info(f"Validation complete. Results: {val_results}")

        self.save_results(val_results, self.val_loader.dataset.df, type="val")

        return val_results

    def save_results(self, results: dict, input_df: pl.DataFrame, type: str):

        metrics_df = input_df.with_columns(
            [
                pl.Series(
                    name="pearsonr_per_channel", values=results["pearson_r_per_channel"]
                ),
                pl.Series(name="Y", values=[arr.tolist() for arr in results["Y"]]),
                pl.Series(name="Yp", values=[arr.tolist() for arr in results["Yp"]]),
                pl.Series(
                    name="Zp",
                    values=[
                        arr.tolist() if arr is not None else None
                        for arr in results["Zp"]
                    ],
                ),
                pl.Series(name="Xp", values=[arr.tolist() for arr in results["Xp"]]),
                pl.lit(results["pearson_r_mean"]).alias("pearsonr_mean"),
            ]
        )
        if isinstance(results, dict):
            for k, v in results.items():
                if not isinstance(v, (list, dict)):
                    try:
                        metrics_df = metrics_df.with_columns(
                            pl.lit(v).alias(f"metric_{k}")
                        )
                    except Exception:
                        pass

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.results_config.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{type}_results_{ts}"
        metrics_df.write_parquet(
            out_path, partition_by=["participant_id", "session", "block", "trial"]
        )
        self.logger.info(f"Detailed {type} results saved to {out_path}")

        try:
            model_path = out_dir / f"model_{ts}"

            if self.framework_type == "dpad":
                # DPAD models need to discard TF models before pickling
                # This converts them to weight dictionaries
                self.framework.model.idSys.discardModels()
                
                model_path_pkl = f"{model_path}.pkl"
                with open(model_path_pkl, "wb") as f:
                    pickle.dump(self.framework.model.idSys, f)
                
                self.logger.info(f"Saved DPAD model to {model_path_pkl}")
                
                # Save metadata for reference
                metadata = {
                    "framework_type": "dpad",
                    "nx": self.model_params.nx,
                    "n1": self.model_params.n1,
                    "method_code": self.model_params.method_code,
                    "epochs": self.model_params.epochs,
                }
                with open(out_dir / f"model_{ts}_metadata.json", "w") as f:
                    json.dump(metadata, f)
                
            else:
                # PSID models can be pickled directly
                with open(f"{model_path}.pkl", "wb") as f:
                    pickle.dump(self.framework.model.idSys, f)
                self.logger.info(f"Saved PSID model to {model_path}.pkl")

        except Exception as e:
            self.logger.warning(f"Could not save model/trainer artifacts: {e}")
