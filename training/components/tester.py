import polars as pl

from utils.config import Config, get_config
from utils.frameworks import PSIDFramework
from training.components.data import create_dataloaders
from utils.logger import get_logger
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pickle
from utils.stats import pearson_r_per_channel
import h5py


class Tester:

    def __init__(self, config: Config, run_timestamp: Optional[str] = None):
        self.config = config
        self.model_params = config.model
        self.data_params = config.data
        self.results_config = config.results
        self.logger = get_logger()
        self.framework = None
        self.run_timestamp = run_timestamp

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    @classmethod
    def from_config_file(cls, config_path: str, run_timestamp: Optional[str] = None):
        cfg = get_config(config_path)
        return cls(cfg, run_timestamp=run_timestamp)

    def _init_framework(self):
        framework_type = str(self.model_params.name).split("_")[0]
        if framework_type == "psid":
            self.framework = PSIDFramework(self.config)
        else:
            raise ValueError(
                f"Unknown or unsupported framework for testing: {framework_type}"
            )

    def _load_dataloaders(self):
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.data_params, self.results_config
        )

    def _load_model_for_run(self):
        results_dir = Path(self.results_config.save_dir)

        model_path = results_dir / f"model_{self.run_timestamp}.pkl"
        with open(model_path, "rb") as f:
            idSys = pickle.load(f)
        self._init_framework()
        self.framework.model = self.framework._initalize_model()
        self.framework.model.idSys = idSys
        self.logger.info(f"Loaded model from {model_path}")

    @staticmethod
    def _to_Y_list(dataloader) -> List[np.ndarray]:
        return dataloader.get_full_dataset()[0]

    @staticmethod
    def _get_metrics(
        Y_true: List[np.ndarray],
        Yp: Optional[List[np.ndarray]],
        Zp: Optional[List[np.ndarray]],
        Xp: Optional[List[np.ndarray]],
        meta: Dict[str, List[Any]],
    ) -> Dict[str, Any]:

        pearson_per_trial, pearson_overall_mean = pearson_r_per_channel(Y_true, Yp)

        pearson_trial_means = []
        for r_list in pearson_per_trial:
            valid = [r for r in r_list if not (r is None or np.isnan(r))]
            pearson_trial_means.append(np.mean(valid) if len(valid) > 0 else np.nan)

        return {
            "Y": [y.tolist() for y in Y_true],
            "Yp": [Yp_i.tolist() for Yp_i in Yp],
            "Zp": [Zp_i.tolist() if Zp_i is not None else None for Zp_i in Zp],
            "Xp": [Xp_i.tolist() for Xp_i in Xp],
            "pearson_per_channel": pearson_per_trial,
            "pearson_mean": pearson_trial_means,
            "pearson_overall_mean": pearson_overall_mean,
            "time": meta.get("time", []),
            "offset": meta.get("offset", []),
            "chunk_margin": meta.get("chunk_margin", []),
            "margined_duration": meta.get("margined_duration", []),
            "stim": meta.get("stim", []),
            "participant_id": meta.get("participant_id", []),
            "session": meta.get("session", []),
            "block": meta.get("block", []),
            "trial": meta.get("trial", []),
            "input_channels": meta.get("input_channels", []),
        }

    def run_predictions(self):

        self._load_dataloaders()
        self._load_model_for_run()

        self.results = {}

        input_stats = self.train_loader.dataset.get_preprocessing_stats()

        for split_name, loader in (
            ("train", self.train_loader),
            ("val", self.val_loader),
            ("test", self.test_loader),
        ):
            Y_list, _ = loader.get_full_dataset()
            Zp, Yp, Xp = self.framework._predict(Y_list)
            meta = {
                "time": [],
                "chunk_margin": [],
                "margined_duration": [],
                "stim": [],
                "participant_id": [],
                "session": [],
                "block": [],
                "trial": [],
                "input_channels": loader.dataset.input_channels,
                "offset": [],
            }
            for idx in range(len(loader.dataset)):
                _, _, md = loader.dataset[idx]
                meta["time"].append(md.get("time"))
                meta["chunk_margin"].append(md.get("chunk_margin"))
                meta["margined_duration"].append(md.get("margined_duration"))
                meta["stim"].append(md.get("stim"))
                meta["participant_id"].append(md.get("participant_id"))
                meta["session"].append(md.get("session"))
                meta["block"].append(md.get("block"))
                meta["trial"].append(md.get("trial"))
                meta["offset"].append(md.get("offset"))

            split_results = self._get_metrics(Y_list, Yp, Zp, Xp, meta)

            m = (
                self.model_params.forcast.m
                if hasattr(self.model_params, "forcast")
                else 0
            )
            if m > 0:
                try:
                    # Use per-trial margin (in seconds) if available from metadata
                    margin_list = meta.get("chunk_margin", [])
                    f_res = self.framework.model.validate_forecast(
                        Y_list, margin=margin_list
                    )
                    split_results["forecast"] = f_res
                except Exception as e:
                    self.logger.warning(
                        f"Forecast validation failed for split {split_name}: {e}"
                    )

            split_results["input_mean"] = input_stats.get("input_mean")
            split_results["input_std"] = input_stats.get("input_std")
            self.results[split_name] = split_results

    def save_results(self):
        results_dir = Path(self.results_config.save_dir)
        for k in self.results:
            results_path = (
                results_dir / k / f"test_results_{self.run_timestamp}.parquet"
            )
            results_ = self.results[k].copy()
            del results_["input_channels"]
            del results_["pearson_per_channel"]
            del results_["pearson_mean"]
            del results_["pearson_overall_mean"]
            del results_["input_mean"]
            del results_["input_std"]
            results_df = pl.from_dict(results_)
            results_df.write_parquet(
                results_path,
                partition_by=["participant_id", "session", "block", "trial"],
            )

    def compute_and_save_stats(self):

        def create_dataset(group, name, data):
            if data is None:
                group.create_dataset(name, data=h5py.Empty("f"))
            else:
                group.create_dataset(name, data=data)

        results_dir = Path(self.results_config.save_dir)
        stats_path = results_dir / f"test_stats_{self.run_timestamp}.hdf5"
        with h5py.File(stats_path, "w") as f:
            f.attrs["model_name"] = self.model_params.name
            f.attrs["nx"] = self.model_params.nx
            f.attrs["n1"] = self.model_params.n1
            f.attrs["i"] = self.model_params.i
            f.attrs["run_timestamp"] = self.run_timestamp
            A_Eigs = np.linalg.eig(self.framework.model.idSys.A)[0]
            isStable = np.max(np.abs(A_Eigs)) < 1
            f.attrs["is_stable"] = isStable

            f.create_dataset("A", data=self.framework.model.idSys.A)
            f.create_dataset("Cy", data=self.framework.model.idSys.C)
            f.create_dataset("Cz", data=self.framework.model.idSys.Cz)
            f.create_dataset("Q", data=self.framework.model.idSys.Q)
            f.create_dataset("R", data=self.framework.model.idSys.R)
            f.create_dataset("S", data=self.framework.model.idSys.S)

            prep_group = f.create_group("preprocessing")

            create_dataset(
                prep_group, "Y_mean", self.framework.model.idSys.YPrepModel.mean
            )
            create_dataset(
                prep_group, "Y_std", self.framework.model.idSys.YPrepModel.std
            )
            create_dataset(
                prep_group, "Z_mean", self.framework.model.idSys.ZPrepModel.mean
            )
            create_dataset(
                prep_group, "Z_std", self.framework.model.idSys.ZPrepModel.std
            )

            stats_group = f.create_group("analysis_stats")

            A_11 = self.framework.model.idSys.A[
                0 : self.model_params.n1, 0 : self.model_params.n1
            ]
            stats_group.create_dataset("eigvals_relevant", data=np.linalg.eigvals(A_11))

            A_22 = self.framework.model.idSys.A[
                self.model_params.n1 : self.model_params.nx,
                self.model_params.n1 : self.model_params.nx,
            ]
            stats_group.create_dataset(
                "eigvals_irrelevant", data=np.linalg.eigvals(A_22)
            )

            stats_group.create_dataset(
                "z_readout_norm",
                data=np.linalg.norm(
                    self.framework.model.idSys.Cz[:, 0 : self.model_params.n1], axis=0
                ),
            )
            stats_group.create_dataset(
                "y_readout_norm",
                data=np.linalg.norm(self.framework.model.idSys.C, axis=0),
            )
