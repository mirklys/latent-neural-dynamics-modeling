import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from utils.config import Config
import torch
from torch.utils.data import Dataset
from utils.logger import get_logger


class TrialDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        data_params: Config,
        split: str = "train",
    ):
        self.parquet_path = Path(parquet_path)
        self.data_params = data_params
        self.split = split

        self.df = pl.read_parquet(self.parquet_path)

        self.input_channels = self.data_params.channels.input
        self.output_channels = self.data_params.channels.output
        self.is_neural_behavioral = self.data_params.channels.is_neural_behavioral
        self.preprocess = self.data_params.preprocess

        self._input_mean = None
        self._input_std = None
        self._output_mean = None
        self._output_std = None

        logger = get_logger()
        logger.info(
            f"Loaded split='{self.split}' dataset from {self.parquet_path} with {len(self.df)} trials; "
            f"input_channels={self.input_channels}, output_channels={self.output_channels}, "
            f"is_neural_behavioral={self.is_neural_behavioral}, preprocess={self.preprocess}"
        )

        if self.split == "train" and self.preprocess:
            self._compute_preprocessing_stats()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        row = self.df[idx]

        Y = self._extract_channels(row, self.input_channels)

        if self.is_neural_behavioral:
            Z = self._extract_channels(row, self.output_channels)
        else:
            Z = None

        if self.preprocess:
            Y = self._preprocess_data(Y, self._input_mean, self._input_std)
            if Z is not None:
                Z = self._preprocess_data(Z, self._output_mean, self._output_std)

        time_vec = row["time"][0] if "time" in row.columns else None
        fs = 1 / (np.mean(np.diff(time_vec)))
        chunk_margin = row["chunk_margin"][0] if "chunk_margin" in row.columns else None
        chunk_margin_ts = int(np.round(chunk_margin * fs))
        margined_duration = (
            row["margined_duration"][0] if "margined_duration" in row.columns else None
        )
        stim = row["stim"][0] if "stim" in row.columns else None

        offset = row["offset"][0] if "offset" in row.columns else None

        metadata = {
            "participant_id": row["participant_id"][0],
            "session": row["session"][0],
            "block": row["block"][0],
            "trial": row["trial"][0],
            "trial_length": Y.shape[0],
            "time": np.array(time_vec) if time_vec is not None else None,
            "offset": float(offset) if offset is not None else None,
            "chunk_margin": chunk_margin,
            "margined_duration": margined_duration,
            "stim": stim,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "fs": fs,
            "chunk_margin_ts": chunk_margin_ts
        }

        logger = get_logger()
        logger.info(f"Loaded trial idx={idx} with metadata: {metadata}")

        return Y, Z, metadata

    def _extract_channels(
        self, row: pl.DataFrame, channel_list: List[str]
    ) -> np.ndarray:
        channel_data = []

        for channel_prefix in channel_list:
            matching_cols = [
                col for col in row.columns if col == f"{channel_prefix}"
            ]

            if not matching_cols:
                raise ValueError(
                    f"No columns found for channel prefix: {channel_prefix}"
                )

            for col in matching_cols:
                trial = row[col][0]
                channel_data.append(np.array(trial))

        if len(channel_data) == 1:
            result = channel_data[0].reshape(-1, 1)
        else:
            result = np.column_stack(channel_data)

        return result

    def _compute_preprocessing_stats(self):
        input_data_list = []
        output_data_list = []

        for idx in range(len(self)):
            row = self.df[idx]
            Y = self._extract_channels(row, self.input_channels)
            input_data_list.append(Y)

            if self.is_neural_behavioral:
                Z = self._extract_channels(row, self.output_channels)
                output_data_list.append(Z)

        all_input_data = np.vstack(input_data_list)
        self._input_mean = np.mean(all_input_data, axis=0, keepdims=True)
        self._input_std = np.std(all_input_data, axis=0, keepdims=True)

        if self.is_neural_behavioral:
            all_output_data = np.vstack(output_data_list)
            self._output_mean = np.mean(all_output_data, axis=0, keepdims=True)
            self._output_std = np.std(all_output_data, axis=0, keepdims=True)

    def _preprocess_data(
        self, data: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]
    ) -> np.ndarray:

        data = data - mean
        data = data / std

        return data

    def set_preprocessing_stats(
        self,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        output_mean: Optional[np.ndarray] = None,
        output_std: Optional[np.ndarray] = None,
    ):
        self._input_mean = input_mean
        self._input_std = input_std
        self._output_mean = output_mean
        self._output_std = output_std

    def get_preprocessing_stats(self) -> dict:
        return {
            "input_mean": self._input_mean,
            "input_std": self._input_std,
            "output_mean": self._output_mean,
            "output_std": self._output_std,
        }

    def get_all_data(self) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        Y_list = []
        Z_list = [] if self.is_neural_behavioral else None
        meta_list = []
        for idx in range(len(self)):
            Y, Z, meta = self[idx]
            Y_list.append(Y)
            meta_list.append(meta)
            if Z is not None:
                Z_list.append(Z)

        return Y_list, Z_list, meta_list


class TrialDataLoader:

    def __init__(
        self,
        dataset: TrialDataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            Y_batch = [item[0] for item in batch]
            Z_batch = [item[1] for item in batch] if batch[0][1] is not None else None
            metadata_batch = [item[2] for item in batch]

            yield Y_batch, Z_batch, metadata_batch

    def get_full_dataset(self) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        return self.dataset.get_all_data()


def create_dataloaders(
    data_params: Config,
    results_config: Config,
) -> Tuple[TrialDataLoader, TrialDataLoader, TrialDataLoader]:

    split_dir = Path(results_config.save_dir) / "split"
    logger = get_logger()
    logger.info(f"Creating dataloaders from split directory: {split_dir}")

    train_dataset = TrialDataset(
        split_dir / "train.parquet", data_params, split="train"
    )

    val_dataset = TrialDataset(split_dir / "val.parquet", data_params, split="val")

    test_dataset = TrialDataset(split_dir / "test.parquet", data_params, split="test")

    stats = train_dataset.get_preprocessing_stats()
    val_dataset.set_preprocessing_stats(
        stats["input_mean"],
        stats["input_std"],
        stats["output_mean"],
        stats["output_std"],
    )
    test_dataset.set_preprocessing_stats(
        stats["input_mean"],
        stats["input_std"],
        stats["output_mean"],
        stats["output_std"],
    )

    in_shape = stats["input_mean"].shape
    out_shape = stats["output_mean"].shape if stats["output_mean"] is not None else None
    logger.info(
        f"Loaded datasets: train={len(train_dataset)} trials, val={len(val_dataset)} trials, test={len(test_dataset)} trials; "
        f"preprocessing shapes -> input_mean: {in_shape}, output_mean: {out_shape}"
    )

    batch_size = data_params.batch_size

    train_loader = TrialDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = TrialDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = TrialDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info(
        f"Constructed dataloaders with batch_size={batch_size}. "
        f"Train steps/epoch≈{len(train_loader)}, Val steps≈{len(val_loader)}, Test steps≈{len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
