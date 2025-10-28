from utils.config import Config
import polars as pl
from pathlib import Path


def create_splits(
    recordings: pl.DataFrame, split_params: Config, results_config: Config
):
    assert split_params.within_session_split, "Only within-session split is supported"

    recordings = recordings.with_columns(
        pl.col("n_epochs").cum_sum().alias("cum_sum_epochs")
    )

    total_epochs = recordings["cum_sum_epochs"][-1]

    if total_epochs == 0:
        raise ValueError("Total epochs cannot be zero. Check the input data.")

    if total_epochs * 0.8 >= split_params.min_train_epochs:
        train_epochs = total_epochs * split_params.train
        val_epochs = total_epochs * split_params.val
        test_epochs = total_epochs * split_params.test
    elif total_epochs * 0.8 < split_params.min_train_epochs <= total_epochs:
        train_epochs = split_params.min_train_epochs
        leftover_epochs = total_epochs - train_epochs

        val_ratio = split_params.val / (split_params.val + split_params.test)
        val_epochs = leftover_epochs * val_ratio
        test_epochs = leftover_epochs * (1 - val_ratio)
    else:
        raise ValueError(
            "Not enough total epochs to satisfy the minimum training requirement."
        )

    train_trials = recordings.filter(pl.col("cum_sum_epochs") <= train_epochs)

    remaining_trials = recordings.filter(~pl.col("trial").is_in(train_trials["trial"]))
    val_trials = remaining_trials.filter(
        pl.col("cum_sum_epochs") <= train_epochs + val_epochs
    )
    test_trials = remaining_trials.filter(~pl.col("trial").is_in(val_trials["trial"]))

    save_dir = Path(results_config.save_dir) / "split"
    save_dir.mkdir(parents=True, exist_ok=True)
    train_trials.write_parquet(save_dir / "train.parquet")
    val_trials.write_parquet(save_dir / "val.parquet")
    test_trials.write_parquet(save_dir / "test.parquet")
