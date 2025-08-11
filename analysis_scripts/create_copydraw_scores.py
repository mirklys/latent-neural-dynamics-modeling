# Post processing the behavioral features to CopyDraw scores
#
import shutil
from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from behavior.post_processing.process_copydraw_yaml import process_trial
from behavior.post_processing.projection import projection_performance_metrics

DATA_ROOT = Path("./data")


def ask_to_overwrite(res_folder: Path, overwrite: bool = False):
    if res_folder.exists():
        if overwrite:
            shutil.rmtree(res_folder)
        else:
            q = ""
            while q not in ["y", "n"]:
                q = input(
                    f"There is alread a results folder at {res_folder},"
                    f"do you want to overwrite? [y, n]"
                )

            if q == "y":
                shutil.rmtree(res_folder)
            else:
                return False
    return True


def process_trial_files(trial_files: list[Path]) -> pd.DataFrame:
    """Process list of individual trial files into a dataframe"""
    dfs = []
    for f in tqdm(trial_files):
        try:
            d = process_trial(f)
            di = pd.DataFrame(
                {
                    k: v
                    for k, v in d.items()
                    if isinstance(v, str) or not isinstance(v, Iterable)
                },
                index=[0],
            )
            di["file"] = f
            dfs.append(di)
        except Exception as e:
            print(f"Error processing {f=}, {e=}, skipping file")

    df = pd.concat(dfs).reset_index(drop=True)

    df["startTStamp"] = df["start_t_stamp"] - df["start_t_stamp"].min()
    df["startTStamp"] /= 60  # in minutes

    return df


def train_and_attach_lda(
    df: pd.DataFrame, feature_set: str, cfg: dict
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    """
    Train an LDA model and attach the results to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the data.
    feature_set : str
        The feature set to use for training the LDA model.
    cfg : dict
        Configuration dictionary containing model parameters.

    Returns
    -------
    tuple[pd.DataFrame, object, pd.DataFrame]
        A tuple containing the modified dataframe with LDA results, the trained LDA model, and the LDA components dataframe.
    """

    # Get the labels :)
    y, ix_clean, model, df_splits = projection_performance_metrics(
        scores=df[cfg[feature_set]],
        # scores=df[cfg['EXTENDED_PERFORMANCE_METRICS']],
        block_labels=df["stim"],
        block_indices=df["ix_block"],
        trial_indices=df["ix_trial"],
        detrend=True,
        t_stamps=df["startTStamp"].values,
        reject_outliers=True,
        session=session,
    )

    df["final_label"] = y
    df["final_clean"] = ix_clean

    return df, model, df_splits


def plot_results(df: pd.DataFrame, model: object):
    """Quick scatter plot to investigate features"""

    px.scatter(df[df.final_clean], x="startTStamp", y="final_label", color="stim").show(
        renderer="browser"
    )

    # quick feature importance plot
    cfg = yaml.safe_load(open("./configs/paradigm_config.yaml"))
    ex_feat = cfg["STANDARD_PERFORMANCE_METRICS"]
    dfeat = pd.DataFrame(
        {"feature": ex_feat[: len(model.coef_[0])], "weight": model.coef_[0]}
    )
    px.bar(dfeat, x="feature", y="weight").show()


def create_copydraw_scores(
    session: str,
    feature_set: str = "STANDARD_PERFORMANCE_METRICS",
    overwrite: bool = False,
    plot: bool = False,
    proj_with_existing_model: bool = False,
    existing_model: Path = None,
) -> int:
    """
    Create CopyDraw scores for a given session.

    Parameters
    ----------
    session : str
        The session identifier (e.g., "day2", "day3", "day4").
    feature_set : str, optional
        The feature set to use for training the model, by default "STANDARD_PERFORMANCE_METRICS".
    overwrite : bool, optional
        Whether to overwrite existing results, by default False.
    plot : bool, optional
        Whether to plot the results, by default False.
    proj_with_existing_model : bool, optional
        Whether to project with an existing model, by default False.
    existing_model : Path, optional
        Path to the existing model file, by default None.

    Returns
    -------
    int
        Returns -1 if the process is aborted, otherwise 0.
    """

    cfg = yaml.safe_load(open("./configs/paradigm_config.yaml"))
    copydraw_folder = DATA_ROOT.joinpath(f"sub-p001_ses-{session}", "behavioral")

    # The results folder
    res_folder = copydraw_folder.resolve().joinpath("projection_results")
    continue_val = ask_to_overwrite(res_folder, overwrite)
    if not continue_val:
        return -1

    res_folder.mkdir(exist_ok=True, parents=True)

    trial_files = list(copydraw_folder.rglob("*_block*_trial*.yaml"))
    assert trial_files, f"No trial files found at {copydraw_folder}"

    df = process_trial_files(trial_files)

    if session == "day2":
        # block 01 was aborted after 2 trials, block 14 aborted after 6 trials
        df = df[(df.ix_block > 1) & (df.ix_block < 14)]

    df = df.sort_values(["ix_block", "ix_trial"]).reset_index(drop=True)

    # relevant for day 4
    if proj_with_existing_model:
        model = joblib.load(existing_model)
        df = add_copy_draw_score_from_existing_model(
            df, existing_model, feature_set, cfg
        )

        df.to_hdf(
            res_folder.joinpath("motoric_scores.hdf"),
            key="joined_motoric_scores",
        )

    else:
        df, model, df_splits = train_and_attach_lda(df, feature_set, cfg)

        # store model and
        df.to_hdf(
            res_folder.joinpath("motoric_scores.hdf"),
            key="joined_motoric_scores",
        )
        joblib.dump(model, res_folder.joinpath("proj_model.joblib"))
        df_splits.to_hdf(res_folder.joinpath("lda_cross_val.hdf"), key="lda_cross_val")
        df_splits.to_hdf(
            Path("./data/").joinpath(f"lda_cross_val_{session}.hdf"),
            key="lda_cross_val",
        )

    if plot:
        plot_results(df, model)


def add_copy_draw_score_from_existing_model(
    df: pd.DataFrame,
    model_file: Path = ".data/sub-p001_ses-day3/behavioral/projection_results/proj_model.joblib",
    feature_set: str = "STANDARD_PERFORMANCE_METRICS",
    cfg: dict = yaml.safe_load(open("./configs/paradigm_config.yaml")),
) -> pd.DataFrame:
    model = joblib.load(model_file)

    x = df[cfg[feature_set]]

    xs = StandardScaler().fit_transform(x)
    y = model.decision_function(xs)

    df["final_label"] = y
    df["final_clean"] = True

    return df


if __name__ == "__main__":
    # Move over the sessions extracting the information of individual json files,
    # one per copy draw trace, then aggregate and information and train a classifier
    # based on the average behavioral metrics according to STANDARD_PERFORMANCE_METRICS
    # from `./configs/paradigm_config.yaml`
    session = "day2"
    create_copydraw_scores(session=session)
    session = "day3"
    create_copydraw_scores(session=session)

    session = "day4"
    # for day 4 we project with the day 3 model, as not enough on/off trials available
    # for day 4 to train an individual model
    create_copydraw_scores(
        session=session,
        proj_with_existing_model=True,
        existing_model=Path(
            "./data/sub-p001_ses-day3/behavioral/projection_results/proj_model.joblib"
        ),
    )

    # Persist all motor score information in a single hdf file
    dfs = pd.concat(
        [
            pd.read_hdf(
                f"./data/sub-p001_ses-{day}/behavioral/projection_results/motoric_scores.hdf"
            ).assign(session=day)
            for day in ["day2", "day3", "day4"]
        ]
    )
    dfs.to_hdf("./data/behavioral_data_closed_loop.hdf", key="motoric_scores")
