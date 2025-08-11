from collections.abc import Iterable
from logging import getLogger
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import yaml
from scipy.linalg import LinAlgError
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from behavior.loading import load_copydraw_record_yaml
from chrono_split import ChronoGroupsSplit

logger = getLogger("LDA projection")


def projection_performance_metrics(
    scores,
    block_labels: np.ndarray,
    block_indices: np.ndarray,
    trial_indices: np.ndarray,
    detrend=False,
    t_stamps=None,
    reject_outliers=False,
    session: str = "",
    suffix: str = "",
):
    """
    trains LDA (if stim labels are available) or PCA (if labels not available)
    and projects the copyDraw trace features onto the generated subspace

    Parameters
    ----------
    scores : np.array NxD
        array containing the copyDraw behavioral features
    block_labels : np.array Nx1
        stim condition for each of the observations in scores
    block_indices : np.array Nx1
        the block numbers used for the cross validation results
    detrend : bool, optional
        detrend the features in scores across the session. The default is False
    t_stamps : np.array Nx1, optional
        time stamps of the execution time of each of the observations in
        scores. The default is None.
    reject_outliers : bool, optional
        Perform outlier rejection using DBSCAN. After clustering, observations
        not belonging to a cluster are marked as outliers.
        The default is False.
    return_performance : bool, optional
        return decoding performance of the LDA model (AUC).
        The default is False.
    return_proj_model : TYPE, optional
        trained LDA model. The default is False.

    Returns
    -------
    ret : tuple
        [labels,
         onehot_accepted (outliers),
         trained LDA model
         ].

    """
    if block_labels is not None:
        assert scores.shape[0] == block_labels.shape[0]
        projector = LDA(solver="eigen", shrinkage="auto")
    else:
        logger.info("no block labels provided, performing label projection with PCA")
        projector = PCA(n_components=1)
    # detect outliers

    scores_normalized = StandardScaler().fit_transform(scores)
    if reject_outliers:
        clustering = DBSCAN(eps=4).fit(scores_normalized)
        if all(clustering.labels_ == -1):
            print("DBSCAN failed! Using all values.")
            onehot_accepted = np.ones((len(scores_normalized)), dtype=bool)
            ix_accepted = np.where(onehot_accepted)[0]
        else:
            onehot_accepted = clustering.labels_ != -1
            ix_accepted = np.where(onehot_accepted)[0]
    else:
        onehot_accepted = np.ones((len(scores_normalized)), dtype=bool)
        ix_accepted = np.where(onehot_accepted)[0]

    # Note the labels will be used as text, i.e. 'off', 'on'. LDA will sort
    # alphabetically so that positive decission function values belong to
    # 'on' and negative values to 'off'
    l_block_labels = block_labels[ix_accepted]

    if all(l_block_labels == "off"):
        import pdb

        pdb.set_trace()

    l_scores = scores_normalized[ix_accepted]

    if len(ix_accepted) == 0:
        import pdb

        print("-" * 80)
        print(f">>>> {ix_accepted=}")
        print(f">>>> {t_stamps[-8:]=}")
        print("-" * 80)

        pdb.set_trace()

    if detrend:
        if t_stamps is None:
            t_stamps = np.arange(scores.shape[0])
        scores_detrended = scores_normalized
        for ix_feat, c_feat in enumerate(l_scores.T):
            c_model_detrend = LinearRegression().fit(
                t_stamps[ix_accepted].reshape(-1, 1), c_feat
            )
            scores_detrended[:, ix_feat] -= c_model_detrend.predict(
                t_stamps.reshape(-1, 1)
            )
    else:
        scores_detrended = scores_normalized

    projector.fit(scores_detrended[ix_accepted], l_block_labels)
    labels = projector.decision_function(
        StandardScaler().fit_transform(scores_detrended)
    )

    df = create_and_store_fit_details(
        l_scores,
        scores_detrended,
        l_block_labels,
        block_indices,
        trial_indices,
        ix_accepted,
        t_stamps,
        session=session,
        suffix=suffix,
    )
    # px.scatter(df, x="t", y="y", color="stim", facet_col="proj").show()

    return labels, onehot_accepted, projector, df


def create_and_store_fit_details(
    l_scores,
    scores_detrended,
    l_block_labels,
    block_indices,
    trial_indices,
    ix_accepted,
    t_stamps,
    session="",
    suffix: str = "",
):
    projector = LDA(solver="eigen", shrinkage="auto")

    dfs = []

    l_scores_dt = scores_detrended[ix_accepted]

    selected_bi = block_indices.to_numpy()[ix_accepted]
    selected_ti = trial_indices.to_numpy()[ix_accepted]
    sel_t_stamps = t_stamps[ix_accepted]
    l_block_labels = l_block_labels.to_numpy()

    # also add a cross validation
    cv = ChronoGroupsSplit()
    splits = cv.split(l_scores, l_block_labels, selected_bi)
    for i, (_, st) in enumerate(splits):
        try:
            print(
                f"split {i} - test blocks {np.unique(selected_bi[st])} with "
                f"stim {np.unique(l_block_labels[st])}"
            )
            if len(np.unique(l_block_labels[st])) == 1:
                import pdb

                pdb.set_trace()
        except:
            import pdb

            pdb.set_trace()

    for train_data_str, train_data in zip(
        ("normal", "normal_detrended"), (l_scores, l_scores_dt)
    ):
        projector = LDA(solver="eigen", shrinkage="auto")
        projector.fit(train_data, l_block_labels)

        for src_data_str, src_data in zip(
            ("normal", "normal_detrended"), (l_scores, l_scores_dt)
        ):
            # Get the projections on detrended data for reference
            for proj_str in ("transform", "decision_function"):
                if proj_str == "transform":
                    transformer = projector.transform

                elif proj_str == "decision_function":
                    transformer = projector.decision_function

                print(f"projecting {src_data_str=} with {proj_str=}, {train_data_str=}")
                dw = proj_to_data_frame(
                    transformer(src_data),
                    l_block_labels,
                    sel_t_stamps,
                    proj_str=proj_str,
                    train_data_str=train_data_str,
                    src_data_str=src_data_str,
                    split_str="all",
                )
                dw["block"] = selected_bi
                dw["ix_trial"] = selected_ti
                dfs.append(dw)

                for i, (ix_train, ix_test) in enumerate(splits):
                    try:
                        projector = LDA(solver="eigen", shrinkage="auto")
                        projector.fit(train_data[ix_train], l_block_labels[ix_train])
                        if proj_str == "transform":
                            transformer = projector.transform

                        elif proj_str == "decision_function":
                            transformer = projector.decision_function

                        dw = proj_to_data_frame(
                            transformer(src_data[ix_test]),
                            l_block_labels[ix_test],
                            sel_t_stamps[ix_test],
                            proj_str=proj_str,
                            train_data_str=train_data_str,
                            src_data_str=src_data_str,
                            split_str=str(i),
                        )
                        dw["block"] = selected_bi[ix_test]
                        dw["ix_trial"] = selected_ti[ix_test]
                        dfs.append(dw)
                    except LinAlgError as err:
                        # The leading minor of B can be not positive definite
                        print(f"Error {err} on split {i}")

    df = pd.concat(dfs)

    # df.to_csv(f"{session}_projection_scores_{suffix}.csv", index=False)

    return df


def proj_to_data_frame(
    yproj,
    l_block_labels,
    t_stamps,
    proj_str="",
    train_data_str="",
    src_data_str="",
    split_str="",
) -> pd.DataFrame:
    y = yproj.flatten()
    return pd.DataFrame(
        dict(
            y=y,
            stim=l_block_labels,
            t=t_stamps,
            proj=[proj_str] * len(y),
            train_data=[train_data_str] * len(y),
            src_data=[src_data_str] * len(y),
            split=split_str,
        )
    )


def kin_scores(var_pos, delta_t, sub_sampled=False):
    kin_res = {"pos_t": var_pos}

    velocity, velocity_mag = deriv_and_norm(var_pos, delta_t)
    accel, accel_mag = deriv_and_norm(velocity, delta_t)
    jerk, jerk_mag = deriv_and_norm(accel, delta_t)

    N = len(var_pos)

    # average
    # divided by number of timepoints,
    # because delta t was used to calc instead of total t
    kin_res["speed"] = np.sum(velocity_mag) / N
    kin_res["acceleration"] = np.sum(accel_mag) / N
    kin_res["velocity_x"] = np.sum(np.abs(velocity[:, 0])) / N
    kin_res["velocity_y"] = np.sum(np.abs(velocity[:, 1])) / N
    kin_res["acceleration_x"] = np.sum(np.abs(accel[:, 0])) / N
    # matlab code does not compute y values, incorrect indexing
    kin_res["acceleration_y"] = np.sum(np.abs(accel[:, 1])) / N

    # isj
    # in matlab this variable is overwritten
    isj_ = np.sum((jerk * delta_t**3) ** 2, axis=0)
    kin_res["isj_x"], kin_res["isj_y"] = isj_[0], isj_[1]
    kin_res["isj"] = np.mean(isj_)

    kin_res["speed_t"] = velocity * delta_t
    kin_res["accel_t"] = accel * delta_t**2
    kin_res["jerk_t"] = jerk * delta_t**3

    if sub_sampled:
        kin_res = {f"{k}_sub": v for k, v in kin_res.items()}

    return kin_res


def computeScoreSingleTrial(traceLet, template, trialTime):
    trial_results = {}

    # compute avg delta_t
    delta_t = trialTime / traceLet.shape[0]
    trial_results["delta_t"] = delta_t

    # Kinematic scores
    kin_res = kin_scores(traceLet, delta_t)
    trial_results = {**trial_results, **kin_res}

    # sub sample
    traceLet_sub = movingmean(traceLet, 5)
    traceLet_sub = traceLet_sub[::3, :]  # take every third point
    kin_res_sub = kin_scores(traceLet_sub, delta_t * 3, sub_sampled=True)
    trial_results = {**trial_results, **kin_res_sub}

    # dtw
    # dtw_res = dtw_py.dtw_features(traceLet, template)
    dtw_res = {}
    trial_results = {**trial_results, **dtw_res}

    # misc
    # trial_results["dist_t"] = _w_to_dist_t(
    #     trial_results["w"].astype(int),
    #     trial_results["pos_t"],
    #     template,
    #     trial_results["pathlen"],
    # )
    #
    # # normalize distance dt by length of copied template (in samples)
    # trial_results["dt_norm"] = trial_results["dt_l"] / (
    #     trial_results["pathlen"] + 1
    # )
    #
    # # get length of copied part of the template (in samples)
    # trial_results["len"] = (trial_results["pathlen"] + 1) / len(template)

    return trial_results


def _w_to_dist_t(w, trace, template, pathlen, template_idx_in_w: int = 0):
    """This is a copy of how dist_t is computed in matlab.

    with the added feature of informing it as to whether w is indexed the other way ie [trace_idxs, template_idxs]   # noqa
    """

    tmp1 = template[w[:pathlen, template_idx_in_w], :]
    tmp2 = trace[w[:pathlen, int(not template_idx_in_w)]]
    dist_t = np.sqrt(np.sum((tmp1 - tmp2) ** 2, axis=1))
    return dist_t


def process_trial(trial_file: Path, use_longest_only: bool = True):
    """Actual trial post processing takes place here."""

    logger.info(f"Loading trial data for {trial_file=}")
    res = load_copydraw_record_yaml(trial_file)
    res["stim"] = derive_stim(trial_file)

    # scale the template to how it would be on the screen in real pixels
    # as the trace_let is recorded in screen pixel coords
    temp = res["template_pix"] * res["template_scaling"]
    scaled_template = temp - (
        res["template_pos"] / res["scaling_matrix"][0, 0] / res["template_scaling"]
    )

    res["scaled_template"] = scaled_template

    trace = (
        [
            tr
            for tr in res["traces_pix"]
            if len(tr) == max([len(e) for e in res["traces_pix"]])
        ][0]
        if use_longest_only
        else res["trace_let"]
    )

    # do dtw etc
    traceLet = np.asarray(trace)
    template = res["scaled_template"]
    trialTime = res["trial_time"]
    scores = computeScoreSingleTrial(traceLet, template, trialTime)

    return {**res, **scores}


def test_plot_template_vs_tracelet(temp: np.ndarray, trace: np.ndarray):
    """Debugging function to check if the scaling is applied correctly"""

    plt.plot(temp[:, 0], temp[:, 1], color="#5555ff", label="template")
    plt.plot(trace[:, 0], trace[:, 1], color="#ff5555", label="trace")
    plt.legend()


def derive_stim(fpath: Path) -> str:
    """For a given session dir get the stim value for a given block"""
    if fpath.stem.startswith("STIM_OFF_"):
        return "off"
    elif fpath.stem.startswith("STIM_ON_"):
        return "on"
    else:
        logger.warning(f"Cannot derive stim state from {fpath.stem=}")
        return "unknown"


def deriv_and_norm(var, delta_t):
    """
    Given an array (var) and timestep (delta_t), computes the derivative
    for each timepoint and returns it (along with the magnitudes)

    """
    # This is not the same as the kinematic scores in the matlab code!
    deriv_var = np.diff(var, axis=0) / delta_t
    deriv_var_norm = np.linalg.norm(deriv_var, axis=1)
    return deriv_var, deriv_var_norm


# TODO: find an implementation for this
def movingmean(arr, w_size):
    """This is trying to mimic some of the functionality from:
    https://uk.mathworks.com/matlabcentral/fileexchange/41859-moving-average-function
    which (I think) is the function used in compute_scoreSingleTrial.m
    (not in matlab by default). Returns an array of the same size by shrinking
    the window for the start and end points."""

    # round down even window sizes
    if w_size % 2 == 0:
        w_size -= 1

    w_tail = np.floor(w_size / 2)

    arr_sub = np.zeros_like(arr)

    for j, col in enumerate(arr.T):  # easier to work with columns like this
        for i, val in enumerate(col):
            # truncate window if needed
            start = i - w_tail if i > w_tail else 0
            stop = i + w_tail + 1 if i + w_tail < len(col) else len(col)
            s = slice(int(start), int(stop))

            # idxs reversed bc .T
            arr_sub[i, j] = np.mean(col[s])

    return arr_sub


def test_plot_last_test_trial_for_calibration():
    # Load last trial of last run if VPtest
    pth = list(Path("./data/VPtest/copyDraw/raw_behavioral/").rglob("*_trial*.yaml"))[
        -1
    ]
    res = load_copydraw_record_yaml(pth)

    temp = res["template_pix"] * res["template_scaling"]
    trace = res["trace_let"]

    # pos shift
    sm = res["scaling_matrix"]
    ttemp = temp - res["template_pos"] / sm[0, 0] / res["template_scaling"]

    test_plot_template_vs_tracelet(ttemp, trace)


def create_copydraw_results_data(
    copydraw_folder: Path,
    feature_set: str = "STANDARD_PERFORMANCE_METRICS",
    overwrite: bool = False,
    session: str = "",
) -> tuple[pd.DataFrame, Any]:
    """
    For a given folder, collect all data under raw_behavioral and compose
    into a single copydraw data frame

    Steps include:
    - calculation of kinematic scores (velocity, acceleration, jitter)
    - dtw for calculation of a distance metric
    - fitting of the LDA model for projection
    """

    cfg = yaml.safe_load(open("./configs/paradigm_config.yaml"))

    # The results folder
    res_folder = copydraw_folder.resolve().joinpath("projection_results")
    if res_folder.exists():
        if overwrite:
            res_folder.unlink()
        else:
            q = ""
            while q not in ["y", "n"]:
                q = input(
                    f"There is alread a results folder at {res_folder},"
                    f"do you want to overwrite? [y, n]"
                )

            if q == "y":
                res_folder.unlink()
            else:
                return res_folder

    res_folder.mkdir()

    # Old file structure
    trial_files = list(
        copydraw_folder.joinpath("raw_behavioral").rglob("*_block*_trial*.yaml")
    )
    if trial_files == []:
        # New file structure
        trial_files = list(copydraw_folder.parent.rglob("*_block*_trial*.yaml"))

    assert trial_files, f"No trial files found at {copydraw_folder}"
    trial_data = []
    for f in tqdm(trial_files):
        try:
            trial_data.append(process_trial(f))
        except Exception as e:
            logger.error(f"Error processing {f=}, {e=}, skipping file")

    # only get the non iterables, to keep the frame lean
    df_scores = pd.concat(
        [
            pd.DataFrame(
                {
                    k: v
                    for k, v in d.items()
                    if isinstance(v, str) or not isinstance(v, Iterable)
                },
                index=[0],
            )
            for d in trial_data
        ]
    ).reset_index(drop=True)

    df_scores["startTStamp"] = (
        df_scores["start_t_stamp"] - df_scores["start_t_stamp"].min()
    )
    df_scores["startTStamp"] /= 60  # in minutes

    # Get the labels :)
    y, ix_clean, model, df_splits = projection_performance_metrics(
        scores=df_scores[cfg[feature_set]],
        block_labels=df_scores["stim"],
        block_indices=df_scores["ix_block"],
        trial_indices=df_scores["ix_trial"],
        detrend=True,
        t_stamps=df_scores["startTStamp"].values,
        reject_outliers=True,
        session=session,
    )

    df_scores["final_label"] = y
    df_scores["final_clean"] = ix_clean

    # store model and
    df_scores.to_hdf(
        res_folder.joinpath("motoric_scores.hdf"), key="joined_motoric_scores"
    )
    joblib.dump(model, res_folder.joinpath("proj_model.joblib"))
    df_splits.to_hdf(res_folder.joinpath("lda_cross_val.hdf"), key="lda_cross_val")

    return df_scores, model


def create_label_per_step(
    trial_data,
    model,
    feature_set,
    cfg,
    df_scores,
):
    """
    Project the metric along time - currently this only works with the
    STANDARD_PERFORMANCE_METRICS
    """
    if feature_set != "STANDARD_PERFORMANCE_METRICS":
        raise ValueError("Currently only works with STANDARD_PERFORMANCE_METRICS")

    data = []
    for td in trial_data:
        min_len = min(
            [len(td[k]) for k in ["speed_t_sub", "accel_t_sub", "jerk_t_sub"]]
        )
        df = pd.DataFrame(
            {
                "speed_sub": td["speed_t_sub"].mean(axis=1)[-min_len:],
                "velocity_x_sub": td["speed_t_sub"][-min_len:, 0],
                "velocity_y_sub": td["speed_t_sub"][-min_len:, 1],
                "acceleration_sub": td["accel_t_sub"].mean(axis=1)[-min_len:],
                "acceleration_x_sub": td["accel_t_sub"][-min_len:, 0],
                "acceleration_y_sub": td["accel_t_sub"][-min_len:, 1],
                "isj_sub": td["jerk_t_sub"].mean(axis=1)[-min_len:],
                "isj_x_sub": td["jerk_t_sub"][-min_len:, 0],
                "isj_y_sub": td["jerk_t_sub"][-min_len:, 1],
            }
        )
        df["ix_block"] = td["ix_block"]
        df["ix_trial"] = td["ix_trial"]
        df["start_t_stamp"] = td["start_t_stamp"]
        df["stim"] = td["stim"]
        data.append(df)

    df = pd.concat(data)
    df["final_label"] = model.decision_function(df[cfg[feature_set]])

    df = df.sort_values(["ix_block", "ix_trial", "start_t_stamp"])

    fig = px.scatter(
        df, y="speed_sub", color="stim", facet_col="ix_block", facet_col_wrap=4
    )
    fig.show()

    dg = (
        df.groupby(["ix_block", "ix_trial", "stim"])["final_label"].mean().reset_index()
    )
    dg["calc"] = "per_sample_proj_means"
    dt = df_scores[["ix_block", "ix_trial", "stim", "final_label"]].copy()
    dt["calc"] = "means_proj"
    dm = pd.concat([dg, dt])
    fig = px.box(
        dm,
        x="ix_block",
        y="final_label",
        color="stim",
        # points="all",
        facet_row="calc",
    )
    figp = px.scatter(
        dm,
        x="ix_block",
        y="final_label",
        color="stim",
        # points="all",
        facet_row="calc",
        marginal_y="violin",
    )
    for tr in fig["data"]:
        if tr["xaxis"] == "x":
            figp.add_trace(tr, row=1, col=1)
        else:
            figp.add_trace(tr, row=2, col=1)

    figp.update_layout(title="Sess - per sample vs mean projections")

    figp.show()


def plot_projection(df_scores):
    ix_clean = df_scores.final_clean
    auc = roc_auc_score(df_scores["stim"], df_scores["final_label"])
    plt.scatter(
        df_scores[np.logical_not(ix_clean)]["startTStamp"],
        df_scores[np.logical_not(ix_clean)]["final_label"],
        c="k",
        label="outlier",
    )
    plt.title(f"Total (overfitted) AUC {auc:.3%}")
    sns.scatterplot(
        x="startTStamp", y="final_label", hue="stim", data=df_scores[ix_clean]
    )
    plt.show()


def plot_projection_plotly(df: pd.DataFrame):
    ix_clean = df.final_clean
    fig = px.scatter(
        df[ix_clean],
        x="startTStamp",
        y="final_label",
        color="stim",
        marginal_y="box",
    )

    fig.show()


if __name__ == "__main__":
    sessions = [
        "",
    ]
    # for session in sessions:
    session = sessions[0]
    copydraw_folder = Path(f"../../../data/{session}/behavioral/copydraw/")
    feature_set = "STANDARD_PERFORMANCE_METRICS"

    df, model = create_copydraw_results_data(
        copydraw_folder=copydraw_folder,
        feature_set=feature_set,
        session=session,
    )

    import plotly.express as px

    px.scatter(df[df.final_clean], x="startTStamp", y="final_label", color="stim").show(
        renderer="browser"
    )

    # quick feature importance plot
    cfg = yaml.safe_load(open("./configs/paradigm_config.yaml"))
    ex_feat = cfg["EXTENDED_PERFORMANCE_METRICS"]
    dfeat = pd.DataFrame(
        {"feature": ex_feat[: len(model.coef_[0])], "weight": model.coef_[0]}
    )
    px.bar(dfeat, x="feature", y="weight").show()
