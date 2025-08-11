# Decoding CopyDraw scores from ECoG data
import pickle
import re
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dareplane_utils.signal_processing.filtering import FilterBank
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from chrono_split import ChronoGroupsSplit
from load_xdf import create_raws_from_mat_and_xdf, get_xdf_files

DATA_ROOT = Path("./data")


def create_epoch(raws: mne.io.BaseRaw, files: list[Path]):
    """
    Slice raw data into epochs based on the start markers and add metadata.
    Stimulation and block number are extracted from the file names.
    """

    epochs = []
    for r, f in tqdm(zip(raws, files)):
        block = int(re.findall(r"block(\d+)", f.name)[0])
        stim = re.findall(r"copydraw_([^_\.]+)", f.name)[0]
        if "clcopydraw" in f.name:
            stim = "on_cl"

        ev, evid = mne.events_from_annotations(r)
        imap = {v: int(k) for k, v in evid.items()}
        ev[:, 2] = [imap[v] for v in ev[:, 2]]

        if len(ev) == 12 and (ev[:, 2] == 1).all():
            # for day3 we only have 1s as markers in the AO
            ev[:, 2] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        start_marker = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        sev = ev[np.isin(ev[:, 2], start_marker)]

        if len(sev) > 0:
            epo = mne.Epochs(r, sev, tmin=-0.5, tmax=10, baseline=None, preload=True)

            epo.metadata = pd.DataFrame(
                {
                    "ix_block": [block] * len(epo),
                    "stim": [stim] * len(epo),
                    "ix_trial": [
                        start_marker.index(v) + 1
                        for ii, v in enumerate(sev[:, 2])
                        if epo.drop_log[ii] == ()
                    ],
                    "file": [f.name] * len(epo),
                }
            )
            epochs.append(epo)

    return mne.concatenate_epochs(epochs)


def create_bandpass_filter_features_from_raws(
    raws: list[mne.io.BaseRaw],
    bands: dict = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma_low": (30, 45),
        "gamma_high": (55, 70),
    },
) -> list[np.ndarray]:
    """
    Create bandpass filter features from raw ECoG/LFP data.

    Parameters
    ----------
    raws : list[mne.io.BaseRaw]
        List of raw ECoG/LFP data objects.
    bands : dict, optional
        Dictionary of frequency bands with their corresponding frequency ranges, by default {
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma_low": (30, 45),
            "gamma_high": (55, 70),
        }

    Returns
    -------
    list[np.ndarray]
        List of numpy arrays containing the bandpass filter features for each frequency band.
        Dimensions: (n_timepoints, n_channels, n_bands)
    """

    max_t_len = max([r.times[-1] - r.times[0] for r in raws])

    # Create a filter bank with a buffer to fit the whole epochs
    fb = FilterBank(
        bands,
        n_in_channels=len(raws[0].ch_names),
        sfreq=raws[0].info["sfreq"],
        filter_buffer_s=max_t_len,
        n_lookback=int(raws[0].info["sfreq"] * 0.1),  # for moving average
    )

    X = []

    # Filter every epoch with the online filter
    for i, raw in enumerate(raws):
        data = raw.get_data()

        fb.filter(data.T, raw.times)

        X.append(fb.get_data())
        fb.n_new = 0

    return X


def add_behavioral_metadata(epo: mne.Epochs, session: str):
    """
    Add the behavioral data as meta data to the epochs object. Behavior data
    is generated via `./create_copydraw_scores.py`
    """

    pro_folder = DATA_ROOT.joinpath(
        f"sub-p001_ses-{session}", "behavior", "projection_results"
    )

    hdfs = list(pro_folder.glob("*motoric_scores.hdf"))

    if len(hdfs) > 1:
        print(f"Found multiple hdfs in {pro_folder=}, continuing with" f" {hdfs[0]=}")
    df = pd.read_hdf(hdfs[0])

    dm = epo.metadata

    # get the relevant blocks, assuming that dm is sorted by block
    df = df[df.ix_block.isin(dm.ix_block)]
    df = df.sort_values(["startTStamp"]).reset_index(drop=True)

    # if closed loop session -> add stim to the criterion
    if session == "day4":
        dm.loc[dm.file.str.contains("clcopydraw"), "stim"] = "on_cl"
        dmm = pd.merge(
            dm,
            df[[c for c in df.columns]],  # use stim label already with metadata
            left_on=["ix_block", "ix_trial", "stim"],
            right_on=["ix_block", "ix_trial", "stim"],
            how="left",
            indicator=True,
        )

    else:
        dmm = pd.merge(
            dm,
            df[
                [c for c in df.columns if c != "stim"]
            ],  # use stim label already with metadata
            left_on=["ix_block", "ix_trial"],
            right_on=["ix_block", "ix_trial"],
            how="left",
            indicator=True,
        )

    if any(dmm._merge != "both"):
        print(
            ">>> WARNING: Merging of behavioral data was incomplete!"
            f"{dmm[dmm._merge != 'both']}"
        )

    epo = epo[dmm._merge == "both"]
    dm = dmm[dmm._merge == "both"].drop(columns="_merge")
    epo.metadata = dm
    return epo


def cross_validate(
    X: np.ndarray,
    dm: pd.DataFrame,
    model: Pipeline,
    train: bool = True,
    split_col: str = "stim",
) -> tuple[Pipeline, pd.DataFrame]:
    """
    Perform cross-validation on the given data and model.

    Parameters
    ----------
    X : np.ndarray
        The input features for the model (n_trials, n_features).
    dm : pd.DataFrame
        The dataframe containing the metadata (n_trials, n_features).
    model : Pipeline
        The machine learning pipeline to be used for training and validation.
    train : bool, optional
        Whether to train the model, by default True. If false, just evaluate
        with the model without retraining.
    split_col : str, optional
        The column name in the dataframe used for balancing the splitting of the
        data, by default "stim". Splits always consider the block number which
        are expected to be in dm['ix_block'].

    Returns
    -------
    tuple[Pipeline, pd.DataFrame]
        A tuple containing the trained model pipeline and the dataframe with cross-validation results.
    """

    dm = dm[dm.use].reset_index(drop=True)
    cv = ChronoGroupsSplit()
    splits = cv.split(X, dm[split_col], dm.ix_block)

    # just for checking
    stim_block_pairs = [
        [
            (blk, stim)
            for blk, stim in zip(
                dm.iloc[ix_test, :].ix_block.unique(),
                dm.iloc[ix_test, :][split_col].unique(),
            )
        ]
        for _, ix_test in splits
    ]
    labels = [
        f"block={r[0][0]}|stim={r[0][1]} - block={r[1][0]}|" f"stim={r[1][1]}"
        for r in stim_block_pairs
    ]

    scores = []
    ypreds = []
    ytrues = []
    iss = []
    blocks = []
    pearsonrs = []
    ix_tests = []
    for i, (ix_train, ix_test) in enumerate(splits):
        if train:
            fold_pl = clone(model)
            fold_pl.fit(X[ix_train], dm.final_label[ix_train])
        else:
            fold_pl = model
        ypred = fold_pl.predict(X[ix_test])
        scores.append([r2_score(dm.final_label[ix_test], ypred)] * len(ix_test))
        pearsonrs.append(
            [pearsonr(dm.final_label[ix_test], ypred).statistic] * len(ix_test)
        )
        ypreds.append(ypred)
        ytrues.append(dm.final_label[ix_test])
        iss.append([i] * len(ix_test))
        blocks.append([labels[i]] * len(ix_test))
        ix_tests.append(ix_test)

    if train:
        model.fit(X, dm.final_label)

    dr = pd.DataFrame(
        {
            "scores": np.hstack(scores),
            "pearsonr": np.hstack(pearsonrs),
            "ypred": np.hstack(ypreds),
            "ytrue": np.hstack(ytrues),
            "i_fold": np.hstack(iss),
            "test_blocks": np.hstack(blocks),
            "ix_test": np.hstack(ix_tests),
        }
    )

    return model, dr


def create_permutation_scores(
    Xl: np.ndarray,
    dm: pd.DataFrame,
    model: Pipeline,
    session: str,
    model_session: str | None = None,
    n_perm: int = 500,
):
    """
    Create permutation scores for the given data and model and store them
    as a numpy array.

    Parameters
    ----------
    Xl : np.ndarray
        The input features for the model (n_trials, n_features).
    dm : pd.DataFrame
        The dataframe containing the metadata (n_trials, n_features).
    model : Pipeline
        The machine learning pipeline to be used for training and validation.
    session : str
        The session identifier.
    model_session : str or None, optional
        The model session identifier, by default None.
    n_perm : int, optional
        The number of permutations to perform, by default 500.

    """
    model_session = session if model_session is None else model_session
    mean_r = []
    # create permutation scores
    for _ in tqdm(range(n_perm)):
        np.random.shuffle(dm.final_label.values)
        if session == "day4":
            model, dr = cross_validate(Xl, dm, model=model, split_col="split")
        else:
            model, dr = cross_validate(
                Xl,
                dm,
                model=model,
            )
        mean_r.append(dr.pearsonr.mean())

    bstrap_file_name = (
        f"./data/bootstrap_n{n_perm}_mean_r_{model_session}_model_{session}.npy"
    )
    np.save(
        bstrap_file_name,
        np.asarray(mean_r),
    )
    mean_r = np.load(bstrap_file_name)

    print(
        f"Pearsonr quantiles for boostrap distribution: {np.quantile(mean_r, [0.05, 0.95])}"
    )


def plot_psds(raws: list[mne.io.BaseRaw]):
    """
    Plot PSDs of a list of raw ECoG/LFP data, each element will appear in a separate
    row.
    """

    dps = []
    for r in raws:
        psd = r.compute_psd(
            fmax=150,
            n_fft=2**17,
            n_overlap=2**12,
        )
        data = psd.data.T
        dps.append(
            pd.DataFrame(10 * np.log10(data) + 120, columns=r.ch_names).assign(
                freqs=psd.freqs, file=r._filenames[0]
            )
        )

    pio.templates.default = "plotly_white"

    dp = pd.concat(dps).melt(id_vars=["freqs", "file"], value_name="PSD [dB]")
    # segments were closed using jumpers, only 2 and 5 are actually used
    dp = dp[~dp.variable.isin(["LFP_3", "LFP_4", "LFP_6", "LFP_7"])]

    colors = px.colors.sample_colorscale(
        "Viridis", np.linspace(0, 1, dp.variable.nunique())
    )
    cmap = dict(zip(dp.variable.unique(), colors))
    fig = px.line(
        dp,
        x="freqs",
        y="PSD [dB]",
        color="variable",
        facet_col="file",
        facet_col_wrap=1,
        facet_row_spacing=0.03,
        height=3600,
        color_discrete_map=cmap,
    ).update_xaxes(title="freqs [Hz]", showticklabels=True)
    fig.show()


def load_data_raw_filtered(session: str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load the raw data for a given session and perform bandpass filtering.
    This will create a few intermediate results as `work in progress` within
    ./data/wip_*.pkl

    Parameters
    ----------
    session : str
        The session identifier.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        A tuple containing the filtered data as a numpy array and the metadata
        as a pandas DataFrame.
    """

    files = get_xdf_files(session)
    files.sort(key=lambda f: int(re.findall(r"block(\d+)", f.name)[0]))

    # load raw
    # raw data for the offline analysis is loaded from the mat files stored
    # on the Neuro Omega. The LSL xdf files exhibited dropped data when the
    # PC load was > 95%.
    raws = create_raws_from_mat_and_xdf(files, day=session)

    for r in tqdm(raws):
        r.resample(300)

    loaded_files = [r._filenames[0] for r in raws]

    bands: dict = {
        # "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "betal_low": (13, 20),
        "beta_high": (20, 30),
        "gamma_low": (30, 45),
        "gamma_high": (55, 70),
    }

    channels = ["ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]
    rraws = [r.copy().pick(channels) for r in raws]
    X = create_bandpass_filter_features_from_raws(rraws, bands=bands)

    # Using the reshape will stack [ECOG_1_theta, ECOG_1_alpha, ...]
    nraws = []
    for data, r in zip(X, rraws):
        # Note, the duration of the last annotation might be slightly longer than the
        # max data range, which is ok as this is the end marker, which gets
        # assigned a duration until the end of recording
        print(f"Creating auxraw for {r._filenames}")
        info = mne.create_info(
            ["_".join([ch, band]) for ch in channels for band in bands.keys()],
            sfreq=r.info["sfreq"],
            ch_types="ecog",
        )
        auxraw = mne.io.RawArray(data.reshape(-1, len(channels) * len(bands)).T, info)
        auxraw.set_annotations(r.annotations)
        nraws.append(auxraw)

    pickle.dump(nraws, open(f"./data/wip_raw_bandpassed_{session}.pkl", "wb"))
    nraws = pickle.load(open(f"./data/wip_raw_bandpassed_{session}.pkl", "rb"))

    loaded_files = [r._filenames[0] for r in rraws]
    epochs = create_epoch(nraws, loaded_files)
    pickle.dump(epochs, open(f"./data/wip_epochs_{session}.pkl", "wb"))
    epochs = pickle.load(open(f"./data/wip_epochs_{session}.pkl", "rb"))

    dm = epochs.metadata
    epo = add_behavioral_metadata(epochs, session)

    pickle.dump(epo, open(f"./data/wip_epo_{session}.pkl", "wb"))
    epo = pickle.load(open(f"./data/wip_epo_{session}.pkl", "rb"))

    dm = epo.metadata

    X = epo.get_data().mean(axis=-1)
    Xl = np.log10(X)
    dm["use"] = dm.final_clean  # for now use all

    return Xl, dm


def decode_ecog_data(session: str, train: bool = True):
    """
    Decode CopyDraw scores from ECOG data. This is sotring the results as
    ./data/wip_{session}_ridge.hdf files and will create the permutation scores
    by calling create_permutation_scores().

    Parameters
    ----------
    session : str
        The session identifier.
    train : bool, optional
        Whether to train the model, by default True.

    """
    Xl, dm = load_data_raw_filtered(session)

    # add auxiliary split column for day 4
    if session == "day4":
        # Here we just group neighboring blocks, as for the aDBS day, grouping
        # neighboring stim ON/OFF does not work
        dm["ix_block_old"] = dm.ix_block.copy()

        dm["ix_block"] = dm.file_x.map(
            dict(zip(dm.file_x.unique(), range(dm.file_x.nunique())))
        )
        dm["split"] = dm.ix_block % 2

    if train:
        model = Pipeline(
            [("StandardScaler", StandardScaler()), ("Regressor", Ridge(alpha=1))]
        )
        split_col = "split" if session == "day4" else "stim"
        model, dr = cross_validate(Xl, dm, model=model, split_col=split_col, train=True)
        joblib.dump(model, f"./data/model_{session}.joblib")

    elif train is False and session == "day4":
        model = joblib.load("./data/model_day3.joblib")
        model, dr = cross_validate(Xl, dm, model=model, split_col="split", train=False)

    # print the correlation - get value per fold (first .mean() as it was
    # just written replicated for every entry)
    rval = dr.groupby("i_fold")["pearsonr"].mean().mean()
    print(f"Mean Pearsonr: {rval}")

    # For day4 store only the results evaluated on the day3 model (for the scatter comparison)
    #
    if session != "day4" or train is False:
        dr.to_hdf(f"./data/wip_{session}_ridge.hdf", key="ridge_decoding")
    # px.scatter(dr, x="ypred", y="ytrue", trendline="ols").show()

    # full prediction -> used for day 4 data with day 3 model
    rval_full = pearsonr(dm.final_label, model.predict(Xl))
    print(f"Full Pearsonr: {rval_full}")

    if train:
        create_permutation_scores(Xl, dm, model, session, n_perm=2000)
    elif train is False and session == "day4":
        create_permutation_scores(
            Xl, dm, model, session, model_session="day3", n_perm=2000
        )


if __name__ == "__main__":
    # Create the decoding results by decoding CopyDraw scores from ECOG data
    # Make sure the CopyDraw results have been created before by running
    # `create_copydraw_scores.py`

    decode_ecog_data(session="day3")
    decode_ecog_data(session="day4", train=False)
    decode_ecog_data(
        session="day4", train=True
    )  # for investigating the weights of the model, knowning that we cannot properly train for day 4
