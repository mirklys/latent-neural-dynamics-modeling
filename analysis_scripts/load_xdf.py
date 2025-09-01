# This file contains the loading utility functions used for processing xdf and mat files
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import plotly.io as pio
import pyxdf
from rich.console import Console
from scipy.io import loadmat
from tqdm import tqdm

pio.templates.default = "simple_white"
DATA_ROOT = Path("./data")

AO_STREAM_NAME = "AODataStream"
ECOG_CHANNELS_PATTERN = "CECOG_HF_2___[0-4]*___Array_3___[0-4]*"
LFP_CHANNELS_PATTERN = "CECOG_HF_1___[0-9]*___Array_[12]___[0-9]*"

# Map of the channel names from the raw `*.mat` files provided by the Neuro Omega
CHANNEL_MAP = {
    "CECOG_HF_1___01___Array_1___01": "LFP_1",
    "CECOG_HF_1___02___Array_1___02": "LFP_2",
    "CECOG_HF_1___03___Array_1___03": "LFP_3",
    "CECOG_HF_1___04___Array_1___04": "LFP_4",
    "CECOG_HF_1___05___Array_1___05": "LFP_5",
    "CECOG_HF_1___06___Array_1___06": "LFP_6",
    "CECOG_HF_1___07___Array_1___07": "LFP_7",
    "CECOG_HF_1___08___Array_1___08": "LFP_8",
    "CECOG_HF_1___09___Array_2___09": "LFP_9",
    "CECOG_HF_1___10___Array_2___10": "LFP_10",
    "CECOG_HF_1___11___Array_2___11": "LFP_11",
    "CECOG_HF_1___12___Array_2___12": "LFP_12",
    "CECOG_HF_1___13___Array_2___13": "LFP_13",
    "CECOG_HF_1___14___Array_2___14": "LFP_14",
    "CECOG_HF_1___15___Array_2___15": "LFP_15",
    "CECOG_HF_1___16___Array_2___16": "LFP_16",
    "CECOG_HF_2___01___Array_3___01": "ECOG_1",
    "CECOG_HF_2___02___Array_3___02": "ECOG_2",
    "CECOG_HF_2___03___Array_3___03": "ECOG_3",
    "CECOG_HF_2___04___Array_3___04": "ECOG_4",
    "CECOG_HF_2___05___Array_3___05": "EOG_1",
    "CECOG_HF_2___06___Array_3___06": "EOG_2",
    "CECOG_HF_2___07___Array_3___07": "EOG_3",
    "CECOG_HF_2___08___Array_3___08": "EOG_4",
}

console = Console(highlight=False)


def get_valid_mne_channel_types():
    ch_types = [
        "meg",
        "eeg",
        "stim",
        "eog",
        "ecg",
        "emg",
        "ref_meg",
        "misc",
        "resp",
        "chpi",
        "exci",
        "ias",
        "syst",
        "seeg",
        "dipole",
        "gof",
        "bio",
        "ecog",
        "fnirs",
        "dbs",
    ]
    return ch_types


def read_lsl_data(pdata, trg_container="lsl_data_raw"):
    """
    Read all lsl files for a given session and experiment and store as an
    mne.Raw data
    """
    conf = pdata["config"].data

    xdf_files = list(conf["data_root"].rglob("*copydraw*.xdf"))

    raw = load_xdf_and_downsample(xdf_files, trg_sfreq=300)

    pdata.add(raw, name=trg_container, header={"files": xdf_files})

    return pdata


def load_xdf_and_downsample(xdf_files, trg_sfreq=300) -> mne.io.BaseRaw:
    """Load and downsample the xdf data"""

    raws = []
    for f in tqdm(xdf_files, desc="Reading xdf files: "):
        try:
            r = parse_xdf_to_raw_ao_22k(f)
            r.resample(trg_sfreq)

            outdir = f.parents[1].joinpath("processed")
            outdir.mkdir(exist_ok=True)
            r.save(outdir.joinpath(f.stem + "-raw.fif"), overwrite=True)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

        #
        # raws.append(r)

    return mne.concatenate_raws(raws)


def parse_xdf_to_raw_ao_22k(fname, meta_map={}, **kwargs) -> mne.io.Raw:
    """Parse the recorded stream"""

    # NOTE: load_xdf will return float32 numpy array for the brain vision
    # data stream, where mne will work with float64....
    xdf_data, _ = pyxdf.load_xdf(fname)

    info = create_mne_info_for_ao_data(xdf_data)

    # df, time_scaling = xdf_timeseries_to_data_array(xdf_data)
    d = [d for d in xdf_data if d["info"]["name"][0] == "AODataStream"][0]
    data = d["time_series"]

    # Assuming the last sample is not screwed up
    times = d["time_stamps"]
    dt = times[-1] - times[0]
    len(times) / dt

    # scale data to V from uV
    data *= 10**-6

    raw = mne.io.RawArray(data.T, info)

    raw = add_markers(xdf_data, raw)

    return raw


def parse_xdf_to_raw_ao_22k_with_correction(fname, meta_map={}, **kwargs) -> mne.io.Raw:
    """Parse the recorded stream"""

    # NOTE: load_xdf will return float32 numpy array for the brain vision
    # data stream, where mne will work with float64....
    xdf_data, _ = pyxdf.load_xdf(fname)

    info = create_mne_info_for_ao_data(xdf_data)

    # df, time_scaling = xdf_timeseries_to_data_array(xdf_data)
    d = [d for d in xdf_data if d["info"]["name"][0] == "AODataStream"][0]
    data = d["time_series"]

    # Assuming the last sample is not screwed up
    times = d["time_stamps"]
    dt = times[-1] - times[0]
    len(times) / dt

    # scale data to V from uV
    data *= 10**-6

    raw = mne.io.RawArray(data.T, info)

    raw = add_markers(xdf_data, raw)

    return raw


def load_channel_data_from_mat(
    file: Path, pattern: str = ECOG_CHANNELS_PATTERN, load_both: bool = True
) -> mne.io.Raw:
    """
    Load the channel data from Neuro Omega *.mat file. The `pattern`
    parameter can be used to load either only ECoG or LFP or both types of data
    """
    data = loadmat(file, simplify_cells=True)

    if load_both:
        channels = [
            e
            for e in data.keys()
            if re.fullmatch(ECOG_CHANNELS_PATTERN, e)
            or re.fullmatch(LFP_CHANNELS_PATTERN, e)
        ]
        n_ecog = len([e for e in channels if re.fullmatch(ECOG_CHANNELS_PATTERN, e)])
        n_lfp = len([e for e in channels if re.fullmatch(LFP_CHANNELS_PATTERN, e)])

    else:
        channels = [e for e in data.keys() if re.fullmatch(pattern, e)]
        n_ecog = len(channels) if pattern == ECOG_CHANNELS_PATTERN else 0
        n_lfp = len(channels) if pattern == LFP_CHANNELS_PATTERN else 0

    channel_names = [f"LFP_{i + 1}" for i in range(n_lfp)] + [
        f"ECOG_{i + 1}" for i in range(n_ecog)
    ]

    hdr = get_ao_channels_header(data, channels[0])

    info = mne.create_info(
        channel_names,
        hdr["sfreq"],
        ch_types=["dbs"] * n_lfp + ["ecog"] * n_ecog,
    )

    # Unit conversion to uV
    data_matrix = np.asarray([data[c] for c in channels]) / 10**6
    tmp = mne.io.RawArray(data_matrix, info)

    if "CPORT__1" in data:
        vmrk = data["CPORT__1"]
        vmrk = np.array(vmrk[:, vmrk[1, :] != 0], dtype=float)
        # pdb.set_trace()

        # convert to seconds, substract baseline and convert to data rate
        vmrk[0, :] = (
            vmrk[0, :] / (data["CPORT__1_KHz"] * 1000) - hdr["t_start"]
        ) * hdr["sfreq"]

        # to mne structure
        events = np.ones((vmrk.shape[1], 3), dtype=int)
        events[:, 0] = vmrk[0, :]
        events[:, -1] = vmrk[1, :]

    else:
        # -> no markers from parallel port received -> create one at start
        print("=" * 80)
        print(f"Did not find any CPORT__1 in file: \n{file}")
        print("=" * 80)
        events = np.array([[1, 1, 999]], dtype=int)

    annotations = mne.annotations_from_events(events, sfreq=hdr["sfreq"])
    tmp.set_annotations(annotations)
    tmp._filenames = [file]

    return tmp


def get_ao_channels_header(data, prefix):
    """
    Given the dict from the loaded mat file, extract the header for a
    given channel prefix
    """

    d = dict(
        t_start=data[prefix + "_TimeBegin"],
        n_samples=data[prefix].shape[0],
        gain=1e6 * data[prefix + "_Gain"],
        bit_res=data[prefix + "_BitResolution"],
        sfreq=data[prefix + "_KHz"] * 1e3,
    )

    return d


def get_AO_file_data(fname: Path) -> dict:
    """Get the *.mat file provided a *.xdf file path"""
    stem = fname.stem
    pp = fname.parents[1].joinpath("AO")
    files = list(pp.glob(f"{stem}*mat"))

    d = {f: loadmat(f) for f in files}

    return d


def add_markers(xdf_data: dict, raw: mne.io.BaseRaw, sfreq: int = 22_000) -> mne.io.Raw:
    """
    Add markers to the raw EEG data from XDF data.

    Parameters
    ----------
    xdf_data : dict
        The XDF data containing the markers.
    raw : mne.io.BaseRaw
        The raw EEG data object.
    sfreq : int, optional
        The sampling frequency, by default 22,000.

    Returns
    -------
    mne.io.Raw
        The raw EEG data object with markers added.
    """

    markers = create_markers(xdf_data)
    # adjust with info
    marker_t = markers["time_stamps"] - markers["time_stamps"][0]
    marker_idx = (marker_t * sfreq).astype(int)

    # valid markers
    mrk_msk = marker_idx < len(raw.times)
    marker_t = marker_t[mrk_msk]
    marker_idx = marker_idx[mrk_msk]

    # first is index, second is ignored, third is marker id
    mrk = np.array(
        [
            marker_idx,
            np.zeros(len(marker_idx), dtype=int),
            np.asarray(markers["time_series"])[mrk_msk],
        ]
    ).T

    # get time deltas for annots - last dt till last xdf_data point
    tms = raw.times
    dts = np.hstack([np.diff(tms[mrk[:, 0]]), tms[-1] - tms[mrk[-1, 0]]])
    annot = mne.Annotations(
        tms[mrk[:, 0]], dts, np.asarray(markers["time_series"])[mrk_msk]
    )
    raw.set_annotations(annot)
    return raw


def create_markers(xdf_data: list[dict]):
    """Extract the CopyDrawParadigmMarkerStream markers from a loaded xdf_data list"""
    name = "CopyDrawParadigmMarkerStream"
    markers = [d for d in xdf_data if d["info"]["name"][0] == name][0]
    tss = [ts[0] if isinstance(ts, np.ndarray) else ts for ts in markers["time_series"]]
    markers["time_series"] = tss

    mrk_id_map = dict(zip(np.unique(tss), np.arange(len(np.unique(tss)))))

    markers["mrk_ids"] = [mrk_id_map[x] for x in tss]

    return markers


def create_mne_info(xdf_data: list[dict]):
    """Create an mne.Info object from the xdf_data"""
    # get channel info from all streams
    ch_info = {}
    for d in xdf_data:
        name = d["info"]["name"][0]
        channels = d["info"]["desc"][0]
        if channels is not None:
            channel_info = channels["channels"][0]["channel"]
            ch_names = [c["label"][0] for c in channel_info]
            ch_types = [c["type"][0].lower() for c in channel_info]
            ch_types = [
                "misc" if c not in get_valid_mne_channel_types() else c
                for c in ch_types
            ]
        else:
            ch_types, ch_names = ["misc"], [name]

        sfreq = round(d["info"]["effective_srate"], -2)
        if sfreq <= 0:
            sfreq = 22_000
        ch_info[name] = {
            "ch_names": ch_names,
            "ch_types": ch_types,
            "sfreq": sfreq,
        }

    conc_ch_names = [i for v in ch_info.values() for i in v["ch_names"]]
    conc_ch_types = [i for v in ch_info.values() for i in v["ch_types"]]
    max_sfreq = max([v["sfreq"] for v in ch_info.values()])
    if max_sfreq <= 0:
        max_sfreq = 22_000

    info = mne.create_info(conc_ch_names, max_sfreq, ch_types=conc_ch_types)

    return info


def create_mne_info_for_ao_data(xdf_data):
    """Create an mne.Info object from the xdf_data for the Neuro Omega AODataStream"""
    # get channel info from all streams
    ch_info = {}
    for d in xdf_data:
        if d["info"]["name"][0] == "AODataStream":
            name = d["info"]["name"][0]

            sfreq = round(d["info"]["effective_srate"], -2)
            if sfreq <= 0:
                sfreq = 22_000
            nchan = int(d["info"]["channel_count"][0])

            ch_info[name] = {
                "ch_names": [f"Ch_{i}" for i in range(nchan)],
                "ch_types": ["eeg"] * nchan,
                "sfreq": sfreq,
            }

    conc_ch_names = [i for v in ch_info.values() for i in v["ch_names"]]
    conc_ch_types = [i for v in ch_info.values() for i in v["ch_types"]]

    if len(conc_ch_names) == 24:
        console.print("[red]ASSUMING DEFAULT CHANNEL NAMES - 16 LFP + 4 ECOG + 4 EOG")

        conc_ch_names = (
            [f"LFP_{i + 1}" for i in range(16)]
            + [f"ECOG_{i + 1}" for i in range(4)]
            + [f"EOG_{i + 1}" for i in range(4)]
        )
        conc_ch_types = (
            ["dbs" for i in range(16)]
            + ["ecog" for i in range(4)]
            + ["eog" for i in range(4)]
        )

    max_sfreq = max([v["sfreq"] for v in ch_info.values()])
    if max_sfreq <= 0:
        max_sfreq = 22_000

    print(f"max_sfreq: {max_sfreq}")

    info = mne.create_info(conc_ch_names, max_sfreq, ch_types=conc_ch_types)

    return info


def xdf_timeseries_to_data_array(xdf_data, replace_missing_with="ffill"):
    """Transform to a 2D data frame for the use within an mne.Raw object

    Parameters
    ----------
    xdf_data: list
        list of data read from xdf via pyxdf.load_xdf
    replace_missing_with: str ('ffill'|'nan'|'0')
        options on how to deal with missing values or upsample


    Returns
    -------
    df: pandas.DataFrame
        data frame with one column for each recorderd channel. Names are as
        <stream_nbr>_<channel_nbr>
    time_scaling: int
        factor (multiple of 10) to scale the time stamps by to have the smallest
        time distance still being reflected as an integer

    """

    indeces, time_scaling, data_idx = compute_indices(xdf_data)

    df = pd.DataFrame([], index=indeces)

    for i, d in enumerate(xdf_data):
        if i != data_idx:
            idx_map = find_closest_idx(df.index, d["time_stamps"] * time_scaling)

        ts = np.asarray(d["time_series"])
        ts = ts.reshape((ts.shape[0], -1))  # make at least 2d

        for j in range(ts.shape[1]):  # columns per stream
            c_name = f"{i}_{j}"
            if i == data_idx:
                df[c_name] = ts[:, j]
            else:
                df[c_name] = np.nan
                df.loc[idx_map, c_name] = ts[:, j]

    if replace_missing_with == "ffill":
        df = df.ffill()
    elif replace_missing_with == "nan":
        df = df.fillna(np.nan)
    elif replace_missing_with == "0":
        df = df.fillna(0)
    else:
        raise KeyError(
            f"Unknow option for {replace_missing_with=} - valid are"
            " 'ffil', 'nan' or '0'"
        )

    return df, time_scaling


def find_closest_idx(trg_idx, src_idx, return_type="value"):
    """Get a list of the closest matching indeces
    Set return_type to either return the value at target or the index
    """

    # the trg_idx have to be sorted as they represent consecutive timestamps
    # NOTE: This always aligns with the next best timepoint, i.e if
    # trg_idx = [1, 2, 3, 4, 5] then 4.1 -> 5, but 3 -> 3
    idx_where = [np.searchsorted(trg_idx, v, side="left") for v in src_idx]

    if return_type == "value":
        return trg_idx[idx_where]
    elif return_type == "index":
        return idx_where
    else:
        raise KeyError(f"Unknows {return_type=} - use 'value' or 'index'")


def compute_indices(xdf_data):
    """
    Check the distances between recordings, the time ranges and sampling
    frequencies for each data element in xdf_data and create and appropriate
    index set

    Parameters
    ----------
    xdf_data: list
        xdf data read from an lsl recording -> a list of each stream recorded

    Returns
    -------
    indeces: list of int
        integer timing indeces
    scaling: int
        scaling applied to the time stamps for creating integer indeces
    data_idx: int
        index of data coresponding to the indeces selected if > 0. -999 if
        no single stream was used as master but a compound was selected
    """
    # get the time extend covered by each data and the freqs
    tranges = np.array(
        [[d["time_stamps"].min(), d["time_stamps"].max()] for d in xdf_data]
    )
    sfreqs = [round(d["info"]["effective_srate"], -2) for d in xdf_data]

    # get signal which has the minimum time distance between consecutive
    # time stamps
    min_dt = [min(np.diff(d["time_stamps"])) for d in xdf_data]

    max_sfreq = max(sfreqs)
    total_trange = [tranges[:, 0].min(), tranges[:, 1].max()]

    # create indeces by scaling appropriately so that the smallest time step
    # can be represented as a integer step
    # Note: int(x) + int(bool(x % 1)) will always round up to the next int
    # 4.001 -> 5, 4.0 -> 4
    scaling_factor = 10 ** (
        int(np.log10(max_sfreq)) + int(bool(np.log10(max_sfreq) % 1))
    )

    # check if there is one time series which has a range broader than all
    # others, the highest sfreq and smallest dt -> use this to arrange all
    # others along this
    # --> actually the most likely case with a BV recording
    min_dt_idx = np.argmin(min_dt)
    if (
        sfreqs[min_dt_idx] == max_sfreq
        and tranges[min_dt_idx, 0] == min(total_trange)
        and tranges[min_dt_idx, 1] == max(total_trange)
    ):
        indeces = (xdf_data[min_dt_idx]["time_stamps"] * scaling_factor).astype(int)
    else:
        # We need to create a range which suits all, extending the time series
        # with the smallest dt accross the larges range
        indeces = xdf_data[min_dt_idx]["time_stamps"]
        dt = 1 / sfreqs[min_dt_idx]

        # prepend until first index is smaller than first recorded timestamp
        d_pre = (indeces[0] - total_trange[0]) / dt
        prep = [
            indeces[0] - (i + 1) * dt for i in range(int(d_pre) + int(bool(d_pre % 1)))
        ][::-1]

        # append until last index is bigger than last recorded timestamp
        d_post = (total_trange[1] - indeces[-1]) / dt
        postp = [
            indeces[0] - (i + 1) * dt
            for i in range(int(d_post) + int(bool(d_post % 1)))
        ]

        indeces = (np.hstack([prep, indeces, postp]) * scaling_factor).astype(int)

        min_dt_idx = -999

    return indeces, scaling_factor, min_dt_idx


def get_xdf_files(session: str):
    """Parse the DATA_ROOT directory for all copydraw related xdf files"""
    files = list(
        DATA_ROOT.joinpath(f"sub-p001_ses-{session}", "lsl").glob("*copydraw*.xdf")
    )

    return files


def get_AO_data(fname, with_cport_marker: bool = True) -> pd.DataFrame:
    """
    Given an xdf file, load the according *.mat file into a dataframe.
    `mat` files are used for offline analysis, as they do not have missing
    data packages.
    """
    d = get_AO_file_data(fname)

    data = d[list(d.keys())[-1]]

    dm = data["CECOG_HF_2___01___Array_3___01"][0]
    tstart = data["CECOG_HF_2___01___Array_3___01_TimeBegin"][0]
    tm = np.linspace(0, int(len(dm) / 22_000), len(dm))
    df = pd.DataFrame({"time": tm, "data": dm, "src": "AO"})

    # add the actual channel data
    df = df.assign(**{v: data[k][0] for k, v in CHANNEL_MAP.items()})

    if with_cport_marker:
        # find marker to align
        cport_ix = (
            ((data["CPORT__1"][0] / (data["CPORT__1_KHz"] * 1000) - tstart) * 22_000)
            .astype(int)
            .flatten()
        )
        marker = data["CPORT__1"][1]

        # ignore markers == 0, they were used to reset the hardware marker
        df.loc[cport_ix[marker != 0], "marker"] = marker[marker != 0]

    return df


def get_xdf_data(fname) -> pd.DataFrame:
    """Extract the AODataStream and CopyDrawParadigmMarkerStream from the XDF file to a pandas DataFrame"""

    xdf_data, _ = pyxdf.load_xdf(fname, dejitter_timestamps=False)
    streams = [d["info"]["name"][0] for d in xdf_data]
    dx = xdf_data[streams.index("AODataStream")]["time_series"][:, 16]
    tx = xdf_data[streams.index("AODataStream")]["time_stamps"]

    df = pd.DataFrame({"time": tx, "data": dx, "src": "LSL"}).sort_values("time")

    cpm = xdf_data[streams.index("CopyDrawParadigmMarkerStream")]
    mmsk = cpm["time_stamps"] < tx.max()
    ix_marker = [np.argmin(np.abs(df.time - v)) for v in cpm["time_stamps"][mmsk]]
    df.loc[ix_marker, "marker"] = cpm["time_series"][mmsk, 0]

    # markers only if left over (in case the LFP/ECOG data stream aborted)
    dm = pd.DataFrame(
        {"time": cpm["time_stamps"][~mmsk], "marker": cpm["time_series"][~mmsk, 0]}
    ).assign(src="LSL")
    da = pd.concat([df, dm]).reset_index(drop=True)

    tmin = da.time.iloc[0]
    da.time -= tmin

    # also return the time stamps data separately as the alignment might be off
    # if the closest data sample is a stray
    dts = pd.DataFrame({"time": cpm["time_stamps"], "marker": cpm["time_series"][:, 0]})
    dts.time -= tmin

    return da, dts


def get_AO_with_lsl_markers(fname, day: str = "day4") -> pd.DataFrame:
    """Check if the first markers time differences agree, then align to
    the LSL markers to the AO data"""

    dflsl, dts = get_xdf_data(fname)

    if day == "day3":
        dfao = manual_alignment_day3(fname)

    else:
        if "block6_clcopydraw" in str(fname):  # 'clcopydraw' only available for day 4
            # adjust given visual inspection
            dfao = get_AO_data(fname)
            # Time difference between LSL and AO data streams found after
            # visual inspection of the data streams.
            dt = -47.98
            # dadj = dflsl.copy()
            # dadj.time += dt
            # dp = pd.concat([dadj, dfao[["time", "data", "src"]]])
            # plot_window = [13, 15]
            # px.line(
            #     dp[(dp.time < plot_window[-1]) & (dp.time > plot_window[0])],
            #     x="time",
            #     y="data",
            #     color="src",
            # ).show()
            dts2 = dts[dts.time + dt > 0]
            toffset = 0.015
            dts2.time += toffset
            dfao["lsl_marker"] = np.nan
            t0 = dfao[dfao.marker.notna()].time.iloc[0]
            dts2["ao_time"] = (dts2.time - dts2.time.iloc[0]) + t0
            for row in dts2.itertuples():
                if row.ao_time < dfao.time.iloc[-1]:
                    idx = np.argmin(np.abs(dfao.time - row.ao_time))
                    dfao.loc[idx, "lsl_marker"] = row.marker

            # # validate alignment by plotting
            # nplot = 100_000
            # dp1 = dfao[dfao.time >= t0].iloc[:nplot].copy()
            # dp1.time -= dp1.time.iloc[0]
            # dp2 = dflsl[dflsl.time >= dts2.time.iloc[0]].iloc[:nplot].copy()
            # dp2.time -= dp2.time.iloc[0]
            # px.line(pd.concat([dp1, dp2]), x="time", y="data", color="src").show()

        else:
            try:
                dfao = get_AO_data(fname)
                dfao = align_on_markers(dfao, dts)
            except KeyError:
                # load without marker and try aligning on stim artifact
                print(f"No stim channel found for {fname.stem}")
                dfao = get_AO_data(fname, with_cport_marker=False)
                dfao = align_on_stim_artifact(dfao, dflsl, day=day, threshold=2000)

        # nplot = 10_000_000
        # plot_with_marker_lines(
        #     pd.concat([dflsl.iloc[:nplot], dfao.iloc[:nplot]]), lsl_markers=dts
        # )

    return dfao


def manual_alignment_day3(fname):
    """
    Align the markers from LSL with the offline recordings.
    The alignment was checked and parametrizes for each block individually
    """
    dflsl, dts = get_xdf_data(fname)
    dfao = get_AO_data(fname, with_cport_marker=False)

    # plt.subplot(211)
    # plt.plot(dflsl.time, dflsl.data, label="LSL")
    # plt.subplot(212)
    # plt.plot(dfao.time, dfao.data, label="AO")
    # plt.legend()
    # plt.show()

    # Alignment was done either on the first stimulation artifcat, in case
    # stimulation started after the recording, or on the max value in a
    # time window, if stimulation started before the recording / or in the
    # stim OFF case. Alignment was visualized for each alignment.
    if "block1_" in str(fname):
        dfao = align_on_max_value(dfao, dflsl, t_window=[10, 150], fname=fname)
    elif "block2_" in str(fname):
        dfao = align_on_stim_artifact(
            dfao, dflsl, day="day3", threshold=1000, fname=fname
        )
    elif "block3_" in str(fname):
        dfao = align_on_max_value(dfao, dflsl, t_window=[10, 150], fname=fname)
    elif "block4_" in str(fname):
        # stim started before recording -> choose max to align
        dfao = align_on_max_value(
            dfao, dflsl, manual_offset=-0.001, t_window=[0, 150], fname=fname
        )
    elif "block5_" in str(fname):
        dfao = align_on_max_value(dfao, dflsl, t_window=[10, 150], fname=fname)
    elif "block6_" in str(fname):
        # stim started before recording -> choose max to align
        dfao = align_on_max_value(dfao, dflsl, t_window=[50, 150], fname=fname)
    elif "block7_" in str(fname):
        dfao = align_on_max_value(dfao, dflsl, t_window=[10, 150], fname=fname)
    elif "block8_" in str(fname):
        dfao = align_on_stim_artifact(
            dfao, dflsl, day="day3", threshold=1000, fname=fname
        )
    elif "block9_" in str(fname):
        dfao = align_on_max_value(dfao, dflsl, t_window=[10, 150], fname=fname)
    elif "block10_" in str(fname):
        dfao = align_on_stim_artifact(
            dfao, dflsl, day="day3", threshold=1000, fname=fname
        )
    elif "block11_" in str(fname):
        dfao = align_on_max_value(dfao, dflsl, t_window=[10, 150], fname=fname)
    elif "block12_" in str(fname):
        # lsl not complete here, simply use the AO markers
        dfao = get_AO_data(fname)
        dfao["lsl_marker"] = dfao.marker

    return dfao


def align_on_markers(dfao: pd.DataFrame, dts: pd.DataFrame) -> pd.DataFrame:
    """
    Align data from the Neuro Omega (dfao) with the markers LSL (dts) based on the
    time difference of markers. This is done as for hardware reasons, the
    markers in the Neuro Omega recordings might not reflect the correct numerical
    values, or might not be present at all (see align_on_stim_artifact, or max)
    for how we deal with the alignment in case no markers were present.
    """
    dt_lsl = np.diff(dts[dts.marker.notna()].time)
    dt_ao = np.diff(dfao[dfao.marker.notna()].time)

    # allow for 10% deviation
    # Assume that if any deviation, then there are more markers in AO
    ii = None
    for i in range(3):  # check at most the first 3
        for j in range(3):  # check up to the first 3 distances
            if np.abs(dt_ao[i : i + j + 1].sum() - dt_lsl[0]) / dt_ao[0] < 0.1:
                ii = i if ii is None else ii

    if ii is None:
        raise KeyError("No matching marker distances found")

    # align to the first marker selected by `i` above
    dists = dt_lsl.cumsum()
    t0idx = dfao[dfao.marker.notna()].index[ii]
    t0 = dfao.time.iloc[t0idx]
    idx = [t0idx] + [np.searchsorted(dfao.time, t0 + v) for v in dists]
    dfao.loc[idx, "lsl_marker"] = dts[dts.marker.notna()].marker.values

    return dfao


def align_on_stim_artifact(
    dfao: pd.DataFrame,
    dflsl: pd.DataFrame,
    day: str = "day4",
    threshold: float = 2000,
    fname: str = "",
) -> pd.DataFrame:
    """
    Align data from the Neuro Omega (dfao) with the markers LSL (dts) based on the
    occurance of the first stimulation artifact. This was a very accurate method
    if the recording started before stimulation was started, but not applicable
    if stimulation was ongoing already.
    """

    if day == "day4":
        ix_lsl_first_peak = dflsl[dflsl.data > threshold].index[1]
        ix_ao_first_peak = dfao[dfao.data > threshold].index[2]
    else:
        ix_lsl_first_peak = dflsl[(dflsl.data > threshold) & (dflsl.time > 1)].index[0]
        ix_ao_first_peak = dfao[(dfao.data > threshold) & (dfao.time > 1)].index[0]

    # # --- plotting the alignment
    # n = 100_000
    # lsl_first_peak_data = dflsl.iloc[
    #     max(ix_lsl_first_peak - n, 0) : ix_lsl_first_peak + n
    # ]
    # ao_first_peak_data = dfao.iloc[max(ix_ao_first_peak - n, 0) : ix_ao_first_peak + n]
    #
    # data = [lsl_first_peak_data, ao_first_peak_data]
    #
    # # for c in ["LFP_15", "LFP_16", "ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]:
    # #     dw = ao_first_peak_data.copy()
    # #     dw = dw.assign(data=dw[c], src=f"AO_{c}")
    # #     data.append(dw)
    # dt = pd.concat(data)
    # #
    # dt = dt.assign(
    #     auxt=dt.groupby(["src"])["time"].transform(lambda x: np.arange(len(x)))
    # )
    #
    # fig = px.line(dt, x="auxt", y="data", color="src")
    # fig.show()

    dt = dfao.iloc[ix_ao_first_peak]["time"] - dflsl.iloc[ix_lsl_first_peak]["time"]

    # Some additional offset might be required as the alignment on the stim
    # pulse could still be impacted by chunking
    # -> take the time series from LSL after the marker and check in LSL where
    # we have the best overlap
    chk_idx_range = 10_000
    chk_len = 400
    idx_lsl_first_marker_idx = dflsl[dflsl.marker.notna()].index[0]
    t_lsl_first_marker = dflsl.iloc[idx_lsl_first_marker_idx].time
    ao_test_idx = np.searchsorted(dfao.time, t_lsl_first_marker + dt)
    dao_chk = dfao.iloc[
        ao_test_idx - chk_idx_range : ao_test_idx + chk_idx_range
    ].copy()
    dlsl_chk = dflsl[
        idx_lsl_first_marker_idx : idx_lsl_first_marker_idx + chk_len
    ].copy()
    differences = np.asarray(
        [
            dao_chk.iloc[i : i + chk_len].data.values - dlsl_chk.data.values
            for i in range(2 * chk_idx_range - chk_len)
        ]
    )
    idx_shift = np.argmin(np.abs(differences).mean(axis=1)) - chk_idx_range

    # t=  - .0715
    dm = dflsl[dflsl.marker.notna()]
    idx = [np.searchsorted(dfao.time, v + dt + idx_shift / 22_000) for v in dm.time]
    dfao["lsl_marker"] = np.nan
    dfao.loc[idx, "lsl_marker"] = dflsl[dflsl.marker.notna()].marker.values

    # sanety check plot
    # plt.subplot(211)
    # plt.plot(dfao.time, dfao.data)
    # plt.plot(
    #     dfao[dfao.lsl_marker.notna()].time, dfao[dfao.lsl_marker.notna()].data, "r*"
    # )
    #
    # plt.subplot(212)
    # plt.plot(dflsl.time, dflsl.data)
    # plt.plot(dflsl[dflsl.marker.notna()].time, dflsl[dflsl.marker.notna()].data, "r*")
    # plt.show()

    # n = 4000
    # mlsl_idx = dm.index[0]
    # mao_idx = dfao[dfao.lsl_marker.notna()].index[0]
    # dp = pd.concat(
    #     [dflsl.iloc[mlsl_idx - n : mlsl_idx + n], dfao.iloc[mao_idx - n : mao_idx + n]]
    # )
    # dp = dp.assign(
    #     auxt=dp.groupby(["src"])["time"].transform(lambda x: np.arange(len(x)))
    # )
    # fig = px.line(dp, x="auxt", y="data", color="src")
    # fig = fig.update_layout(title=f"{fname}")
    # fig.show()
    #
    return dfao


def align_on_max_value(
    dfao: pd.DataFrame,
    dflsl: pd.DataFrame,
    manual_offset: float = 0,
    nplot: int = 20_000,
    t_window: list[float] = [0, 2],
    fname: str = "",
) -> pd.DataFrame:
    """For blocks without stimulation, we need to align on the max value instead of a stimulation peak"""

    tmsk_ao = (dfao.time > t_window[0]) & (dfao.time < t_window[1])
    tmsk_lsl = (dflsl.time > t_window[0]) & (dflsl.time < t_window[1])
    ix_lsl_first_peak = dflsl[tmsk_lsl][
        dflsl[tmsk_lsl].data == dflsl[tmsk_lsl].data.max()
    ].index[0]
    ix_ao_first_peak = dfao[tmsk_ao][
        dfao[tmsk_ao].data == dfao[tmsk_ao].data.max()
    ].index[0]

    # # --- plotting the alignment
    # n = 100_000
    # lsl_first_peak_data = dflsl.iloc[ix_lsl_first_peak - n : ix_lsl_first_peak + n]
    # ao_first_peak_data = dfao.iloc[ix_ao_first_peak - n : ix_ao_first_peak + n]
    #
    # data = [lsl_first_peak_data, ao_first_peak_data]
    # #
    # # for c in ["LFP_15", "LFP_16", "ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]:
    # #     dw = ao_first_peak_data.copy()
    # #     dw = dw.assign(data=dw[c], src=f"AO_{c}")
    # #     data.append(dw)
    # dt = pd.concat(data)
    #
    # dt = dt.assign(
    #     auxt=dt.groupby(["src"])["time"].transform(lambda x: np.arange(len(x)))
    # )

    # fig = px.line(dt, x="auxt", y="data", color="src")
    # fig.show()

    dt = dfao.iloc[ix_ao_first_peak]["time"] - dflsl.iloc[ix_lsl_first_peak]["time"]
    print(f"Found time difference {dt=}, expected difference ~3.5s.")

    # Some additional offset might be required as the alignment on the stim
    # pulse could still be impacted by chunking
    # -> take the time series from LSL after the marker and check in LSL where
    # we have the best overlap
    chk_idx_range = 10_000
    chk_len = 400
    idx_lsl_first_marker_idx = dflsl[dflsl.marker.notna()].index[0]
    t_lsl_first_marker = dflsl.iloc[idx_lsl_first_marker_idx].time
    ao_test_idx = np.searchsorted(dfao.time, t_lsl_first_marker + dt)
    dao_chk = dfao.iloc[
        ao_test_idx - chk_idx_range : ao_test_idx + chk_idx_range
    ].copy()
    dlsl_chk = dflsl[
        idx_lsl_first_marker_idx : idx_lsl_first_marker_idx + chk_len
    ].copy()
    differences = np.asarray(
        [
            dao_chk.iloc[i : i + chk_len].data.values - dlsl_chk.data.values
            for i in range(2 * chk_idx_range - chk_len)
        ]
    )
    idx_shift = np.argmin(np.abs(differences).mean(axis=1)) - chk_idx_range

    # t=  - .0715
    dm = dflsl[dflsl.marker.notna()]
    idx = [
        np.searchsorted(dfao.time, v + dt + manual_offset + idx_shift / 22_000)
        for v in dm.time
    ]
    dfao["lsl_marker"] = np.nan
    dfao.loc[idx, "lsl_marker"] = dflsl[dflsl.marker.notna()].marker.values

    # sanety check plot
    # plt.subplot(211)
    # plt.plot(dfao.time, dfao.data)
    # plt.plot(
    #     dfao[dfao.lsl_marker.notna()].time, dfao[dfao.lsl_marker.notna()].data, "r*"
    # )
    #
    # plt.subplot(212)
    # plt.plot(dflsl.time, dflsl.data)
    # plt.plot(dflsl[dflsl.marker.notna()].time, dflsl[dflsl.marker.notna()].data, "r*")
    # plt.show()

    mlsl_idx = dm.index[0]
    mao_idx = dfao[dfao.lsl_marker.notna()].index[0]
    dp = pd.concat(
        [
            dflsl.iloc[mlsl_idx - nplot : mlsl_idx + nplot],
            dfao.iloc[mao_idx - nplot : mao_idx + nplot],
        ]
    )
    dp = dp.assign(
        auxt=dp.groupby(["src"])["time"].transform(lambda x: np.arange(len(x)))
    )
    # fig = px.line(dp, x="auxt", y="data", color="src")
    # fig = fig.update_layout(title=f"{fname}")
    # fig.show()
    #
    return dfao


def create_raws_from_mat_and_xdf(
    files: list[str], day: str = "day4"
) -> list[mne.io.Raw]:
    """
    Create raw LFP/ECOG/EOG data objects from *.mat and *.xdf files. Channel data is read
    from *.mat files as these do not suffer from missing data (occured due to
    network/processing overhead). The *.xdf files contain the correct markers
    used to slice the epochs, so data is align based on common artifacts such
    as the stimulation artifact or a matching maximum signal value in the first
    ~150 recordign seconds. All alignment methods are validated by plotting.


    Parameters
    ----------
    files : list[str]
        List of file paths to the *.xdf files. *.mat files will be derived accordingly.
    day : str, optional
        The day identifier, by default "day4".

    Returns
    -------
    list[mne.io.Raw]
        List of raw data objects.
    """

    ao_data = []
    for fname in files:
        print(f"Processing {fname}")
        try:
            ao_data.append(get_AO_with_lsl_markers(fname, day=day))
        except Exception as e:
            print(f"Failed for {fname}: {e}")

    # add meta col
    for i, fname in enumerate(files):
        ao_data[i]["file"] = fname.stem

    chs = [
        c
        for c in ao_data[0].columns
        if c.startswith("LFP") or c.startswith("ECOG") or c.startswith("EOG")
    ]
    info = mne.create_info(
        chs,
        22_000,
        ch_types="eeg",
    )
    raws = []
    for d, fname in zip(ao_data, files):
        d = d.reset_index(drop=True)  # ensure that index is running integer
        # scale data to V from uV
        data = d[chs].to_numpy()
        data_V = data * 10**-6

        raw = mne.io.RawArray(data_V.T, info)

        stem = fname.stem
        pp = fname.parents[1].joinpath("AO")
        files = list(pp.glob(f"{stem}*mat"))

        raw._filenames = [files[0]]

        # first is index, second is ignored, third is marker id
        dm = d[d.lsl_marker.notna()]
        mrk = np.array(
            [
                dm.index,
                np.zeros(len(dm), dtype=int),
                np.asarray(dm["lsl_marker"], dtype=int),
            ]
        ).T

        # get time deltas for annots - last dt till last xdf_data point
        tms = raw.times
        dts = np.hstack([np.diff(tms[mrk[:, 0]]), tms[-1] - tms[mrk[-1, 0]]])
        annot = mne.Annotations(
            tms[mrk[:, 0]], dts, np.asarray(dm["lsl_marker"], dtype=int)
        )
        raw.set_annotations(annot)
        raws.append(raw)

    return raws
