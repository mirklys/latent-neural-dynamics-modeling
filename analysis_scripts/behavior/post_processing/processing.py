from logging import getLogger
from pathlib import Path

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np

from behavior.behavioral_scores import dtw_py

logger = getLogger("CopyDraw post processing")


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
    try:
        dtw_res = dtw_py.dtw_features(traceLet, template)
    except ValueError as e:
        print(f"DTW failed: {e}")
        dtw_res = {"dt": -1}  # indicating a failure
        return {**trial_results, **dtw_res}

    trial_results = {**trial_results, **dtw_res}

    # misc
    trial_results["dist_t"] = _w_to_dist_t(
        trial_results["w"].astype(int),
        trial_results["pos_t"],
        template,
        trial_results["pathlen"],
    )
    trial_results["dist_t_sub"] = bn.move_mean(trial_results["dist_t"], 5, min_count=5)[
        ::3
    ]
    trial_results["dist_sub"] = np.nanmean(trial_results["dist_t_sub"])

    # normalize distance dt by length of copied template (in samples)
    trial_results["dt_norm"] = trial_results["dt_l"] / (trial_results["pathlen"] + 1)

    # get length of copied part of the template (in samples)
    trial_results["len"] = (trial_results["pathlen"] + 1) / len(template)

    trial_results = calculate_directional_features(trial_results)

    return trial_results


def calculate_directional_features(tres: dict) -> dict:
    """
    Segment speed, acceleration, jerk and distance in 8 bins as was done in
    Sebastians paper: https://ieeexplore.ieee.org/document/8839739
    """

    # Note that for the distance, we can only take the first element of
    # an uninterupted trace. Thus the 'dist_t' vector might be much smaller
    # than e.g. the speed_t.
    nmax = len(tres["dist_t_sub"])

    s = tres["speed_t_sub"]
    # do not consider entries where speed is 0
    idx_to_deg = np.arctan2(s[:, 1], s[:, 0]) * 180 / np.pi
    # convert to positive angles
    idx_to_deg = [ang if ang >= 0 else 360 + ang for ang in idx_to_deg]
    bins = [i * 360 / 8 for i in range(8)]
    idx_to_bin = np.digitize(idx_to_deg, bins)

    for bini in np.unique(idx_to_bin):
        msk = (idx_to_bin == bini) & (s[:, 0] != 0) & (s[:, 1] != 0)
        tres[f"speed_sub_bin_{bini}"] = np.nanmean(tres["speed_t_sub"][msk, :][:nmax])
        tres[f"accel_sub_bin_{bini}"] = np.nanmean(
            tres["accel_t_sub"][msk[:-1], :][:nmax]
        )
        tres[f"jerk_sub_bin_{bini}"] = np.nanmean(
            tres["jerk_t_sub"][msk[:-2], :][:nmax]
        )
        tres[f"dist_sub_bin_{bini}"] = np.nanmean(tres["dist_t_sub"][msk[:nmax]])

    return tres


def _w_to_dist_t(w, trace, template, pathlen, template_idx_in_w: int = 0):
    """This is a copy of how dist_t is computed in matlab.

    with the added feature of informing it as to whether w is indexed the other way ie [trace_idxs, template_idxs]   # noqa
    """

    tmp1 = template[w[:pathlen, template_idx_in_w], :]
    tmp2 = trace[w[:pathlen, int(not template_idx_in_w)]]
    dist_t = np.sqrt(np.sum((tmp1 - tmp2) ** 2, axis=1))
    return dist_t


def add_scaled_trace(res: dict) -> dict:
    # scale the template to how it would be on the screen in real pixels
    # as the trace_let is recorded in screen pixel coords
    temp = res["template_pix"] * res["template_scaling"]
    scaled_template = temp - (
        res["template_pos"] / res["scaling_matrix"][0, 0] / res["template_scaling"]
    )

    res["scaled_template"] = scaled_template

    return res


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
