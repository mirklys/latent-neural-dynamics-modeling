import re
from logging import getLogger
from pathlib import Path

import numpy as np

from behavior.loading import deriv_and_norm, derive_stim, load_copydraw_record_yaml

logger = getLogger(__name__)


def process_trial(trial_file: Path, use_longest_only: bool = True) -> dict:
    """Actual trial post processing takes place here."""

    logger.info(f"Loading trial data for {trial_file=}")
    res = load_copydraw_record_yaml(trial_file)

    # Overwrite ix block as saving for the closed loop session did not
    # increment the block idx in the yaml files (visible in the filesystem though)
    # only files with 'STIM_UNKNOWS', i.e. stim not set for paradigm, do show
    # the problem.
    if "STIM_UNKNOWN" in trial_file.stem:
        res["ix_block"] = int(re.search(r"block_(\d+)_", str(trial_file)).group(1))

    stim = derive_stim(trial_file)
    if str(trial_file.parent).endswith("_cl"):
        stim += "_cl"
    res["stim"] = stim

    # scale the template to how it would be on the screen in real pixels
    # as the trace_let is recorded in screen pixel coords
    temp = res["template_pix"] * res["template_scaling"]
    scaled_template = temp - (
        res["template_pos"] / res["scaling_matrix"][0, 0] / res["template_scaling"]
    )

    res["scaled_template"] = scaled_template

    # Note: for training the model for first closed loop, I just used the complete traces
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
    scores = compute_simple_score(traceLet, template, trialTime)

    return {**res, **scores}


def compute_simple_score(traceLet, template, trialTime):
    """Copydraw relevant scores without DTW"""
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

    return trial_results


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
