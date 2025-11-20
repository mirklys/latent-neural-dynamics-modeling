from typing import List, Tuple, Any, Union
import numpy as np


def _pearson_list_2d(y_true_2d: np.ndarray, y_pred_2d: np.ndarray) -> List[float]:
    r_list: List[float] = []

    if y_true_2d is None or y_pred_2d is None:
        return r_list
    if y_true_2d.ndim != 2 or y_pred_2d.ndim != 2:
        raise ValueError(
            "Expect 2D arrays shaped (time, channels) for single-trial computation."
        )
    if (
        y_true_2d.shape[0] != y_pred_2d.shape[0]
        or y_true_2d.shape[1] != y_pred_2d.shape[1]
    ):
        raise ValueError(
            "y_true and y_pred must have the same shape for correlation computation."
        )

    for c in range(y_true_2d.shape[1]):
        t = y_true_2d[:, c]
        p = y_pred_2d[:, c]
        if np.std(t) < 1e-12 or np.std(p) < 1e-12:
            r = np.nan
        else:
            r = float(np.corrcoef(t, p)[0, 1])
        r_list.append(r)
    return r_list


def pearson_r_per_channel(
    y_true: Union[np.ndarray, List[np.ndarray]],
    y_pred: Union[np.ndarray, List[np.ndarray]],
) -> Tuple[List[Any], float]:

    trials_true: List[np.ndarray]
    trials_pred: List[np.ndarray]

    if isinstance(y_true, list) and isinstance(y_pred, list):
        trials_true = y_true
        trials_pred = y_pred
    elif isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        if y_true.ndim == 2 and y_pred.ndim == 2:
            r_list = _pearson_list_2d(y_true, y_pred)
            valid = [r for r in r_list if not (r is None or np.isnan(r))]
            r_mean = float(np.mean(valid)) if len(valid) > 0 else np.nan
            return r_list, r_mean
        elif y_true.ndim == 3 and y_pred.ndim == 3:
            trials_true = [y_true[i] for i in range(y_true.shape[0])]
            trials_pred = [y_pred[i] for i in range(y_pred.shape[0])]
        else:
            raise ValueError("Unsupported input shapes for pearson_r_per_channel.")
    else:
        raise ValueError(
            "y_true and y_pred types must match (both list or both ndarray)."
        )

    per_trial: List[List[float]] = []
    all_valid: List[float] = []
    n_trials = min(len(trials_true), len(trials_pred))
    for i in range(n_trials):
        yt = trials_true[i]
        yp = trials_pred[i]
        r_list = _pearson_list_2d(yt, yp)
        per_trial.append(r_list)
        all_valid.extend([r for r in r_list if not (r is None or np.isnan(r))])

    overall_mean = float(np.mean(all_valid)) if len(all_valid) > 0 else np.nan
    return per_trial, overall_mean
