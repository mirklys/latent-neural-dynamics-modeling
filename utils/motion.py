import numpy as np
from scipy.interpolate import interp1d
from utils.miscellaneous import contains_nulls


def interpolate(coordinates: list, original_length_ts: int) -> list:
    if len(coordinates) == 0 or original_length_ts <= 0:
        return []

    original_indices = np.linspace(0, 1, num=len(coordinates))
    target_indices = np.linspace(0, 1, num=original_length_ts)

    interpolator = interp1d(
        original_indices, coordinates, kind="linear", fill_value="extrapolate"
    )
    interpolated_coordinates = interpolator(target_indices)

    return interpolated_coordinates.tolist()


def tracing_speed(
    x: list,
    y: list,
    time: list,
    moving_avg_window_ms: int = 50,
) -> list:

    if contains_nulls(x) or contains_nulls(y) or contains_nulls(time):
        return None
    
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(time)
    dt[dt == 0] = np.finfo(float).eps

    instantaneous_speed = np.sqrt(dx**2 + dy**2) / dt
    instantaneous_speed = np.insert(instantaneous_speed, 0, 0)

    window_size_samples = int((moving_avg_window_ms / 1000) * 1000)
    smoothed_speed = np.convolve(
        instantaneous_speed,
        np.ones(window_size_samples) / window_size_samples,
        mode="same",
    )
    return smoothed_speed.tolist()
