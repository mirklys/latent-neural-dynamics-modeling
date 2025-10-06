import numpy as np
from scipy.interpolate import interp1d

def interpolate(coordinates: list, original_length_ts: int) -> list:
    if len(coordinates) == 0 or original_length_ts <= 0:
        return []

    original_indices = np.linspace(0, 1, num=len(coordinates))
    target_indices = np.linspace(0, 1, num=original_length_ts)

    interpolator = interp1d(original_indices, coordinates, kind="linear", fill_value="extrapolate")
    interpolated_coordinates = interpolator(target_indices)

    return interpolated_coordinates.tolist()