from pathlib import Path
import json
from scipy.io import loadmat


def read_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        contents = json.load(f)
    return contents


def load_mat_into_dict(mat_file: str) -> dict:
    data_dict = loadmat(mat_file)
    return data_dict


def list_files(folder_path: Path) -> list:

    return [str(item) for item in folder_path.iterdir()]
