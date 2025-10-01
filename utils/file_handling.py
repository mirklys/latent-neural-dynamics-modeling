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


def list_files(folder_path: Path) -> list[str]:
    return [str(item) for item in folder_path.iterdir()]


def get_child_subchilds_tuples(
    parent_folder: Path, full_paths: bool = False
) -> list[tuple[str, ...]]:
    def recurse(folder: Path) -> list[tuple[str, ...]]:
        tuples = []
        for child in folder.iterdir():
            if child.is_dir():
                parent_part = str(folder) if full_paths else folder.name
                child_part = str(child) if full_paths else child.name
                tuples.append((parent_part, child_part))
                tuples.extend(recurse(child))
        return tuples

    all_tuples = []
    for child in parent_folder.iterdir():
        if child.is_dir():
            all_tuples.extend(recurse(child))
    return all_tuples
