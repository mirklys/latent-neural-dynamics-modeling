from pathlib import Path
import json


def read_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        contents = json.load(f)
    return contents


def list_files(folder_path: Path) -> list:

    return [str(item) for item in folder_path.iterdir()]
