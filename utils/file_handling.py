from pathlib import Path


def list_files(folder_path: Path) -> list:
    
    return [str(item) for item in folder_path.iterdir()]
