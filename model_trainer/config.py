import os

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH", "./data")


def get_data(*path: str) -> str:
    return os.path.join(DATA_FOLDER, *path)
