import logging
import os

log = logging.getLogger(__name__)

DATA_FOLDER = os.getenv("LANGUAGE_DATA_FOLDER") or "~/.cache/language_data"


def get_path(*path: str) -> str:
    return os.path.expanduser(os.path.join(DATA_FOLDER, *path))
