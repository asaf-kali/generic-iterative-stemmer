import os
from datetime import timedelta
from time import time


def sort_dict_by_values(d: dict) -> dict:
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


def remove_file_exit_ok(file_path: str) -> bool:
    try:
        os.remove(file_path)
        return True
    except:  # noqa
        return False


class MeasureTime:
    def __init__(self):
        self.start = self.end = 0

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()

    @property
    def delta(self) -> float:
        return round(self.duration.total_seconds(), 3)

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.end - self.start)
