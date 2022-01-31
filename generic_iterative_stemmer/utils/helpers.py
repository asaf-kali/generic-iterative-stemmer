import dataclasses
import os
from dataclasses import dataclass


def sort_dict_by_values(d: dict) -> dict:
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


def remove_file_exit_ok(file_path: str) -> bool:
    try:
        os.remove(file_path)
        return True
    except:  # noqa
        return False


@dataclass
class Serializable:
    def __dict__(self):
        return self.to_dict()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
