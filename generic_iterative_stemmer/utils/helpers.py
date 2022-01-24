import os


def sort_dict_by_values(d: dict) -> dict:
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


def remove_file_exit_ok(file_path: str) -> bool:
    try:
        os.remove(file_path)
        return True
    except:  # noqa
        return False
