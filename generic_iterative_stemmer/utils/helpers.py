def sort_dict_by_values(d: dict) -> dict:
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
