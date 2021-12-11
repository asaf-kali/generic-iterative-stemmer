from model_trainer.inflections import reduce_inflections_dict


def test_reduce_inflections():
    inflections = {
        "a": "x",
        "x": "y",
        "b": "x",
        "א": "ב",
        "c": "x",
        "y": "z",
        "d": "a",
    }
    expected = {
        "a": "z",
        "b": "z",
        "c": "z",
        "d": "z",
        "x": "z",
        "y": "z",
        "א": "ב",
    }
    reduced = reduce_inflections_dict(inflections)
    assert reduced == expected
