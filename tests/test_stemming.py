from model_trainer.stemming import reduce_stem_dict


def test_reduce_empty_stem_dict():
    assert reduce_stem_dict({}) == {}


def test_reduce_simple_stem_dict():
    assert reduce_stem_dict({"x": "y"}) == {"x": "y"}


def test_reduce_complex_stem_dict():
    stem_dict = {
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
    reduced = reduce_stem_dict(stem_dict)
    assert reduced == expected
