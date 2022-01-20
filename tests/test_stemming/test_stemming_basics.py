from generic_iterative_stemmer.training.stemming import reduce_stem_dict


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


def test_reduce_stem_dict_handles_cycles():
    stem_dict = {
        "a": "x",
        "b": "y",
        "y": "x",
        "x": "y",
    }
    option_1 = {
        "a": "x",
        "b": "x",
        "y": "x",
    }
    option_2 = {
        "a": "y",
        "b": "y",
        "x": "y",
    }

    actual = reduce_stem_dict(stem_dict)
    assert actual == option_1 or actual == option_2
