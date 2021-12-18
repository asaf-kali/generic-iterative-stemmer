from generic_iterative_stemmer.training.stemming import reduce_stem_dict
from generic_iterative_stemmer.training.stemming.stemming_trainer import stem_sentence


def test_stem_sentence():
    sentence = "a b c d e f g"
    stem_dict = {"a": "x"}
    stemmed = stem_sentence(sentence=sentence, stem_dict=stem_dict)
    expected = "x b c d e f g"
    assert stemmed == expected


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
