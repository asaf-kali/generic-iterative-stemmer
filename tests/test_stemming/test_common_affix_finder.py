from generic_iterative_stemmer.training.stemming import StemDict
from generic_iterative_stemmer.training.stemming.common_affix_finder import (
    Affixes,
    CommonAffixesFinder,
    Removal,
    get_removals,
    identify_affixes,
)
from generic_iterative_stemmer.utils import get_logger

log = get_logger(__name__)


def test_find_common_affixes(small_stem_dict: StemDict):
    common_affixes_finder = CommonAffixesFinder(stem_dict=small_stem_dict)
    result = common_affixes_finder.find_common_affixes()
    assert result.to_dict() == {
        "prefixes_by_length": {
            1: {"ו": 0.3113, "ה": 0.1744, "ב": 0.1032, "ש": 0.0912, "מ": 0.0871},
            2: {
                "וה": 0.1523,
                "וב": 0.0772,
                "ומ": 0.0347,
                "כש": 0.0285,
                "שב": 0.027,
                "מה": 0.0248,
                "המ": 0.0227,
                "ול": 0.0219,
                "שה": 0.0151,
                "שי": 0.0132,
                "או": 0.0114,
                "קו": 0.0114,
                "וי": 0.0105,
                "אל": 0.0099,
                "מו": 0.0095,
                "בו": 0.0095,
                "שמ": 0.0093,
                "אי": 0.0091,
                "וש": 0.0085,
                "וא": 0.0082,
                "בר": 0.0076,
            },
        },
        "suffixes_by_length": {
            1: {"ה": 0.1619, "מ": 0.1557, "ו": 0.1455, "י": 0.1218, "ת": 0.1144, "נ": 0.0851},
            2: {
                "ימ": 0.2015,
                "ות": 0.0702,
                "ונ": 0.0441,
                "יה": 0.0404,
                "ית": 0.0373,
                "יו": 0.0319,
                "ינ": 0.021,
                "המ": 0.0197,
                "נו": 0.0142,
                "ני": 0.0136,
                "תו": 0.0136,
                "לי": 0.0126,
                "קה": 0.0109,
                "יק": 0.0102,
                "קי": 0.0098,
            },
        },
    }
    log.info("result", extra={"result": result.to_dict()})


def test_identify_affixes():
    assert identify_affixes("abc", "abc") == Affixes(prefixes=[], suffixes=[])
    assert identify_affixes("abc", "ac") == Affixes(prefixes=[], suffixes=[])
    assert identify_affixes("abcde", "bcd") == Affixes(prefixes=["a"], suffixes=["e"])
    assert identify_affixes("abcde", "bfd") == Affixes(prefixes=["a"], suffixes=["e"])
    assert identify_affixes("abcde", "cf") == Affixes(prefixes=["a", "ab"], suffixes=["e", "de"])
    assert identify_affixes("כשדלתותיה", "דלת") == Affixes(prefixes=["כ", "כש"], suffixes=["ה", "יה", "תיה", "ותיה"])


def test_get_removals():
    assert get_removals("a", "b") == [
        Removal(index=0, letter="a"),
    ]
    assert get_removals("abxcd", "x") == [
        Removal(index=0, letter="a"),
        Removal(index=1, letter="b"),
        Removal(index=3, letter="c"),
        Removal(index=4, letter="d"),
    ]
    assert get_removals("abxcd", "xyz") == [
        Removal(index=0, letter="a"),
        Removal(index=1, letter="b"),
        Removal(index=3, letter="c"),
        Removal(index=4, letter="d"),
    ]
