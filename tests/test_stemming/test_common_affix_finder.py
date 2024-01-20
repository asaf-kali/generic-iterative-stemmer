import logging

from generic_iterative_stemmer.training.stemming import StemDict
from generic_iterative_stemmer.training.stemming.common_affix_finder import (
    Affixes,
    CommonAffixesFinder,
    Removal,
    get_removals,
    identify_affixes,
)

log = logging.getLogger(__name__)


def test_find_common_affixes(small_stem_dict: StemDict):
    common_affixes_finder = CommonAffixesFinder(stem_dict=small_stem_dict)
    result = common_affixes_finder.find_common_affixes()
    as_dict = result.dict()
    log.info("Result", extra={"result": as_dict})
    assert as_dict == {
        "prefixes_by_length": {
            1: {"ו": 0.3155, "ה": 0.1724, "ב": 0.1013, "ש": 0.0927, "מ": 0.0845},
            2: {
                "וה": 0.156,
                "וב": 0.076,
                "כש": 0.0326,
                "ומ": 0.0312,
                "שב": 0.0272,
                "ול": 0.024,
                "המ": 0.0229,
                "מה": 0.0225,
                "שה": 0.0177,
                "קו": 0.0141,
                "וי": 0.012,
                "שי": 0.0116,
                "או": 0.0109,
                "שמ": 0.0105,
                "מו": 0.0099,
                "אל": 0.0091,
                "הא": 0.008,
                "וא": 0.0078,
                "אי": 0.0074,
                "דו": 0.0074,
            },
        },
        "suffixes_by_length": {
            1: {"מ": 0.1619, "ה": 0.1603, "ו": 0.1413, "י": 0.1225, "ת": 0.1151, "נ": 0.078},
            2: {
                "ימ": 0.209,
                "ות": 0.0587,
                "ונ": 0.0413,
                "ית": 0.0403,
                "יה": 0.0373,
                "ינ": 0.0237,
                "יו": 0.0233,
                "המ": 0.0207,
                "נו": 0.013,
                "תו": 0.0113,
                "ני": 0.0113,
                "יס": 0.011,
            },
        },
    }


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
