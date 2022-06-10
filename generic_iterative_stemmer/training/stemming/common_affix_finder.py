import difflib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, TypeVar

import numpy as np
from pydantic import BaseModel

from tests.utils.settings import Settings

from ...training.stemming import StemDict

log = logging.getLogger(__name__)
T = TypeVar("T")
Histogram = Dict[Any, float]
HistogramByLength = Dict[int, Histogram]


@dataclass(frozen=True)
class Removal:
    index: int
    letter: str


class AffixesByLength(BaseModel):
    prefixes_by_length: Dict[int, Histogram]
    suffixes_by_length: Dict[int, Histogram]


@dataclass(frozen=True)
class Affixes:
    prefixes: List[str]
    suffixes: List[str]


def get_removals(inflection: str, stem: str) -> List[Removal]:
    edits = []
    delta = 0
    for i, diff in enumerate(difflib.ndiff(inflection, stem)):
        sign, letter = diff[0], diff[-1]
        if sign == " ":
            continue
        elif sign == "-":
            edits.append(Removal(i - delta, letter))
        elif sign == "+":
            delta += 1
    return edits


def identify_affixes(inflection: str, stem: str) -> Affixes:
    removals = get_removals(inflection, stem)
    prefix, prefixes = "", []
    for i, removal in enumerate(removals):
        if removal.index != i:
            break
        prefix += removal.letter
        prefixes.append(prefix)
    suffix, suffixes = "", []
    for i, removal in enumerate(reversed(removals)):
        if removal.index != len(inflection) - i - 1:
            break
        suffix = removal.letter + suffix
        suffixes.append(suffix)
    return Affixes(prefixes, suffixes)


class CommonAffixesFinder:
    def __init__(self, stem_dict: StemDict, max_prefix_length: int = 2, max_suffix_length: int = 2):
        self.stem_dict = stem_dict
        self.max_prefix_length = max_prefix_length
        self.max_suffix_length = max_suffix_length

    def find_common_affixes(self) -> AffixesByLength:
        prefixes_by_length, suffixes_by_length = defaultdict(dict), defaultdict(dict)  # type: ignore
        for inflection, stem in self.stem_dict.items():
            self._count_affixes(
                inflection=inflection,
                stem=stem,
                prefixes_by_length=prefixes_by_length,
                suffixes_by_length=suffixes_by_length,
            )
        common_affixes = _get_most_common_affixes(prefixes_by_length, suffixes_by_length)
        return common_affixes

        # prefixes = [inflection[:prefix_length] for inflection, stem in self.stem_dict.items()]
        # hist2 = calculate_histogram(prefixes)
        # plot_histogram(hist2, "Prefixes", "Prefix", "Frequency")
        # return top

    def _count_affixes(
        self, inflection: str, stem: str, prefixes_by_length: HistogramByLength, suffixes_by_length: HistogramByLength
    ):
        affixes = identify_affixes(inflection, stem)
        for prefix in affixes.prefixes:
            if len(prefix) > self.max_prefix_length:
                continue
            histogram = prefixes_by_length[len(prefix)]
            histogram[prefix] = histogram.get(prefix, 0) + 1
        for suffix in affixes.suffixes:
            if len(suffix) > self.max_suffix_length:
                continue
            histogram = suffixes_by_length[len(suffix)]
            histogram[suffix] = histogram.get(suffix, 0) + 1


def _get_most_common_affixes(
    prefixes_by_length: HistogramByLength, suffixes_by_length: HistogramByLength
) -> AffixesByLength:
    common_affixes = AffixesByLength(prefixes_by_length={}, suffixes_by_length={})
    for length, histogram in prefixes_by_length.items():
        common_histogram = _filter_common_affixes_from_histogram(histogram, "prefix", length)
        common_affixes.prefixes_by_length[length] = common_histogram
    for length, histogram in suffixes_by_length.items():
        common_histogram = _filter_common_affixes_from_histogram(histogram, "suffix", length)
        common_affixes.suffixes_by_length[length] = common_histogram
    return common_affixes


def _filter_common_affixes_from_histogram(histogram: Histogram, affix_type: str, length: int) -> Histogram:
    histogram.pop("", None)
    histogram = dict(sorted(histogram.items(), key=lambda x: x[1], reverse=True))
    values = np.array(list(histogram.values()))
    total, mean, std = values.sum(), values.mean(), values.std()
    common = {key: round(value / total, 4) for key, value in histogram.items() if value - mean >= std / 2}
    affix_type = affix_type.lower()
    log.info(f"{len(common)} common {affix_type}es of size {length} found out of {len(histogram)} total")
    plot_histogram(histogram, f"Removed {affix_type}es of size {length}", affix_type.title(), "Frequency")
    return common


def plot_histogram(histogram: Histogram, title: str, x_label: str, y_label: str):
    if not Settings.is_debug:
        return

    import matplotlib.pyplot as plt

    histogram = dict(sorted(histogram.items(), key=lambda x: x[1], reverse=True)[:20])

    x = np.arange(len(histogram))
    y = [histogram[k] for k in histogram.keys()]
    plt.bar(x, y)
    plt.xticks(x, [k for k in histogram.keys()])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# def calculate_histogram(elements: Collection[T]) -> Histogram:
#     histogram: Dict[T, float] = {element: 0 for element in elements}
#     for element in elements:
#         histogram[element] += 1
#     histogram = dict(sorted(histogram.items(), key=lambda x: x[1], reverse=True))
#     histogram = {k: v / len(elements) for k, v in histogram.items()}
#     return histogram
