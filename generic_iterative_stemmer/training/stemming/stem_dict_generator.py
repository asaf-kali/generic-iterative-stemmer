import logging
from typing import Dict, Iterable, Optional

import editdistance
from gensim.models import KeyedVectors
from tqdm import tqdm

from ...utils import measure_time

log = logging.getLogger(__name__)

StemDict = Dict[str, str]


class StemDictGenerator:
    def __init__(
        self,
        model: KeyedVectors,
        k: Optional[int] = 10,
        min_grade: Optional[float] = 0.8,
        max_len_diff: Optional[int] = 5,
        max_edit_distance: Optional[int] = 2,
    ):
        self.model = model
        self.k = k
        self.min_grade = min_grade
        self.max_len_diff = max_len_diff
        self.max_edit_distance = max_edit_distance

    def generate_word_stemming(self, word: str) -> dict:
        similarities = self.model.most_similar(word, topn=self.k)
        stem_dict = {}
        for other, grade in similarities:
            if grade < self.min_grade:
                continue
            if word not in other:
                if self.max_edit_distance and editdistance.eval(word, other) > self.max_edit_distance:
                    continue
            if self.max_len_diff and abs(len(word) - len(other)) > self.max_len_diff:
                continue
            stem_dict[other] = word
        return stem_dict

    @measure_time
    def generate_model_stemming(self, words: Iterable[str] = None):
        if words is None:
            words = self.model.key_to_index.keys()
        model_stem_dict = {}
        log.info("Generating stem dict...")
        for word in tqdm(words):
            word_stemming = self.generate_word_stemming(word=word)
            model_stem_dict.update(word_stemming)
        log.info(f"Total {len(model_stem_dict)} stems generated")
        reduced_dict = reduce_stem_dict(stem_dict=model_stem_dict)
        return reduced_dict


def reduce_stem_dict(stem_dict: StemDict) -> StemDict:
    log.debug(f"Reducing stem dict of size {len(stem_dict)}")
    reduced_dict: StemDict = {}
    while len(stem_dict) > 0:
        word_to_reduce = next(iter(stem_dict))
        _reduce_iteration(stem_dict=stem_dict, reduced_dict=reduced_dict, word_to_reduce=word_to_reduce)
    return reduced_dict


def _reduce_iteration(stem_dict: StemDict, reduced_dict: StemDict, word_to_reduce: str):
    representative = stem_dict.pop(word_to_reduce)
    if representative in stem_dict:
        # Meaning representative itself can be reduced
        _reduce_iteration(stem_dict, reduced_dict, word_to_reduce=representative)
    if representative in reduced_dict:
        # Meaning representative points to the most reduced
        reduced = reduced_dict[representative]
    else:
        # Meaning representative is the edge of a chain
        reduced = representative
    if word_to_reduce != reduced:
        reduced_dict[word_to_reduce] = reduced
