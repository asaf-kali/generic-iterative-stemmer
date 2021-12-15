import logging
from typing import Dict, Iterable

from gensim.models import KeyedVectors
from tqdm import tqdm

from utils.logging import measure_time

log = logging.getLogger(__name__)

StemDict = Dict[str, str]


class StemDictGenerator:
    def __init__(self, model: KeyedVectors, k: int = 50, min_grade: float = 0.75, max_len_diff: int = 5):
        self.model = model
        self.k = k
        self.min_grade = min_grade
        self.max_len_diff = max_len_diff

    def generate_word_stemming(self, word: str) -> dict:
        similarities = self.model.most_similar(word, topn=self.k)
        stem_dict = {}
        for other, grade in similarities:
            if word not in other:
                continue
            if grade < self.min_grade:
                continue
            if abs(len(word) - len(other)) > self.max_len_diff:
                continue
            stem_dict[other] = word
        return stem_dict

    @measure_time
    def generate_model_stemming(self, words: Iterable[str] = None):
        if words is None:
            words = self.model.key_to_index.keys()
        model_stem_dict = {}
        log.info("Generating stemming...")
        for word in tqdm(words):
            word_stemming = self.generate_word_stemming(word=word)
            model_stem_dict.update(word_stemming)
        log.info(f"Total {len(model_stem_dict)} stemming generated")
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
        # Meaning the representative itself can be reduced
        _reduce_iteration(stem_dict, reduced_dict, word_to_reduce=representative)
    if representative in reduced_dict:
        # Meaning we know representative is reduced
        reduced_dict[word_to_reduce] = reduced_dict[representative]
    else:
        # Meaning this is the edge of a chain
        reduced_dict[word_to_reduce] = representative
