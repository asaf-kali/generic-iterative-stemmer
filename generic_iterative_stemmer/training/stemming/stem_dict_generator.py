import logging
from typing import Dict, Iterable, List, Optional

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
        k: Optional[int] = 20,
        min_cosine_similarity: Optional[float] = 0.8,
        max_len_diff: Optional[int] = 3,
        max_edit_distance: Optional[int] = 1,
    ):
        self.model = model
        self.k = k
        self.min_cosine_similarity = min_cosine_similarity
        self.max_len_diff = max_len_diff
        self.max_edit_distance = max_edit_distance

    def find_word_inflections(self, word: str) -> List[str]:
        """
        Find which other words in the vocabulary can be stemmed down to this word.
        """
        similarities = self.model.most_similar(word, topn=self.k)
        inflections = []
        for candidate, grade in similarities:
            if grade < self.min_cosine_similarity:
                continue
            if len(candidate) < len(word):
                continue
            if word not in candidate:
                if self.max_edit_distance is None:
                    continue
                edit_distance = editdistance.eval(word, candidate)
                if edit_distance > self.max_edit_distance:
                    continue
            if self.max_len_diff and abs(len(word) - len(candidate)) > self.max_len_diff:
                continue
            # TODO: check which is more common (model.key_to_index) and only replace if word is more common?
            inflections.append(candidate)
        return inflections

    @measure_time
    def generate_model_stemming(self, vocabulary: Iterable[str] = None):
        if vocabulary is None:
            vocabulary = self.model.key_to_index.keys()
        model_stem_dict = {}
        log.info("Generating stem dict...")
        for word in tqdm(vocabulary):
            inflections = self.find_word_inflections(word=word)
            for inflection in inflections:
                model_stem_dict[inflection] = word
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
