import logging
from typing import Dict, Iterable, Optional

import editdistance
from gensim.models import KeyedVectors
from tqdm import tqdm

from ...utils import measure_time
from ...utils.async_task_manager import AsyncTaskManager

log = logging.getLogger(__name__)

StemDict = Dict[str, str]


class StemDictGenerator:
    def __init__(
        self,
        model: KeyedVectors,
        k: Optional[int] = 15,
        min_cosine_similarity: Optional[float] = 0.65,
        max_len_diff: Optional[int] = 3,
        max_edit_distance: Optional[int] = 1,
    ):
        self.model = model
        self.k = k
        self.min_cosine_similarity = min_cosine_similarity
        self.max_len_diff = max_len_diff
        self.max_edit_distance = max_edit_distance

    def find_word_inflections(self, word: str) -> StemDict:
        """
        Find which other words in the vocabulary can be stemmed down to this word.
        """
        similarities = self.model.most_similar(word, topn=self.k)
        stem_dict = {}
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
            w2i = self.model.key_to_index
            if w2i[candidate] < w2i[word]:
                continue
            stem_dict[candidate] = word
        return stem_dict

    @measure_time
    def generate_model_stemming(self, vocabulary: Iterable[str] = None):
        if vocabulary is None:
            vocabulary = self.model.key_to_index.keys()
        log.info("Generating stem dict for words...")
        model_stem_dict = {}
        task_manager = AsyncTaskManager(workers_amount=5)
        log.debug("Appending stemming tasks...")
        for word in vocabulary:
            task_manager.add_task(self.find_word_inflections, args=(word,))
        log.debug("Collecting stemming results...")
        for result in tqdm(task_manager, total=task_manager.total_task_count):
            model_stem_dict.update(result)
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
