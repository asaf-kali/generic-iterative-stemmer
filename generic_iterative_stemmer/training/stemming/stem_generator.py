import logging
from typing import Dict, Iterable

from gensim.models import KeyedVectors
from tqdm import tqdm

from ...utils.async_task_manager import AsyncTaskManager

log = logging.getLogger(__name__)

StemDict = Dict[str, str]


class StemGenerator:
    def __init__(self, workers_amount: int = 5):
        self.workers_amount = workers_amount

    def find_word_inflections(self, model: KeyedVectors, word: str) -> StemDict:
        """
        Find which other words in the vocabulary can be stemmed down to this word.
        """
        raise NotImplementedError()

    @property
    def params(self) -> dict:
        """
        Returns the parameters this StemGenerator is working with (for statistics purposes).
        """
        return {}

    def generate_stemming_dict(self, model: KeyedVectors, vocabulary: Iterable[str]) -> StemDict:
        log.info("Generating stem dict for words...")
        model_stem_dict = {}
        with AsyncTaskManager(workers_amount=self.workers_amount) as task_manager:
            log.debug("Appending stemming tasks...")
            for word in tqdm(vocabulary, desc="Add stemming tasks"):
                task_manager.add_task(self.find_word_inflections, args=(model, word))
            log.debug("Collecting stemming results...")
            for result in tqdm(task_manager, total=task_manager.total_task_count, desc="Generate stem dict"):
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
