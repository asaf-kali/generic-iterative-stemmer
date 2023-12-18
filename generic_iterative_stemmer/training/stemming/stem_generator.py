import logging
from typing import Dict, Iterable, Mapping

from gensim.models import KeyedVectors
from the_spymaster_util.async_task_manager import AsyncTaskManager
from tqdm import tqdm

from generic_iterative_stemmer.helpers import sort_dict_by_values

log = logging.getLogger(__name__)

StemDict = Dict[str, str]


class StemGenerator:
    def __init__(self, workers_amount: int = 5):
        self.workers_amount = workers_amount
        self.model: KeyedVectors = None  # type: ignore
        self.word_to_index: Mapping = None  # type: ignore
        self.vocab_size = 0

    def find_word_inflections(self, word: str) -> StemDict:
        """
        Find which other words in the vocabulary can be stemmed down to this word.
        """
        raise NotImplementedError()

    def set_model(self, model: KeyedVectors):
        self.model = model
        self.word_to_index = model.key_to_index
        self.vocab_size = len(self.word_to_index)

    @property
    def params(self) -> dict:
        """
        Returns the parameters this StemGenerator is working with (for statistics purposes).
        """
        return {}

    def generate_stemming_dict(self, model: KeyedVectors, vocabulary: Iterable[str]) -> StemDict:
        log.info("Generating stem dict for words...")
        model_stem_dict = {}
        self.set_model(model)
        with AsyncTaskManager(workers_amount=self.workers_amount) as task_manager:
            log.debug("Appending stemming tasks...")
            for word in tqdm(vocabulary, desc="Add stemming tasks"):
                task_manager.add_task(self.find_word_inflections, args=(word,))
            log.debug("Collecting stemming results...")
            for result in tqdm(task_manager, total=task_manager.total_task_count, desc="Generate stem dict"):
                model_stem_dict.update(result)
        log.info(f"Total {len(model_stem_dict)} stems generated")
        stem_dict = reduce_stem_dict(stem_dict=model_stem_dict)
        reduced_dict = sort_dict_by_values(stem_dict)
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
