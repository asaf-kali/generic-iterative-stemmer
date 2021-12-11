import logging
from typing import Iterable

from gensim.models import KeyedVectors
from tqdm import tqdm

from utils.logging import measure_time

log = logging.getLogger(__name__)


class InflectionsDictGenerator:
    def __init__(self, model: KeyedVectors, k: int = 50, min_grade: float = 0.75, max_len_diff: int = 5):
        self.model = model
        self.k = k
        self.min_grade = min_grade
        self.max_len_diff = max_len_diff

    def generate_word_inflections(self, word: str) -> dict:
        similarities = self.model.most_similar(word, topn=self.k)
        inflections = {}
        for other, grade in similarities:
            if word not in other:
                continue
            if grade < self.min_grade:
                continue
            if abs(len(word) - len(other)) > self.max_len_diff:
                continue
            inflections[other] = word
        return inflections

    @measure_time
    def generate_model_inflections(self, words: Iterable[str] = None):
        if words is None:
            words = self.model.key_to_index.keys()
        model_inflections = {}
        log.info("Generating inflections...2")
        for word in tqdm(words):
            word_inflections = self.generate_word_inflections(word=word)
            model_inflections.update(word_inflections)
        log.info(f"Total {len(model_inflections)} inflections generated")
        return model_inflections
