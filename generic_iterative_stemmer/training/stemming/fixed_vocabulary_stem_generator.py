import logging
from typing import Iterable

from ...training.stemming import StemDict, StemGenerator

log = logging.getLogger(__name__)


class FixedVocabularyStemGenerator(StemGenerator):
    def __init__(self, legal_words: Iterable[str], workers_amount: int = 5):
        super().__init__(workers_amount=workers_amount)
        self.legal_words = set(legal_words)

    def is_legal(self, word: str) -> bool:
        return word in self.legal_words

    def find_word_inflections(self, word: str) -> StemDict:
        if self.is_legal(word):
            return {}
        similarities = self.model.most_similar(word, topn=5)
        first_similarity = similarities[0]
        return {word: first_similarity[0]}
        # similarities = model.most_similar(word, topn=20)
        # for candidate, grade in similarities:
        #     if not self.is_legal(candidate):
        #         continue
        #     if candidate in word:
        #         return {word: candidate}
        # return {}

    @property
    def params(self) -> dict:
        return {"legal_words": list(self.legal_words)}
