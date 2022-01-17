import logging
from typing import Iterable

from gensim.models import KeyedVectors

from generic_iterative_stemmer.training.stemming import StemDict, StemGenerator

log = logging.getLogger(__name__)


class IllegalWordsStemmer(StemGenerator):
    def __init__(self, legal_words: Iterable[str], workers_amount: int = 5):
        super().__init__(workers_amount=workers_amount)
        self.legal_words = set(legal_words)

    def is_legal(self, word: str) -> bool:
        return word in self.legal_words

    def find_word_inflections(self, model: KeyedVectors, word: str) -> StemDict:
        if self.is_legal(word):
            return {}
        similarities = model.most_similar(word, topn=5)
        first_similarity = similarities[0]
        return {word: first_similarity[0]}

    @property
    def params(self) -> dict:
        return {"legal_words": list(self.legal_words)}
