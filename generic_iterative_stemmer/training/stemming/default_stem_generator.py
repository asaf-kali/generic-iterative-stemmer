import logging
from typing import Optional

import editdistance

from ...training.stemming import StemDict, StemGenerator

log = logging.getLogger(__name__)


class DefaultStemGenerator(StemGenerator):
    def __init__(
        self,
        k: Optional[int] = 10,
        min_cosine_similarity: Optional[float] = 0.75,
        min_cosine_similarity_for_edit_distance: Optional[float] = 0.85,
        max_len_diff: Optional[int] = 3,
        max_edit_distance: Optional[int] = 1,
    ):
        super().__init__()
        self.k = k
        self.min_cosine_similarity = min_cosine_similarity
        self.min_cosine_similarity_for_edit_distance = min_cosine_similarity_for_edit_distance
        self.max_len_diff = max_len_diff
        self.max_edit_distance = max_edit_distance

    def find_word_inflections(self, word: str) -> StemDict:
        """
        Find which other words in the vocabulary can be stemmed down to this word.
        """
        similarities = self.model.most_similar(word, topn=self.k)
        stem_dict = {}
        for inflection, cosine_similarity in similarities:
            if cosine_similarity < self.min_cosine_similarity:
                continue
            if len(inflection) <= len(word):
                continue
            word_frequency, inflection_frequency = self.get_frequency(word), self.get_frequency(inflection)
            if word_frequency < inflection_frequency:
                continue
            if word not in inflection:
                if self.max_edit_distance is None:
                    continue
                edit_distance = editdistance.eval(word, inflection)
                if edit_distance > self.max_edit_distance:
                    continue
                if cosine_similarity < self.min_cosine_similarity_for_edit_distance:
                    continue
            if self.max_len_diff and abs(len(word) - len(inflection)) > self.max_len_diff:
                continue
            log.debug(
                f"Reducing '{inflection}' to '{word}'",
                extra={
                    "stem": word,
                    "inflection": inflection,
                    "grade": round(cosine_similarity, 3),
                    "stem_frequency": round(word_frequency, 3),
                    "inflection_frequency": round(inflection_frequency, 3),
                },
            )
            stem_dict[inflection] = word
        return stem_dict

    def get_frequency(self, word: str) -> float:
        return 1 - self.word_to_index[word] / self.vocab_size

    @property
    def params(self) -> dict:
        return {
            "k": self.k,
            "min_cosine_similarity": self.min_cosine_similarity,
            "min_cosine_similarity_for_edit_distance": self.min_cosine_similarity_for_edit_distance,
            "max_len_diff": self.max_len_diff,
            "max_edit_distance": self.max_edit_distance,
        }
