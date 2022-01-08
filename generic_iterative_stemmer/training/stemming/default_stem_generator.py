from typing import Optional

import editdistance
from gensim.models import KeyedVectors

from generic_iterative_stemmer.training.stemming import StemDict, StemGenerator


class DefaultStemGenerator(StemGenerator):
    def __init__(
        self,
        model: KeyedVectors,
        k: Optional[int] = 10,
        min_cosine_similarity: Optional[float] = 0.75,
        min_cosine_similarity_for_edit_distance: Optional[float] = 0.85,
        max_len_diff: Optional[int] = 3,
        max_edit_distance: Optional[int] = 1,
    ):
        super().__init__(model)
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
        for candidate, grade in similarities:
            if grade < self.min_cosine_similarity:
                continue
            if len(candidate) <= len(word):
                continue
            w2i = self.model.key_to_index
            if w2i[candidate] < w2i[word]:
                continue
            if word not in candidate:
                if self.max_edit_distance is None:
                    continue
                edit_distance = editdistance.eval(word, candidate)
                if edit_distance > self.max_edit_distance:
                    continue
                if grade < self.min_cosine_similarity_for_edit_distance:
                    continue
            if self.max_len_diff and abs(len(word) - len(candidate)) > self.max_len_diff:
                continue
            stem_dict[candidate] = word
        return stem_dict

    @property
    def params(self) -> dict:
        return {
            "k": self.k,
            "min_cosine_similarity": self.min_cosine_similarity,
            "min_cosine_similarity_for_edit_distance": self.min_cosine_similarity_for_edit_distance,
            "max_len_diff": self.max_len_diff,
            "max_edit_distance": self.max_edit_distance,
        }
