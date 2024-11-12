import logging
from typing import Optional, Set

from pydantic import BaseModel
from tqdm import tqdm

from ..stemming import StemDict

log = logging.getLogger(__name__)


class StemSentenceResult(BaseModel):
    stemmed_sentence: str
    total_word_count: int
    total_stem_count: int


class StemCorpusResult(BaseModel):
    unique_stem_count: int = 0
    unique_word_count: int = 0
    total_word_count: int = 0
    total_stem_count: int = 0

    def add(self, stem_sentence_result: StemSentenceResult):
        self.total_word_count += stem_sentence_result.total_word_count
        self.total_stem_count += stem_sentence_result.total_stem_count

    def model_dump(self, *args, **kwargs) -> dict:
        result = super().model_dump(*args, **kwargs)
        result["unique_stemming_ratio"] = round(self.unique_stemming_ratio, 3)
        result["total_stemming_ratio"] = round(self.total_stemming_ratio, 3)
        return result

    @property
    def unique_stemming_ratio(self) -> float:
        # The ratio of resulting unique stems to unique words in the corpus.
        return self.unique_stem_count / self.unique_word_count if self.unique_word_count else 0

    @property
    def total_stemming_ratio(self) -> float:
        # The ratio of total stemming to total words in the corpus.
        return self.total_stem_count / self.total_word_count if self.total_word_count else 0


def stem_corpus(
    original_corpus_path: str, output_corpus_path: str, stem_dict: StemDict, allowed_words: Optional[Set[str]] = None
):
    stemmer = CorpusStemmer(stem_dict=stem_dict, allowed_words=allowed_words)
    return stemmer.stem_corpus(
        original_corpus_path=original_corpus_path,
        output_corpus_path=output_corpus_path,
    )


class CorpusStemmer:
    def __init__(self, stem_dict: StemDict, allowed_words: Optional[Set[str]] = None):
        self.stem_dict = stem_dict
        self.stems = set(stem_dict.values())
        self.result_unique_words: Set[str] = set()
        self.allowed_words = allowed_words or set()

    def stem_sentence(self, sentence: str) -> StemSentenceResult:
        words = sentence.split()
        stemmed_words = [self.stem_dict.get(word, word) for word in words]
        if self.allowed_words:
            stemmed_words = [word for word in stemmed_words if word in self.allowed_words]
        self.result_unique_words.update(stemmed_words)
        stemmed_sentence = " ".join(stemmed_words)
        if sentence[-1] == "\n":
            stemmed_sentence += "\n"
        stem_result = StemSentenceResult(
            stemmed_sentence=stemmed_sentence,
            total_word_count=len(stemmed_words),
            total_stem_count=sum(1 if x in self.stems else 0 for x in stemmed_words),
        )
        return stem_result

    def stem_corpus(self, original_corpus_path: str, output_corpus_path: str) -> StemCorpusResult:
        log.info("Stemming corpus...")
        stem_corpus_result = StemCorpusResult()
        with open(original_corpus_path) as original_file, open(output_corpus_path, "w") as output_file:
            for sentence in tqdm(original_file, desc="Corpus stemming"):
                stem_sentence_result = self.stem_sentence(sentence=sentence)
                stem_corpus_result.add(stem_sentence_result)
                output_file.write(stem_sentence_result.stemmed_sentence)
        stem_corpus_result.unique_stem_count = len(self.stems.intersection(self.result_unique_words))
        stem_corpus_result.unique_word_count = len(self.result_unique_words)
        log.info("Stemming corpus done")
        return stem_corpus_result
