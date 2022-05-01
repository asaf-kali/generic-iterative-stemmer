from typing import Set

from pydantic import BaseModel
from tqdm import tqdm

from ...utils import get_logger
from ..stemming import StemDict

log = get_logger(__name__)


class StemSentenceResult(BaseModel):
    stemmed_sentence: str
    total_word_count: int
    total_stem_count: int


class StemCorpusResult(BaseModel):
    unique_stem_count: int
    unique_word_count: int = 0
    total_stem_count: int = 0
    total_word_count: int = 0

    def add(self, stem_sentence_result: StemSentenceResult):
        self.total_word_count += stem_sentence_result.total_word_count
        self.total_stem_count += stem_sentence_result.total_stem_count

    @property
    def unique_stemming_ratio(self) -> float:
        return self.unique_stem_count / self.unique_word_count if self.unique_word_count else 0

    @property
    def total_stemming_ratio(self) -> float:
        return self.total_stem_count / self.total_word_count if self.total_word_count else 0


def stem_corpus(original_corpus_path: str, output_corpus_path: str, stem_dict: StemDict):
    stemmer = CorpusStemmer(stem_dict=stem_dict)
    return stemmer.stem_corpus(
        original_corpus_path=original_corpus_path,
        output_corpus_path=output_corpus_path,
    )


class CorpusStemmer:
    def __init__(self, stem_dict: StemDict):
        self.stem_dict = stem_dict
        self.unique_stems = set(stem_dict.values())
        self.unique_words: Set[str] = set()

    def stem_sentence(self, sentence: str) -> StemSentenceResult:
        words = sentence.split()
        stemmed_words = [self.stem_dict.get(word, word) for word in words]
        self.unique_words.update(stemmed_words)
        stemmed_sentence = " ".join(stemmed_words)
        if sentence[-1] == "\n":
            stemmed_sentence += "\n"
        stem_result = StemSentenceResult(
            stemmed_sentence=stemmed_sentence,
            total_word_count=len(words),
            total_stem_count=sum(1 if x in self.unique_stems else 0 for x in stemmed_words),
        )
        return stem_result

    def stem_corpus(self, original_corpus_path: str, output_corpus_path: str) -> StemCorpusResult:
        log.info("Stemming corpus...")
        stem_corpus_result = StemCorpusResult(unique_stem_count=len(self.unique_stems))
        with open(original_corpus_path) as original_file, open(output_corpus_path, "w") as output_file:
            for sentence in tqdm(original_file, desc="Corpus stemming"):
                stem_sentence_result = self.stem_sentence(sentence=sentence)
                stem_corpus_result.add(stem_sentence_result)
                output_file.write(stem_sentence_result.stemmed_sentence)
        stem_corpus_result.unique_word_count = len(self.unique_words)
        log.info("Stemming corpus done")
        return stem_corpus_result
