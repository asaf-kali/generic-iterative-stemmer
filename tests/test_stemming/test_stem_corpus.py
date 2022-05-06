from test_stemming.conftest import CorpusResource
from utils.data import CORPUS_TINY, get_runtime_file_path

from generic_iterative_stemmer.training.stemming.corpus_stemmer import (
    CorpusStemmer,
    StemCorpusResult,
    stem_corpus,
)


def test_stem_sentence():
    sentence = "a b c d e f g"
    stem_dict = {"a": "x"}
    stemmer = CorpusStemmer(stem_dict=stem_dict)
    result = stemmer.stem_sentence(sentence=sentence)
    expected = "x b c d e f g"
    assert result.stemmed_sentence == expected
    assert result.total_word_count == 7
    assert result.total_stem_count == 1


def test_stem_sentence_on_last_word_in_hebrew_sentence():
    sentence = " הוקמ בשנת קטגוריה הרכבימ מוזיקליימ מירושלימ\n"
    stem_dict = {"מירושלימ": "ירושלימ"}
    stemmer = CorpusStemmer(stem_dict=stem_dict)
    result = stemmer.stem_sentence(sentence=sentence)
    expected = "הוקמ בשנת קטגוריה הרכבימ מוזיקליימ ירושלימ\n"
    assert result.stemmed_sentence == expected
    assert result.total_word_count == 6
    assert result.total_stem_count == 1


def test_stem_corpus():
    corpus_resource = CorpusResource(corpus_name=CORPUS_TINY)
    output = get_runtime_file_path("output.txt")
    stem_dict = {"b": "a", "c": "a", "g": "f", "l": "k"}
    result: StemCorpusResult = stem_corpus(
        original_corpus_path=corpus_resource.test_runtime_corpus_path,
        output_corpus_path=output,
        stem_dict=stem_dict,
    )

    assert result.unique_stem_count == 3
    assert result.total_stem_count == 10
    assert result.total_word_count == 25
    assert result.unique_stemming_ratio == (3 / 16)  # (a, f, k) / (16 unique words in result)
    assert result.total_stemming_ratio == (10 / 25)  # (a + f + k appearances) / (25 total words in result)


def test_stem_corpus_with_allowed_words():
    corpus_resource = CorpusResource(corpus_name=CORPUS_TINY)
    output = get_runtime_file_path("output.txt")
    stem_dict = {"b": "a", "c": "a", "g": "f", "l": "k"}
    allowed_words = {"a", "g"}  # Only "a" should remain, because "g" is stemmed to "f"
    result: StemCorpusResult = stem_corpus(
        original_corpus_path=corpus_resource.test_runtime_corpus_path,
        output_corpus_path=output,
        stem_dict=stem_dict,
        allowed_words=allowed_words,
    )

    assert result.unique_stem_count == 1
    assert result.total_stem_count == 6
    assert result.total_word_count == 6
    assert result.unique_stemming_ratio == 1
    assert result.total_stemming_ratio == 1
