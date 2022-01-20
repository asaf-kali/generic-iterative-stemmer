import os
import shutil

import pytest

from generic_iterative_stemmer.utils import get_path

TEST_CORPUS_FOLDER = "./tests/data/small"


class CorpusResource:
    corpus_name: str
    src_corpus_folder: str
    test_corpus_folder: str

    def __init__(self, corpus_name: str, test_corpus_folder: str = TEST_CORPUS_FOLDER):
        self.corpus_name = corpus_name
        self.src_corpus_folder = get_path(corpus_name)
        self.test_corpus_folder = test_corpus_folder

    def reset_corpus_folder(self):
        shutil.rmtree(self.test_corpus_folder, ignore_errors=True)
        src_corpus_file_path = os.path.join(self.src_corpus_folder, "corpus.txt")
        test_corpus_file_path = os.path.join(self.test_corpus_folder, "iter-1", "corpus.txt")
        os.makedirs(os.path.dirname(test_corpus_file_path))
        shutil.copyfile(src=src_corpus_file_path, dst=test_corpus_file_path)


@pytest.fixture
def corpus_resource() -> CorpusResource:
    corpus_name = "small"
    resource = CorpusResource(corpus_name)
    resource.reset_corpus_folder()
    return resource
