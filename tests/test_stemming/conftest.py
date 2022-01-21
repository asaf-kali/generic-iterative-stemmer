import os
import shutil

import pytest
from utils.data import (
    CORPUS_SMALL,
    TEST_RUNTIME_CORPUS_FOLDER,
    get_test_src_corpus_path,
)


class CorpusResource:
    corpus_name: str
    src_corpus_path: str
    test_runtime_corpus_folder: str
    test_runtime_corpus_path: str

    def __init__(self, corpus_name: str, test_runtime_corpus_folder: str = TEST_RUNTIME_CORPUS_FOLDER):
        self.corpus_name = corpus_name
        self.src_corpus_path = get_test_src_corpus_path(corpus_name)
        self.test_runtime_corpus_folder = test_runtime_corpus_folder
        self.test_runtime_corpus_path = os.path.join(self.test_runtime_corpus_folder, "iter-1", "corpus.txt")
        self.reset_corpus_folder()

    def reset_corpus_folder(self):
        shutil.rmtree(self.test_runtime_corpus_folder, ignore_errors=True)
        os.makedirs(os.path.dirname(self.test_runtime_corpus_path))
        shutil.copyfile(src=self.src_corpus_path, dst=self.test_runtime_corpus_path)


@pytest.fixture
def corpus_resource(corpus_name: str = CORPUS_SMALL) -> CorpusResource:
    resource = CorpusResource(corpus_name)
    return resource
