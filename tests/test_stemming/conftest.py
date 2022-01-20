import os
import shutil

import pytest

TEST_RUNTIME_CORPUS_FOLDER = os.path.join("tests", "runtime_data")


def get_test_src_corpus_path(corpus_name: str) -> str:
    return os.path.join("tests", "data", f"{corpus_name}.txt")


class CorpusResource:
    corpus_name: str
    src_corpus_path: str
    test_runtime_corpus_folder: str

    def __init__(self, corpus_name: str, test_runtime_corpus_folder: str = TEST_RUNTIME_CORPUS_FOLDER):
        self.corpus_name = corpus_name
        self.src_corpus_path = get_test_src_corpus_path(corpus_name)
        self.test_runtime_corpus_folder = test_runtime_corpus_folder

    def reset_corpus_folder(self):
        shutil.rmtree(self.test_runtime_corpus_folder, ignore_errors=True)
        test_corpus_path = os.path.join(self.test_runtime_corpus_folder, "iter-1", "corpus.txt")
        os.makedirs(os.path.dirname(test_corpus_path))
        shutil.copyfile(src=self.src_corpus_path, dst=test_corpus_path)


@pytest.fixture
def corpus_resource() -> CorpusResource:
    corpus_name = "corpus-small"
    resource = CorpusResource(corpus_name)
    resource.reset_corpus_folder()
    return resource
