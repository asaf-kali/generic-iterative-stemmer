import os
import shutil
from unittest import TestCase

from generic_iterative_stemmer.utils import get_path

TEST_CORPUS_FOLDER = "./tests/data/small"


class StemmerIntegrationTest(TestCase):
    corpus_name: str
    src_corpus_folder: str
    test_corpus_folder: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus_name = "small"
        cls.src_corpus_folder = get_path(cls.corpus_name)
        cls.test_corpus_folder = TEST_CORPUS_FOLDER

    def _reset_corpus_folder(self):
        shutil.rmtree(self.test_corpus_folder, ignore_errors=True)
        src_corpus_file_path = os.path.join(self.src_corpus_folder, "corpus.txt")
        test_corpus_file_path = os.path.join(self.test_corpus_folder, "iter-1", "corpus.txt")
        os.makedirs(os.path.dirname(test_corpus_file_path))
        shutil.copyfile(src=src_corpus_file_path, dst=test_corpus_file_path)

    def setUp(self) -> None:
        self._reset_corpus_folder()
