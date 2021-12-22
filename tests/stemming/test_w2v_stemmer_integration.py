import json
import os
import shutil
from typing import Set
from unittest import TestCase

from generic_iterative_stemmer.training import Word2VecStemmingTrainer
from generic_iterative_stemmer.utils import get_path

TEST_CORPUS_DIRECTORY = "./tests/data/small"


class TestWord2VecStemmerIntegration(TestCase):
    corpus_name: str
    src_corpus_directory: str
    test_corpus_directory: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus_name = "small"
        cls.src_corpus_directory = get_path(cls.corpus_name)
        cls.test_corpus_directory = TEST_CORPUS_DIRECTORY

    def _reset_corpus_directory(self):
        shutil.rmtree(self.test_corpus_directory, ignore_errors=True)
        src_corpus_file_path = os.path.join(self.src_corpus_directory, "corpus.txt")
        test_corpus_file_path = os.path.join(self.test_corpus_directory, "iter-1", "corpus.txt")
        os.makedirs(os.path.dirname(test_corpus_file_path))
        shutil.copyfile(src=src_corpus_file_path, dst=test_corpus_file_path)

    def setUp(self) -> None:
        self._reset_corpus_directory()

    def test_load_trainer_from_state_sanity(self):
        trainer = Word2VecStemmingTrainer(corpus_directory=self.test_corpus_directory, max_iterations=2)
        trainer.train()

        loaded_trainer = Word2VecStemmingTrainer.load_from_state_file(self.test_corpus_directory)
        assert loaded_trainer.completed_iterations == 2
        assert loaded_trainer.iteration_folders_names == ["iter-1", "iter-2", "iter-3"]

        loaded_trainer.run_iteration()
        assert loaded_trainer.iteration_folders_names == ["iter-1", "iter-2", "iter-3", "iter-4"]

    def test_stemmed_words_do_not_repeat(self):
        trainer = Word2VecStemmingTrainer(corpus_directory=self.test_corpus_directory, max_iterations=20)
        trainer.train()

        assert trainer.completed_iterations > 1

        stemmed_words: Set[str] = set()
        for iteration_folder in trainer.iteration_folders_paths:
            if not os.path.isdir(iteration_folder):
                continue
            stats_path = os.path.join(iteration_folder, "stats.json")
            if not os.path.exists(stats_path):
                continue
            with open(stats_path) as stats_file:
                stats = json.load(stats_file)
            iteration_stem_dict: dict = stats["stem_dict"]
            iteration_stemmed_words = set(iteration_stem_dict.keys())
            assert stemmed_words.intersection(iteration_stemmed_words) == set()
            stemmed_words.update(iteration_stemmed_words)
