from unittest import TestCase

from generic_iterative_stemmer.training import Word2VecStemmingTrainer
from generic_iterative_stemmer.utils import get_path


class TestWord2VecStemmerIntegration(TestCase):
    corpus_name: str
    corpus_directory: str

    def _clear_iteration_directories(self):
        pass

    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus_name = "small"
        cls.corpus_directory = get_path(cls.corpus_name)

    def tearDown(self) -> None:
        self._clear_iteration_directories()

    def test_load_trainer_from_state_sanity(self):
        trainer = Word2VecStemmingTrainer(corpus_directory=self.corpus_directory, max_iterations=2)
        trainer.train()
        # TODO assert directory 2 and 3 exist
        loaded_trainer = Word2VecStemmingTrainer.load_from_state_file(self.corpus_directory)
        assert loaded_trainer.completed_iterations > 0
        loaded_trainer.run_iteration()
        # TODO assert directory 4 exists
