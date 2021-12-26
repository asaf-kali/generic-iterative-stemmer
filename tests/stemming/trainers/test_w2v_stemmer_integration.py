from typing import Type

from stemming.trainers.test_stemming_trainer_integration import (
    TestStemmingTrainerIntegration,
)

from generic_iterative_stemmer.training.stemming import (
    StemmingTrainer,
    Word2VecStemmingTrainer,
)


class TestWord2VecStemmerIntegration(TestStemmingTrainerIntegration):
    @property
    def stemmer_class(self) -> Type[StemmingTrainer]:
        return Word2VecStemmingTrainer
