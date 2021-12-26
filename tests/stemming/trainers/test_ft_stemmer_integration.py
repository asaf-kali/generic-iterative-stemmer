from typing import Type

from stemming.trainers.test_stemming_trainer_integration import (
    TestStemmingTrainerIntegration,
)

from generic_iterative_stemmer.training.stemming import (
    FastTextStemmingTrainer,
    StemmingTrainer,
)


class TestFastTextStemmerIntegration(TestStemmingTrainerIntegration):
    @property
    def stemmer_class(self) -> Type[StemmingTrainer]:
        return FastTextStemmingTrainer
