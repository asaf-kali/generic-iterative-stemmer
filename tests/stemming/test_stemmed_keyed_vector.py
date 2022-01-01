import os.path

import numpy as np
import pytest
from stemming.conftest import CorpusResource

from generic_iterative_stemmer.errors import StemDictFileNotFoundError
from generic_iterative_stemmer.models import (
    StemmedKeyedVectors,
    get_model_path,
    get_stem_dict_path_from_model_path,
)
from generic_iterative_stemmer.training.stemming import (
    StemmingTrainer,
    Word2VecStemmingTrainer,
)


@pytest.fixture(scope="class")
def trainer_resource() -> StemmingTrainer:
    corpus_name = "small"
    corpus = CorpusResource(corpus_name)
    corpus.reset_corpus_folder()
    trainer = Word2VecStemmingTrainer(corpus_folder=corpus.test_corpus_folder, max_iterations=2)
    trainer.train()
    return trainer


class TestStemmedKeyedVector:
    def test_skv_is_not_loaded_without_stem_dict(self, trainer_resource: StemmingTrainer):
        model_path = get_model_path(base_folder=trainer_resource.last_completed_iteration_folder)
        stem_dict_path = get_stem_dict_path_from_model_path(model_path=model_path)
        assert os.path.exists(stem_dict_path)

        os.rename(stem_dict_path, "temp")
        with pytest.raises(StemDictFileNotFoundError):
            _ = StemmedKeyedVectors.load(model_path)
        os.rename("temp", stem_dict_path)

    def test_loaded_skv_sanity(self, trainer_resource: StemmingTrainer):
        model_path = get_model_path(base_folder=trainer_resource.last_completed_iteration_folder)
        kv = StemmedKeyedVectors.load(model_path)

        assert isinstance(kv, StemmedKeyedVectors)
        assert len(kv.index_to_key) > 0
        some_key = kv.index_to_key[0]
        some_word = kv[some_key]
        assert isinstance(some_word, np.ndarray)

        similarities = kv.most_similar(some_word)
        assert isinstance(similarities, list)
        assert len(similarities) > 0
