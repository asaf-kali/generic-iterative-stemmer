import os.path

import pytest
from stemming.conftest import CorpusResource

from generic_iterative_stemmer.errors import StemDictFileNotFoundError
from generic_iterative_stemmer.models import (
    StemmedKeyedVectors,
    get_model_path,
    get_stem_dict_path_from_model_path,
)
from generic_iterative_stemmer.training.stemming import Word2VecStemmingTrainer


class TestStemmedKeyedVector:
    def test_skv_is_not_loaded_without_stem_dict(self, corpus_resource: CorpusResource):
        trainer = Word2VecStemmingTrainer(corpus_folder=corpus_resource.test_corpus_folder, max_iterations=1)
        trainer.train()

        model_path = get_model_path(base_folder=trainer.last_completed_iteration_folder)
        stem_dict_path = get_stem_dict_path_from_model_path(model_path=model_path)
        assert os.path.exists(stem_dict_path)

        os.remove(stem_dict_path)
        with pytest.raises(StemDictFileNotFoundError):
            _ = StemmedKeyedVectors.load(model_path)
