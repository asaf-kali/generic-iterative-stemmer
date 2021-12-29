import json
import logging
import os
from typing import Set, Type

import pytest
from stemming.conftest import CorpusResource

from generic_iterative_stemmer.errors import StemmingTrainerError
from generic_iterative_stemmer.models import (
    StemmedKeyedVectors,
    get_stem_dict_path_from_iteration_folder,
)
from generic_iterative_stemmer.training.stemming import (
    FastTextStemmingTrainer,
    StemmingTrainer,
    Word2VecStemmingTrainer,
)

log = logging.getLogger(__name__)


def assert_skv_sanity(skv: StemmedKeyedVectors, is_fully_stemmed: bool = True):
    model_vocab = set(skv.key_to_index.keys())
    stemmed_words = set(skv.stem_dict.keys())
    present_words_count = 0
    for stemmed_word in stemmed_words:
        if stemmed_word in model_vocab:
            present_words_count += 1
        assert skv[stemmed_word] is not None
    if is_fully_stemmed:
        # The term "fully" is misleading, because it can be under min_change_count but still non-0.
        assert present_words_count < len(stemmed_words) / 3


@pytest.mark.parametrize("trainer_class", [Word2VecStemmingTrainer, FastTextStemmingTrainer])
class TestStemmingTrainersIntegration:
    def test_load_trainer_from_state_sanity(
        self, corpus_resource: CorpusResource, trainer_class: Type[StemmingTrainer]
    ):
        trainer = trainer_class(corpus_folder=corpus_resource.test_corpus_folder, max_iterations=1)
        trainer.train()

        loaded_trainer = trainer_class.load_from_state_file(corpus_resource.test_corpus_folder)
        assert loaded_trainer.completed_iterations == 1
        assert loaded_trainer.iteration_folders_names == ["iter-1", "iter-2"]

        loaded_trainer.run_iteration()
        assert loaded_trainer.completed_iterations == 2
        assert loaded_trainer.iteration_folders_names == ["iter-1", "iter-2", "iter-3"]

    def test_stemmed_words_do_not_appear_in_more_then_one_iteration(
        self, corpus_resource: CorpusResource, trainer_class: Type[StemmingTrainer]
    ):
        trainer = trainer_class(corpus_folder=corpus_resource.test_corpus_folder, max_iterations=5)
        trainer.train()

        assert trainer.completed_iterations > 1

        stemmed_words: Set[str] = set()
        for iteration_folder in trainer.iteration_folders_paths:
            stats_path = os.path.join(iteration_folder, "stats.json")
            if not os.path.exists(stats_path):
                continue
            with open(stats_path) as stats_file:
                stats = json.load(stats_file)
            iteration_stem_dict: dict = stats["stem_dict"]
            iteration_stemmed_words = set(iteration_stem_dict.keys())
            assert stemmed_words.intersection(iteration_stemmed_words) == set()
            stemmed_words.update(iteration_stemmed_words)

        # TODO: This is actually a different test.
        stem_dict_path = get_stem_dict_path_from_iteration_folder(trainer.last_completed_iteration_folder)
        with open(stem_dict_path) as file:
            completed_stem_dict = json.load(file)

        assert len(completed_stem_dict) == len(stemmed_words)

    # def test_no_stemmed_corpus_is_generated_when_stemming_is_complete(self, trainer_class: Type[StemmingTrainer]):
    #     trainer = trainer_class(
    #         corpus_folder=self.test_corpus_folder, max_iterations=None
    #     )
    #     trainer.train()
    #
    #     assert trainer.completed_iterations > 0
    #     assert trainer.completed_iterations == len(trainer.iteration_folders_names)
    #     assert trainer.is_fully_stemmed
    #     for iteration_folder in trainer.iteration_folders_paths:
    #         model_path = get_model_path(iteration_folder)
    #         stats_path = get_stats_path(iteration_folder)
    #         assert os.path.exists(model_path)
    #         assert os.path.exists(stats_path)

    def test_get_stemmed_keyed_vectors(self, corpus_resource: CorpusResource, trainer_class: Type[StemmingTrainer]):
        trainer = trainer_class(
            corpus_folder=corpus_resource.test_corpus_folder, max_iterations=None, min_change_count=10
        )
        trainer.train()

        kv = trainer.get_stemmed_keyed_vectors()
        assert_skv_sanity(kv, is_fully_stemmed=False)

    def test_get_stemmed_keyed_vectors_when_stem_dict_is_not_saved(
        self, corpus_resource: CorpusResource, trainer_class: Type[StemmingTrainer]
    ):
        trainer = trainer_class(corpus_folder=corpus_resource.test_corpus_folder, max_iterations=1)
        trainer.train(save_stem_dict_when_done=False)

        kv = trainer.get_stemmed_keyed_vectors()
        assert_skv_sanity(kv, is_fully_stemmed=False)

    def test_last_completed_iteration_folder(
        self, corpus_resource: CorpusResource, trainer_class: Type[StemmingTrainer]
    ):
        trainer = trainer_class(corpus_folder=corpus_resource.test_corpus_folder)
        with pytest.raises(StemmingTrainerError):
            _ = trainer.last_completed_iteration_folder

        trainer.run_iteration()
        assert trainer.last_completed_iteration_folder.endswith("iter-1")

        trainer.run_iteration()
        assert trainer.last_completed_iteration_folder.endswith("iter-2")