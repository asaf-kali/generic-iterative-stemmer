import copy
import json
import logging
import os.path
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

from gensim.models import KeyedVectors
from pydantic import BaseModel
from the_spymaster_util.measure_time import MeasureTime

from ...errors import StemmingTrainerError
from ...helpers import remove_file_exit_ok, sort_dict_by_values
from ...models import (
    StemmedKeyedVectors,
    get_model_path,
    get_stem_dict_path_from_model_path,
    save_stem_dict,
)
from . import StemDict, StemGenerator, reduce_stem_dict
from .corpus_stemmer import StemCorpusResult, stem_corpus
from .default_stem_generator import DefaultStemGenerator

log = logging.getLogger(__name__)

ITER_FOLDER_PATTERN = re.compile(r"iter-\d+")


class StemmingIterationStats(BaseModel):
    iteration_number: int
    initial_vocab_size: Optional[int] = None
    time_measures: dict = {}
    training_params: Optional[dict] = None
    stem_generator_params: Optional[dict] = None
    stem_corpus_result: Optional[StemCorpusResult] = None
    stem_dict: Optional[dict] = None

    @property
    def stem_dict_size(self) -> int:
        return 0 if self.stem_dict is None else len(self.stem_dict)


@dataclass
class IterationProgram:
    stem_generator: Optional[StemGenerator] = None
    override_training_params: Optional[dict] = field(default_factory=dict)
    iteration_kwargs: Optional[dict] = field(default_factory=dict)


class StemmingIterationTrainer:
    """
    Each stemming iteration i is given corpus_i (that was generated in the i-1 iteration), and:
        * Trains a model based on corpus i (unless a model is given to it).
        * Generates a stem dict for the trained model.
        * Generates a stemmed corpus (for iteration i+1 to train on).
    """

    def __init__(
        self,
        trainer: "StemmingTrainer",
        stem_generator: StemGenerator,
        iteration_number: int,
        corpus_folder: str,
        model: Optional[KeyedVectors] = None,
        remove_words_not_in_model: bool = False,
        training_params: Optional[dict] = None,
    ):
        self.trainer = trainer
        self.stem_generator = stem_generator
        self.iteration_number = iteration_number
        self.corpus_folder = corpus_folder
        self.model = model
        self.training_params = training_params or {}
        self.remove_words_not_in_model = remove_words_not_in_model
        self.stats = StemmingIterationStats(
            iteration_number=self.iteration_number, training_params=self.training_params
        )

    @property
    def iteration_folder(self) -> str:
        return get_iteration_folder(self.corpus_folder, iteration_number=self.iteration_number)

    @property
    def next_iteration_folder(self) -> str:
        return get_iteration_folder(self.corpus_folder, iteration_number=self.iteration_number + 1)

    @property
    def iteration_corpus_path(self) -> str:
        return get_corpus_path(self.iteration_folder)

    @property
    def next_iteration_corpus_path(self) -> str:
        return get_corpus_path(self.next_iteration_folder)

    @property
    def iteration_trained_model_path(self) -> str:
        return get_model_path(self.iteration_folder)

    @property
    def is_first_iteration(self) -> bool:
        return self.iteration_number == 1

    def _try_load_trained_model(self) -> Optional[KeyedVectors]:
        model_path = self.iteration_trained_model_path
        if os.path.exists(model_path):
            return KeyedVectors.load(model_path)  # type: ignore
        return None

    def run_stemming_iteration(self) -> StemmingIterationStats:
        log.info(f"Running stemming iteration number {self.iteration_number}.")
        self.load_stats()
        with MeasureTime() as mt:
            self.model = self._try_load_trained_model()
            if self.model:
                log.info("Found existing model for iteration, skipping training.")
            else:
                self.model = self.train_model()
            self.generate_stemmed_corpus()
            log.info(f"Stemming iteration {self.iteration_number} completed.")
        self.stats.time_measures["run_stemming_iteration"] = mt.delta
        self.save_stats()
        return self.stats

    def _first_iteration_check(self):
        if not self.is_first_iteration or os.path.exists(self.iteration_corpus_path):
            return
        base_corpus_path = get_corpus_path(self.corpus_folder)
        try:
            os.makedirs(self.iteration_folder, exist_ok=True)
            shutil.move(src=base_corpus_path, dst=self.iteration_corpus_path)
        except Exception as e:
            log.warning(f"Failed to copy base corpus into first iteration folder: {e}")

    def train_model(self) -> KeyedVectors:
        log.info(f"Training model for iteration {self.iteration_number}.")
        self._first_iteration_check()
        with MeasureTime() as mt:
            model = self.trainer.train_model_on_corpus(
                corpus_file_path=self.iteration_corpus_path, **self.training_params
            )
        model.save(self.iteration_trained_model_path)
        self.stats.time_measures["train_model"] = mt.delta
        return model

    def generate_stem_dict(self) -> StemDict:
        self.stats.stem_generator_params = self.stem_generator.params or {}
        self.stats.stem_generator_params["stem_generator_class"] = self.stem_generator.__class__.__name__
        vocabulary = self.model.key_to_index.keys()  # type: ignore
        with MeasureTime() as mt:
            stem_dict = self.stem_generator.generate_stemming_dict(model=self.model, vocabulary=vocabulary)
        self.stats.time_measures["generate_stemmed_corpus"] = mt.delta
        return stem_dict

    def generate_stemmed_corpus(self):
        self.stats.initial_vocab_size = len(self.model.key_to_index)
        if not self.stats.stem_dict:
            self.stats.stem_dict = self.generate_stem_dict()
        self.save_stats()  # We have to save stats before stem_corpus for iteration stem dict to be in the state file.
        if len(self.stats.stem_dict) == 0:
            log.info("Stem dict was empty, skipping corpus stemming.")
            return
        self.stem_corpus()

    def stem_corpus(self):
        with MeasureTime() as mt:
            complete_stem_dict = self.trainer.collect_complete_stem_dict()
        self.stats.time_measures["collect_complete_stem_dict"] = mt.delta
        allowed_words = set(self.model.key_to_index.keys()) if self.remove_words_not_in_model else None
        with MeasureTime() as mt:
            self.stats.stem_corpus_result = stem_corpus(
                original_corpus_path=self.iteration_corpus_path,
                output_corpus_path=self.next_iteration_corpus_path,
                stem_dict=complete_stem_dict,
                allowed_words=allowed_words,
            )
        self.stats.time_measures["stem_corpus"] = mt.delta
        if not self.is_first_iteration and os.path.exists(self.next_iteration_corpus_path):
            remove_file_exit_ok(self.iteration_corpus_path)

    def save_stats(self):
        stats_file = get_stats_path(self.iteration_folder)
        with open(stats_file, "w") as file:
            stats_dict = self.stats.model_dump()
            serialized = json.dumps(stats_dict, indent=2, ensure_ascii=False)
            file.write(serialized)

    def load_stats(self):
        stats_file = get_stats_path(self.iteration_folder)
        try:
            with open(stats_file) as file:
                data = json.load(file)
                self.stats = StemmingIterationStats(**data)
        except Exception as e:  # noqa
            pass


class StemmingTrainer:
    def __init__(
        self,
        corpus_folder: str,
        completed_iterations: int = 0,
        max_iterations: Optional[int] = 10,
        is_fully_stemmed: bool = False,
        min_change_count: int = 0,
        training_program: Optional[List[IterationProgram]] = None,
        default_training_params: Optional[dict] = None,
        default_stem_generator_class: Type[StemGenerator] = DefaultStemGenerator,
        default_stem_generator_params: Optional[dict] = None,
    ):
        self.corpus_folder = corpus_folder
        if not os.path.exists(self.corpus_folder):
            raise StemmingTrainerError(f"Corpus folder {self.corpus_folder} does not exist.")
        self.completed_iterations = completed_iterations
        self.max_iterations = max_iterations
        self.is_fully_stemmed = is_fully_stemmed
        self.min_change_count = min_change_count
        self.training_program = training_program or []
        self.default_training_params = default_training_params or {}
        self.default_stem_generator_class = default_stem_generator_class
        self.default_stem_generator_params = default_stem_generator_params or {}

    @classmethod
    def load_from_state_file(cls, corpus_folder: str, **kwargs) -> "StemmingTrainer":
        state_path = get_stemming_trainer_state_path(corpus_folder)
        with open(state_path) as state_file:
            content = state_file.read()
        state: dict = json.loads(content)
        state.update(kwargs)
        return cls(**state)

    @property
    def state(self) -> dict:
        return {
            "corpus_folder": self.corpus_folder,
            "completed_iterations": self.completed_iterations,
            "max_iterations": self.max_iterations,
            "is_fully_stemmed": self.is_fully_stemmed,
            "default_training_params": self.default_training_params,
            "default_stem_generator_params": self.default_stem_generator_params,
        }

    @property
    def iteration_folders_names(self) -> List[str]:
        corpus_sub_files = os.listdir(self.corpus_folder)
        iter_folders = [file for file in corpus_sub_files if ITER_FOLDER_PATTERN.fullmatch(file)]
        iter_folders.sort(key=lambda file_name: int(file_name[5:]))  # Sort them by iteration index
        return iter_folders

    @property
    def iteration_folders_paths(self) -> List[str]:
        return [os.path.join(self.corpus_folder, folder_name) for folder_name in self.iteration_folders_names]

    @property
    def last_completed_iteration_folder(self) -> str:
        if self.completed_iterations < 1:
            raise StemmingTrainerError("No completed iterations yet.")
        return get_iteration_folder(self.corpus_folder, iteration_number=self.completed_iterations)

    def train(self, save_stem_dict_when_done: bool = True):
        log.info("Starting iterations stemmer training...")
        self.save_state()
        while True:
            if self.max_iterations and self.completed_iterations >= self.max_iterations:
                log.info(f"Reached {self.completed_iterations} iterations, quitting.")
                break
            if self.is_fully_stemmed:
                log.info("Already fully stemmed, quitting.")
                break
            iteration_stats = self.run_iteration()
            if iteration_stats.stem_dict_size <= self.min_change_count:
                log.info(f"Iteration ended with no more then {self.min_change_count} stems, quitting.")
                self.is_fully_stemmed = True
                break
        if save_stem_dict_when_done:
            self.save_stem_dict()
        self.save_state()

    def train_model_on_corpus(self, corpus_file_path: str, **kwargs) -> KeyedVectors:
        raise NotImplementedError()

    def run_iteration(self) -> StemmingIterationStats:
        iteration_number = self.completed_iterations + 1
        iteration_program = self.get_iteration_program(iteration_number=iteration_number)
        stem_generator = self.get_stem_generator(iteration_program=iteration_program)
        training_params = self.get_training_params(iteration_program=iteration_program)
        iteration_kwargs = iteration_program.iteration_kwargs if iteration_program else {}
        iteration_trainer = StemmingIterationTrainer(
            trainer=self,
            stem_generator=stem_generator,
            iteration_number=iteration_number,
            corpus_folder=self.corpus_folder,
            training_params=training_params,
            **iteration_kwargs,  # type: ignore
        )
        iteration_stats = iteration_trainer.run_stemming_iteration()
        self.completed_iterations += 1
        self.save_state()
        return iteration_stats

    def get_iteration_program(self, iteration_number: int) -> Optional[IterationProgram]:
        try:
            return self.training_program[iteration_number - 1]
        except IndexError:
            return None

    def get_training_params(self, iteration_program: Optional[IterationProgram]) -> dict:
        training_params = copy.deepcopy(self.default_training_params)
        if iteration_program:
            iteration_training_params = iteration_program.override_training_params or {}
            training_params.update(iteration_training_params)
        return training_params

    def get_stem_generator(self, iteration_program: Optional[IterationProgram]) -> StemGenerator:
        if iteration_program and iteration_program.stem_generator:
            return iteration_program.stem_generator
        stem_generator_params = self.default_stem_generator_params
        return self.default_stem_generator_class(**stem_generator_params)

    def save_state(self):
        state_path = get_stemming_trainer_state_path(self.corpus_folder)
        with open(state_path, "w") as state_file:
            serialized = json.dumps(self.state, indent=2, ensure_ascii=False)
            state_file.write(serialized)

    def collect_complete_stem_dict(self) -> StemDict:
        stem_dict = {}
        for iteration_folder in self.iteration_folders_paths:
            stats_path = get_stats_path(iteration_folder)
            if not os.path.exists(stats_path):
                continue
            with open(stats_path) as stats_file:
                stats: dict = json.load(stats_file)
            iteration_stem_dict = stats["stem_dict"]
            stem_dict.update(iteration_stem_dict)
        reduced_stem_dict = reduce_stem_dict(stem_dict)
        sorted_stem_dict = sort_dict_by_values(reduced_stem_dict)
        return sorted_stem_dict

    def save_stem_dict(self):
        if self.completed_iterations == 0:
            log.info("No iterations completed, skipping save.")
            return
        stem_dict = self.collect_complete_stem_dict()
        model_path = get_model_path(self.last_completed_iteration_folder)
        save_stem_dict(stem_dict=stem_dict, model_path=model_path)

    def get_stemmed_keyed_vectors(self) -> StemmedKeyedVectors:
        model_path = get_model_path(base_folder=self.last_completed_iteration_folder)
        stem_dict_path = get_stem_dict_path_from_model_path(model_path=model_path)
        if not os.path.exists(stem_dict_path):
            self.save_stem_dict()
        return StemmedKeyedVectors.load(file_name=model_path)


def get_iteration_folder(base_folder: str, iteration_number: int) -> str:
    folder = os.path.join(base_folder, f"iter-{iteration_number:02d}")
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def get_corpus_path(base_folder: str) -> str:
    return os.path.join(base_folder, "corpus.txt")


def get_stats_path(base_folder: str) -> str:
    return os.path.join(base_folder, "stats.json")


def get_stemming_trainer_state_path(base_folder: str) -> str:
    return os.path.join(base_folder, "stemming-trainer-state.json")
