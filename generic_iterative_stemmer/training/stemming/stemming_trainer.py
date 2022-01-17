import json
import logging
import os.path
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type

from gensim.models import KeyedVectors
from tqdm import tqdm

from ...errors import StemmingTrainerError
from ...models import (
    StemmedKeyedVectors,
    get_model_path,
    get_stem_dict_path_from_model_path,
    save_stem_dict,
)
from ...utils import loader, measure_time
from . import StemDict, StemGenerator, reduce_stem_dict
from .default_stem_generator import DefaultStemGenerator

log = logging.getLogger(__name__)

ITER_FOLDER_PATTERN = re.compile(r"iter-\d+")


def stem_sentence(sentence: str, stem_dict: StemDict) -> str:
    words = sentence.split()
    words_replaced = [stem_dict.get(word, word) for word in words]
    sentence_replaced = " ".join(words_replaced)
    if sentence[-1] == "\n":
        sentence_replaced += "\n"
    return sentence_replaced


@measure_time
def stem_corpus(original_corpus_path: str, output_corpus_path: str, stem_dict: StemDict):
    log.info("Stemming corpus...")
    with open(original_corpus_path) as original_file, open(output_corpus_path, "w") as output_file:
        for sentence in tqdm(original_file, desc="Corpus stemming"):
            reduced_sentence = stem_sentence(sentence, stem_dict=stem_dict)
            output_file.write(reduced_sentence)
    log.info("Stemming corpus done")


def get_iteration_folder(base_folder: str, iteration_number: int) -> str:
    folder = os.path.join(base_folder, f"iter-{iteration_number}")
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def get_corpus_path(base_folder: str) -> str:
    return os.path.join(base_folder, "corpus.txt")


def get_stats_path(base_folder: str) -> str:
    return os.path.join(base_folder, "stats.json")


def get_stemming_trainer_state_path(base_folder: str) -> str:
    return os.path.join(base_folder, "stemming-trainer-state.json")


@dataclass
class StemmingIterationStats:
    initial_vocab_size: Optional[int] = None
    stem_dict: Optional[dict] = None
    stem_generator_params: Optional[dict] = None

    @property
    def stem_dict_size(self) -> int:
        return 0 if self.stem_dict is None else len(self.stem_dict)

    def as_dict(self) -> dict:
        return {
            "initial_vocab_size": self.initial_vocab_size,
            "stem_dict_size": self.stem_dict_size,
            "stem_generator_params": self.stem_generator_params,
            "stem_dict": self.stem_dict,
        }


class StemmingIterationTrainer:
    """
    Each stemming iteration i is given the corpus-i (that was generated in the i-1 iteration), and:
        * Trains a model based on corpus i (unless a model is given to it).
        * Generates a stem dict for the trained model.
        * Generates a stemmed corpus for iteration i+1 to base on.
    """

    def __init__(
        self,
        trainer: "StemmingTrainer",
        stem_generator: StemGenerator,
        iteration_number: int,
        corpus_folder: str,
        base_model: KeyedVectors = None,
    ):
        self.trainer = trainer
        self.stem_generator = stem_generator
        self.iteration_number = iteration_number
        self.corpus_folder = corpus_folder
        self.model = base_model
        self.stats = StemmingIterationStats()

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

    def _try_load_trained_model(self) -> Optional[KeyedVectors]:
        model_path = self.iteration_trained_model_path
        if os.path.exists(model_path):
            return loader.load_kv(model_path)
        return None

    def run_stemming_iteration(self) -> StemmingIterationStats:
        log.info(f"Running stemming iteration number {self.iteration_number}.")
        self.model = self._try_load_trained_model()
        if not self.model:
            self.model = self.train_model()
        self.generate_stemmed_corpus()
        log.info(f"Stemming iteration {self.iteration_number} completed.")
        return self.stats

    def train_model(self) -> KeyedVectors:
        model = self.trainer.train_model_on_corpus(
            corpus_file_path=self.iteration_corpus_path, iteration_number=self.iteration_number
        )
        model.save(self.iteration_trained_model_path)
        return model

    def generate_stem_dict(self) -> StemDict:
        self.stats.stem_generator_params = self.stem_generator.params
        vocabulary = self.model.key_to_index.keys()  # type: ignore
        return self.stem_generator.generate_stemming_dict(model=self.model, vocabulary=vocabulary)

    def generate_stemmed_corpus(self):
        self.stats.initial_vocab_size = len(self.model.key_to_index)
        stem_dict = self.generate_stem_dict()
        self.stats.stem_dict = stem_dict
        self.save_stats()
        if len(stem_dict) == 0:
            log.info("Stem dict was empty, skipping corpus stemming.")
            return
        stem_corpus(
            original_corpus_path=self.iteration_corpus_path,
            output_corpus_path=self.next_iteration_corpus_path,
            stem_dict=stem_dict,
        )

    def save_stats(self):
        stats_file = get_stats_path(self.iteration_folder)
        with open(stats_file, "w") as file:
            stats_dict = self.stats.as_dict()
            serialized = json.dumps(stats_dict, indent=2, ensure_ascii=False)
            file.write(serialized)


class StemmingTrainer:
    def __init__(
        self,
        corpus_folder: str,
        completed_iterations: int = 0,
        latest_model: Optional[KeyedVectors] = None,
        max_iterations: Optional[int] = 10,
        is_fully_stemmed: bool = False,
        min_change_count: int = 0,
        stemming_program: List[StemGenerator] = None,
        default_stem_generator_class: Type[StemGenerator] = DefaultStemGenerator,
        default_stem_generator_params: Optional[dict] = None,
    ):
        self.corpus_folder = corpus_folder
        self.completed_iterations = completed_iterations
        self.latest_model = latest_model  # TODO: Think about that field.
        self.max_iterations = max_iterations
        self.is_fully_stemmed = is_fully_stemmed
        self.min_change_count = min_change_count
        self.stemming_program = stemming_program or []
        self.default_stem_generator_class = default_stem_generator_class  # TODO: This can't be serialized and loaded.
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
        # TODO: Replace with serializable model.
        return {
            "corpus_folder": self.corpus_folder,
            "completed_iterations": self.completed_iterations,
            "max_iterations": self.max_iterations,
            "is_fully_stemmed": self.is_fully_stemmed,
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

    @measure_time
    def train(self, save_stem_dict_when_done: bool = True):
        log.info("Starting iterations stemmer training...")
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

    @measure_time
    def train_model_on_corpus(self, corpus_file_path: str, iteration_number: int) -> KeyedVectors:
        raise NotImplementedError()

    @measure_time
    def run_iteration(self) -> StemmingIterationStats:
        iteration_number = self.completed_iterations + 1
        stem_generator = self.get_stem_generator(iteration_number=iteration_number)
        iteration_trainer = StemmingIterationTrainer(
            trainer=self,
            stem_generator=stem_generator,
            iteration_number=iteration_number,
            corpus_folder=self.corpus_folder,
            base_model=self.latest_model,
        )
        iteration_stats = iteration_trainer.run_stemming_iteration()
        self.completed_iterations += 1
        self.latest_model = None
        self.save_state()
        return iteration_stats

    def get_stem_generator(self, iteration_number: int) -> StemGenerator:
        try:
            stem_generator = self.stemming_program[iteration_number]
        except IndexError:
            stem_generator_params = self.default_stem_generator_params
            stem_generator = self.default_stem_generator_class(**stem_generator_params)
        return stem_generator

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
        return reduced_stem_dict

    @property
    def last_completed_iteration_folder(self) -> str:
        if self.completed_iterations < 1:
            raise StemmingTrainerError("No completed iterations yet.")
        return get_iteration_folder(self.corpus_folder, iteration_number=self.completed_iterations)

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
