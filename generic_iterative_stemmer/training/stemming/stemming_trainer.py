import json
import logging
import os.path
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from gensim.models import KeyedVectors
from tqdm import tqdm

from ...utils import measure_time
from . import StemDict, StemDictGenerator

log = logging.getLogger(__name__)

ITER_FOLDER_PATTERN = re.compile(r"iter-\d+")


def stem_sentence(sentence: str, stem_dict: StemDict) -> str:
    words = sentence.split(" ")
    words_replaced = [stem_dict.get(word, word) for word in words]
    sentence_replaced = " ".join(words_replaced)
    return sentence_replaced


@measure_time
def stem_corpus(original_corpus_path: str, output_corpus_path: str, stem_dict: StemDict):
    log.info("Stemming corpus...")
    with open(original_corpus_path) as original_file, open(output_corpus_path, "w") as output_file:
        for sentence in tqdm(original_file):
            reduced_sentence = stem_sentence(sentence, stem_dict=stem_dict)
            output_file.write(reduced_sentence)
    log.info("Stemming corpus done")


def get_iteration_directory(base_directory: str, iteration_number: int) -> str:
    directory = os.path.join(base_directory, f"iter-{iteration_number}")
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_corpus_path(base_directory: str):
    return os.path.join(base_directory, "corpus.txt")


def get_model_path(base_directory: str):
    return os.path.join(base_directory, "model.kv")


def get_stemming_trainer_state_path(base_directory: str):
    return os.path.join(base_directory, "stemming-trainer-state.json")


@dataclass
class StemmingIterationStats:
    initial_vocab_size: Optional[int] = None
    stem_dict: Optional[dict] = None

    @property
    def stem_dict_size(self) -> int:
        return 0 if self.stem_dict is None else len(self.stem_dict)

    def as_dict(self) -> dict:
        return {
            "initial_vocab_size": self.initial_vocab_size,
            "stem_dict_size": self.stem_dict_size,
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
        self, trainer: "StemmingTrainer", iteration_number: int, corpus_directory: str, base_model: KeyedVectors = None
    ):
        self.trainer = trainer
        self.iteration_number = iteration_number
        self.corpus_directory = corpus_directory
        self.model = base_model
        self.stats = StemmingIterationStats()

    @property
    def iteration_directory(self) -> str:
        return get_iteration_directory(self.corpus_directory, iteration_number=self.iteration_number)

    @property
    def next_iteration_directory(self) -> str:
        return get_iteration_directory(self.corpus_directory, iteration_number=self.iteration_number + 1)

    @property
    def iteration_corpus_path(self) -> str:
        return get_corpus_path(self.iteration_directory)

    @property
    def next_iteration_corpus_path(self) -> str:
        return get_corpus_path(self.next_iteration_directory)

    @property
    def iteration_trained_model_path(self) -> str:
        return get_model_path(self.iteration_directory)

    @property
    def has_trained_model(self) -> bool:
        return self.model is not None

    def run_stemming_iteration(self) -> StemmingIterationStats:
        log.info(f"Running stemming iteration number {self.iteration_number}")
        if not self.has_trained_model:
            # TODO: First, try loading it
            self.model = self.train_model()
        self.stem_corpus()
        self.save_stats()
        log.info(f"Stemming iteration {self.iteration_number} completed")
        return self.stats

    def train_model(self) -> KeyedVectors:
        model = self.trainer.train_model_on_corpus(
            corpus_file_path=self.iteration_corpus_path, iteration_number=self.iteration_number
        )
        model.save(self.iteration_trained_model_path)
        return model

    def generate_stem_dict(self) -> StemDict:
        # TODO: Allow inserting more args to the generator.
        stem_generator = StemDictGenerator(model=self.model)
        return stem_generator.generate_model_stemming()

    def stem_corpus(self):
        self.stats.initial_vocab_size = len(self.model.key_to_index)
        stem_dict = self.generate_stem_dict()
        self.stats.stem_dict = stem_dict
        stem_corpus(
            original_corpus_path=self.iteration_corpus_path,
            output_corpus_path=self.next_iteration_corpus_path,
            stem_dict=stem_dict,
        )

    def save_stats(self):
        stats_file = os.path.join(self.iteration_directory, "stats.json")
        with open(stats_file, "w") as file:
            stats_dict = self.stats.as_dict()
            serialized = json.dumps(stats_dict, indent=2, ensure_ascii=False)
            file.write(serialized)


class StemmingTrainer:
    def __init__(
        self,
        corpus_directory: str,
        completed_iterations: int = 0,
        latest_model: Optional[KeyedVectors] = None,
        max_iterations: Optional[int] = 10,
    ):
        self.corpus_directory = corpus_directory
        self.completed_iterations = completed_iterations
        self.latest_model = latest_model
        self.max_iterations = max_iterations

    @classmethod
    def load_from_state_file(cls, corpus_directory: str, **kwargs) -> "StemmingTrainer":
        state_path = get_stemming_trainer_state_path(corpus_directory)
        with open(state_path) as state_file:
            content = state_file.read()
        state: dict = json.loads(content)
        state.update(kwargs)
        return cls(**state)

    @property
    def state(self) -> dict:
        return {
            "corpus_directory": self.corpus_directory,
            "completed_iterations": self.completed_iterations,
            "max_iterations": self.max_iterations,
        }

    @property
    def iteration_folders_names(self) -> List[str]:
        corpus_sub_files = os.listdir(self.corpus_directory)
        iter_folders = [file for file in corpus_sub_files if ITER_FOLDER_PATTERN.fullmatch(file)]
        iter_folders.sort()
        return iter_folders

    @property
    def iteration_folders_paths(self) -> List[str]:
        return [os.path.join(self.corpus_directory, folder) for folder in self.iteration_folders_names]

    @measure_time
    def train(self):
        log.info("Starting stemmer iterations training...")
        while True:
            if self.max_iterations and self.completed_iterations >= self.max_iterations:
                log.info(f"Reached {self.completed_iterations} iterations, quitting.")
                break
            stats = self.run_iteration()
            if stats.stem_dict_size == 0:
                log.info("Iteration ended with no stemming, quitting.")
                break

    @measure_time
    def train_model_on_corpus(self, corpus_file_path: str, iteration_number: int) -> KeyedVectors:
        raise NotImplementedError()

    @measure_time
    def run_iteration(self) -> StemmingIterationStats:
        iteration_trainer = StemmingIterationTrainer(
            trainer=self,
            iteration_number=self.completed_iterations + 1,
            corpus_directory=self.corpus_directory,
            base_model=self.latest_model,
        )
        stats = iteration_trainer.run_stemming_iteration()
        self.completed_iterations += 1
        self.latest_model = None
        self.save_state()
        return stats

    def save_state(self):
        state_path = get_stemming_trainer_state_path(self.corpus_directory)
        with open(state_path, "w") as state_file:
            serialized = json.dumps(self.state, indent=2)
            state_file.write(serialized)

    def _collect_complete_stem_dict(self):
        pass

    def save(self):

        pass
