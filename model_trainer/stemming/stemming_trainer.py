import json
import logging
import os.path
from typing import Optional

from gensim.models import KeyedVectors
from tqdm import tqdm

from model_trainer.stemming import StemDict, StemDictGenerator
from utils.logging import measure_time

log = logging.getLogger(__name__)


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
    return os.path.join(base_directory, f"iter-{iteration_number}")


def get_corpus_path(base_directory: str):
    return os.path.join(base_directory, "corpus.txt")


def get_model_path(base_directory: str):
    return os.path.join(base_directory, "model.kv")


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
        self.stats: dict = {}

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

    def run_stemming_iteration(self):
        if not self.has_trained_model:
            # TODO: First, try loading it
            self.model = self.train_model()
        self.stem_corpus()

    def train_model(self) -> KeyedVectors:
        model = self.trainer.train_model_on_corpus(corpus_file_path=self.iteration_corpus_path)
        model.save(self.iteration_trained_model_path)
        return model

    def generate_stem_dict(self) -> StemDict:
        # TODO: Allow inserting more args to the generator.
        stem_generator = StemDictGenerator(model=self.model)
        return stem_generator.generate_model_stemming()

    def stem_corpus(self):
        self.stats["initial_vocab_size"] = len(self.model.key_to_index)
        stem_dict = self.generate_stem_dict()
        self.stats["stem_dict"] = stem_dict
        self.stats["stem_dict_size"] = len(stem_dict)
        stem_corpus(
            original_corpus_path=self.iteration_corpus_path,
            output_corpus_path=self.next_iteration_corpus_path,
            stem_dict=stem_dict,
        )

    def save_stats(self):
        stats_file = os.path.join(self.corpus_directory, "stats.json")
        with open(stats_file, "w") as file:
            serialized = json.dumps(self.stats)
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
    def load_from_state(cls, state_path: str, **kwargs) -> "StemmingTrainer":
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

    @measure_time
    def train(self):
        log.info("Starting stemming iterations training...")
        while True:
            if self.max_iterations and self.completed_iterations >= self.max_iterations:
                log.info(f"Reached {self.completed_iterations} iterations, quitting.")
                break
            self.run_iteration()

    @measure_time
    def train_model_on_corpus(self, corpus_file_path: str) -> KeyedVectors:
        raise NotImplementedError()

    @measure_time
    def run_iteration(self):
        iteration_trainer = StemmingIterationTrainer(
            trainer=self,
            iteration_number=self.completed_iterations + 1,
            corpus_directory=self.corpus_directory,
            base_model=self.latest_model,
        )
        iteration_trainer.run_stemming_iteration()
        self.completed_iterations += 1
        self.latest_model = iteration_trainer.model
        self.save_state()

    def save_state(self):
        state_file = os.path.join(self.corpus_directory, "stemming-trainer-state.json")
        with open(state_file, "w") as file:
            serialized = json.dumps(self.state)
            file.write(serialized)


class Word2VecStemmingTrainer(StemmingTrainer):
    pass
