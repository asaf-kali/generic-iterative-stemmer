import logging

from tqdm import tqdm

from model_trainer.stemming import StemDict
from utils.logging import measure_time

log = logging.getLogger(__name__)


def stem_sentence(sentence: str, stem_dict: StemDict) -> str:
    words = sentence.split(" ")
    words_replaced = [stem_dict.get(word, word) for word in words]
    sentence_replaced = " ".join(words_replaced)
    return sentence_replaced


@measure_time
def stem_corpus(original_corpus_path: str, reduced_corpus_path: str, stem_dict: StemDict):
    log.info("Stemming corpus...")
    with open(original_corpus_path) as original_file, open(reduced_corpus_path, "w") as output_file:
        for sentence in tqdm(original_file):
            reduced_sentence = stem_sentence(sentence, stem_dict=stem_dict)
            output_file.write(reduced_sentence)
    log.info("Stemming corpus done")


class StemmingTrainer:
    def __init__(self):
        self.current_iteration = 0
        pass

    @measure_time
    def train(self):
        pass

    def stem_corpus_file(self):
        pass


class Word2VecStemmingTrainer(StemmingTrainer):
    pass
