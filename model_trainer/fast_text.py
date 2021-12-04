import logging

import fasttext

from model_trainer.config import get_path
from utils.logging import measure_time

log = logging.getLogger(__name__)


@measure_time
def train(corpus_path: str, output_model_path: str, algorithm: str = "cbow"):
    model = fasttext.train_unsupervised(input=corpus_path, model=algorithm)
    model.save(output_model_path)


def get_model(model_path: str = "wiki-he.fasttext.model.bin"):
    model = fasttext.load_model(model_path)
    return model


if __name__ == '__main__':
    corpus = get_path("small", "corpus.txt")
    output = get_path("small", "wiki-he.fasttext.model")
    train(corpus, output)
