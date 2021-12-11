import logging

import fasttext as ft

from model_trainer.config import get_path
from utils.logging import measure_time

log = logging.getLogger(__name__)


@measure_time
def train(corpus_path: str, output_model_path: str, algorithm: str = "cbow"):
    model = ft.train_unsupervised(input=corpus_path, model=algorithm)
    model.save_model(output_model_path)
    return model


if __name__ == "__main__":
    corpus_name = "small"
    corpus = get_path(corpus_name, "corpus.txt")
    output = get_path(corpus_name, "fasttext.model")
    train(corpus, output)
