import logging
from typing import Optional

import fasttext as ft

from ...utils import measure_time

log = logging.getLogger(__name__)


@measure_time
def train(corpus_path: str, output_model_path: Optional[str] = None, algorithm: str = "cbow"):
    model = ft.train_unsupervised(input=corpus_path, model=algorithm)
    if output_model_path is not None:
        model.save_model(output_model_path)
    return model


if __name__ == "__main__":
    from ...utils import get_path

    corpus_name = "small"
    corpus = get_path(corpus_name, "corpus.txt")
    output = get_path(corpus_name, "fasttext.model")
    train(corpus, output)
