import logging

from gensim.models import FastText

log = logging.getLogger(__name__)


def train(
    corpus_path: str,
    output_model_path: str = None,
    skip_gram: bool = False,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    **kwargs,
):
    sg = 1 if skip_gram else 0
    model = FastText(
        corpus_file=corpus_path,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        **kwargs,
    )
    model.train(
        corpus_file=corpus_path,
        total_words=model.corpus_total_words,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )
    log.debug("FastText training done.")
    if output_model_path is not None:
        log.debug("Saving model.")
        model_file_path = f"{output_model_path}.model"
        kv_file_path = f"{output_model_path}.kv"
        model.save(model_file_path)
        model.wv.save(kv_file_path)
    return model


if __name__ == "__main__":
    from ...utils import get_path

    corpus_name = "small"
    corpus = get_path(corpus_name, "corpus.txt")
    output = get_path(corpus_name, "fasttext.model")
    train(corpus, output)
