import logging
import multiprocessing
from typing import Optional

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

log = logging.getLogger(__name__)


def train(
    corpus_path: str,
    output_model_path: Optional[str] = None,
    skip_gram: bool = False,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    **kwargs,
) -> Word2Vec:
    sentences = LineSentence(corpus_path)
    sg = 1 if skip_gram else 0
    model = Word2Vec(
        sg=sg,  # 0=CBOW , 1=SkipGram
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
        epochs=epochs,
        **kwargs,
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=epochs)
    log.debug("Word2Vec training done.")
    if output_model_path is not None:
        log.debug("Saving model.")
        model_file_path = f"{output_model_path}.model"
        kv_file_path = f"{output_model_path}.kv"
        model.save(model_file_path)
        model.wv.save(kv_file_path)
    return model
