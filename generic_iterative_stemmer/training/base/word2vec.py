import logging
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from ...utils import measure_time

log = logging.getLogger(__name__)


@measure_time
def train(
    corpus_path: str,
    output_model_path: str = None,
    skip_gram: bool = False,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
) -> Word2Vec:
    sentences = LineSentence(corpus_path)
    sg = 1 if skip_gram else 0
    model = Word2Vec(
        sg=sg,  # 0=CBOW , 1=SkipGram
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=5)
    log.debug("Training done")
    if output_model_path is not None:
        log.debug("Saving model")
        model_file_path = f"{output_model_path}.model"
        kv_file_path = f"{output_model_path}.kv"
        model.save(model_file_path)
        model.wv.save(kv_file_path)
    return model


if __name__ == "__main__":
    from ...utils import get_path

    corpus_name = "wiki-he"
    corpus_file = get_path(corpus_name, "corpus.txt")
    output_file = get_path(corpus_name, "cbow")
    train(corpus_file, output_file)
