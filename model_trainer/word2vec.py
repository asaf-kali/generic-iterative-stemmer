import logging
import multiprocessing

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence

from model_trainer.config import get_data
from utils.logging import measure_time

log = logging.getLogger(__name__)


@measure_time
def train(corpus_path: str, output_model_path: str):
    sentences = LineSentence(corpus_path)
    model = Word2Vec(
        sg=1,  # 0=CBOW , 1=SkipGram
        vector_size=100,
        window=5,
        min_count=5,
        workers=multiprocessing.cpu_count(),
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=5)
    log.debug("Training done, saving model...")
    model_file_path = f"{output_model_path}.model"
    kv_file_path = f"{output_model_path}.kv"
    model.save(model_file_path)
    model.wv.save(kv_file_path)


def load_vectors_from_kv(kv_path: str) -> KeyedVectors:
    model: KeyedVectors = KeyedVectors.load(kv_path)  # type: ignore
    log.info("Vectors loaded")
    return model


def load_model(model_path: str) -> Word2Vec:
    model = Word2Vec.load(model_path)
    log.info("Model loaded")
    return model


if __name__ == '__main__':
    corpus = get_data("wiki-he", "corpus.txt")
    output = get_data("wiki-he", "sg")
    train(corpus, output)
