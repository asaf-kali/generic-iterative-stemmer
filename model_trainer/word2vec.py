import logging
import multiprocessing

from gensim.models import Word2Vec
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
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=1)

    # trim unneeded model memory = use (much) less RAM
    # model.init_sims(replace=True)
    model.save(output_model_path)


def get_model(model_path: str):
    model = Word2Vec.load(model_path)
    vectors = model.wv
    # model = KeyedVectors.load_word2vec_format(model_name)
    log.info("Model loaded")
    return vectors


if __name__ == '__main__':
    corpus = get_data("wiki-he", "corpus.txt")
    output = get_data("wiki-he", "wiki-he.word2vec.sg.model")
    train(corpus, output)
