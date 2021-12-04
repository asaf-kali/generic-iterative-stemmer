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
        min_count=1,  # was 5
        workers=multiprocessing.cpu_count(),
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=1)

    # trim unneeded model memory = use (much) less RAM
    # model.init_sims(replace=True)
    model.save(output_model_path)


def getModel(model_name="wiki-he.word2vec.model"):
    model = Word2Vec.load(model_name)
    # model = KeyedVectors.load_word2vec_format(model_name)
    return model


if __name__ == '__main__':
    corpus = get_data("small", "corpus.txt")
    output = get_data("small", "small.word2vec.model")
    train(corpus, output)
