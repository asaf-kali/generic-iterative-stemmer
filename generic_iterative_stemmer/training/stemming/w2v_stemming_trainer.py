from gensim.models import KeyedVectors

from generic_iterative_stemmer.training.base import word2vec
from generic_iterative_stemmer.training.stemming import StemmingTrainer


class Word2VecStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, **kwargs) -> KeyedVectors:
        model = word2vec.train(corpus_path=corpus_file_path, **kwargs)
        return model.wv
