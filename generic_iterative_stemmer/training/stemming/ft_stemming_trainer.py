from gensim.models import KeyedVectors

from generic_iterative_stemmer.training.base import fast_text
from generic_iterative_stemmer.training.stemming import StemmingTrainer


class FastTextStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, **kwargs) -> KeyedVectors:
        model = fast_text.train(corpus_path=corpus_file_path, **kwargs)
        return model.wv
