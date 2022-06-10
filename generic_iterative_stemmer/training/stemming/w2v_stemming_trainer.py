from gensim.models import KeyedVectors

from generic_iterative_stemmer.training.base import word2vec
from generic_iterative_stemmer.training.stemming import StemmingTrainer


class Word2VecStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, **kwargs) -> KeyedVectors:
        model = word2vec.train(corpus_path=corpus_file_path, **kwargs)
        return model.wv


if __name__ == "__main__":
    from utils.loader import get_path
    from utils.logging import configure_logging

    configure_logging()

    corpus_name = "wiki-he-cbow"
    corpus_folder = get_path(corpus_name)
    trainer = Word2VecStemmingTrainer(corpus_folder=corpus_folder, max_iterations=20, completed_iterations=14)
    trainer.train()
