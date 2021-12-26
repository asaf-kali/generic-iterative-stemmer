from gensim.models import KeyedVectors

from generic_iterative_stemmer.training.base import word2vec
from generic_iterative_stemmer.training.stemming import StemmingTrainer


class Word2VecStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, iteration_number: int) -> KeyedVectors:
        model = word2vec.train(corpus_file_path)
        return model.wv


if __name__ == "__main__":
    from generic_iterative_stemmer.utils import get_path

    corpus_name = "wiki-he-filter"
    corpus_folder = get_path(corpus_name)
    trainer = Word2VecStemmingTrainer(corpus_folder=corpus_folder, max_iterations=15, completed_iterations=10)
    trainer.train()
