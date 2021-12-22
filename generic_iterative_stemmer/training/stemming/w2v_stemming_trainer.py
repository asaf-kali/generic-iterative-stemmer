from gensim.models import KeyedVectors

from ..base import word2vec
from . import StemmingTrainer


class Word2VecStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, iteration_number: int) -> KeyedVectors:
        model = word2vec.train(corpus_file_path)
        return model.wv


if __name__ == "__main__":
    from ...utils import get_path

    corpus_name = "wiki-he-filter"
    corpus_folder = get_path(corpus_name)
    trainer = Word2VecStemmingTrainer(corpus_folder=corpus_folder, max_iterations=10, completed_iterations=5)
    trainer.train()
