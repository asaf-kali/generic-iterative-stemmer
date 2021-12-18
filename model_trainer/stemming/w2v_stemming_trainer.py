from gensim.models import KeyedVectors

from model_trainer import word2vec
from model_trainer.config import get_path
from model_trainer.stemming.stemming_trainer import StemmingTrainer


class Word2VecStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, iteration_number: int) -> KeyedVectors:
        model = word2vec.train(corpus_file_path)
        return model.wv


if __name__ == "__main__":
    corpus_name = "wiki-he-filter"
    corpus_directory = get_path(corpus_name)
    trainer = Word2VecStemmingTrainer(corpus_directory=corpus_directory, max_iterations=10, completed_iterations=5)
    trainer.train()
