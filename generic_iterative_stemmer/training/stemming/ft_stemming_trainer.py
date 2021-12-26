from gensim.models import KeyedVectors

from generic_iterative_stemmer.training.base import fast_text
from generic_iterative_stemmer.training.stemming import StemmingTrainer


class FastTextStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, iteration_number: int) -> KeyedVectors:
        model = fast_text.train(corpus_file_path)
        return model.wv


if __name__ == "__main__":
    from generic_iterative_stemmer.utils import get_path

    corpus_name = "wiki-he-fasttext"
    corpus_folder = get_path(corpus_name)
    trainer = FastTextStemmingTrainer(corpus_folder=corpus_folder, max_iterations=15, completed_iterations=10)
    trainer.train()
