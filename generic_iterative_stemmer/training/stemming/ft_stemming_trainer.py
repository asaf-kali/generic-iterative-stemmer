from gensim.models import KeyedVectors

from generic_iterative_stemmer.training.base import fast_text
from generic_iterative_stemmer.training.stemming import StemmingTrainer


class FastTextStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, **kwargs) -> KeyedVectors:
        model = fast_text.train(corpus_path=corpus_file_path, **kwargs)
        return model.wv


if __name__ == "__main__":
    from generic_iterative_stemmer.utils import configure_logging, get_path

    configure_logging()

    corpus_name = "wiki-he-fasttext"
    corpus_folder = get_path(corpus_name)
    trainer = FastTextStemmingTrainer(corpus_folder=corpus_folder, max_iterations=5, completed_iterations=0)
    trainer.train()
