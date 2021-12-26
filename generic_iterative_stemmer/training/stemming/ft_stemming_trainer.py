from gensim.models import FastText, KeyedVectors

from generic_iterative_stemmer.training.stemming import StemmingTrainer


class FastTextStemmingTrainer(StemmingTrainer):
    def train_model_on_corpus(self, corpus_file_path: str, iteration_number: int) -> KeyedVectors:
        model = FastText(corpus_file=corpus_file_path)
        model.train(
            corpus_file=corpus_file_path,
            total_words=model.corpus_total_words,
            total_examples=model.corpus_count,
            epochs=model.epochs,
        )
        return model.wv


if __name__ == "__main__":
    from generic_iterative_stemmer.utils import get_path

    corpus_name = "wiki-he-fasttext"
    corpus_folder = get_path(corpus_name)
    trainer = FastTextStemmingTrainer(corpus_folder=corpus_folder, max_iterations=10, completed_iterations=0)
    trainer.train()
