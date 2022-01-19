from generic_iterative_stemmer.training.stemming import Word2VecStemmingTrainer
from generic_iterative_stemmer.training.stemming.default_stem_generator import (
    DefaultStemGenerator,
)
from generic_iterative_stemmer.training.stemming.stemming_trainer import (
    IterationProgram,
)
from generic_iterative_stemmer.utils import configure_logging, get_path


def main():
    configure_logging()

    corpus_name = "wiki-he-cbow-2"
    corpus_folder = get_path(corpus_name)
    training_program = [
        IterationProgram(stem_generator=DefaultStemGenerator(min_cosine_similarity=0.85, max_edit_distance=0)),
        IterationProgram(stem_generator=DefaultStemGenerator(min_cosine_similarity=0.80, max_edit_distance=0)),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.75, max_edit_distance=1, min_cosine_similarity_for_edit_distance=0.85
            )
        ),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.75, max_edit_distance=2, min_cosine_similarity_for_edit_distance=0.80
            )
        ),
        IterationProgram(stem_generator=DefaultStemGenerator(min_cosine_similarity=0.75, max_edit_distance=0)),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.75, max_edit_distance=1, min_cosine_similarity_for_edit_distance=0.80
            )
        ),
        IterationProgram(stem_generator=DefaultStemGenerator(min_cosine_similarity=0.75, max_edit_distance=0)),
    ]

    trainer = Word2VecStemmingTrainer(corpus_folder=corpus_folder, max_iterations=10, training_program=training_program)
    trainer.train()


if __name__ == "__main__":
    main()
