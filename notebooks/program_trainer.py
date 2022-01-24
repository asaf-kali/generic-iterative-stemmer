from generic_iterative_stemmer.training.stemming import (  # noqa
    FastTextStemmingTrainer,
    Word2VecStemmingTrainer,
)
from generic_iterative_stemmer.training.stemming.default_stem_generator import (
    DefaultStemGenerator,
)
from generic_iterative_stemmer.training.stemming.stemming_trainer import (
    IterationProgram,
)
from generic_iterative_stemmer.utils import configure_logging, get_path


def main():
    configure_logging()

    training_program = [
        IterationProgram(stem_generator=DefaultStemGenerator(min_cosine_similarity=0.80, max_edit_distance=0)),
        IterationProgram(stem_generator=DefaultStemGenerator(min_cosine_similarity=0.75, max_edit_distance=0)),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.75, max_edit_distance=1, min_cosine_similarity_for_edit_distance=0.80
            ),
            training_params={"epochs": 8},
        ),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.75, max_edit_distance=1, min_cosine_similarity_for_edit_distance=0.75
            )
        ),
        IterationProgram(stem_generator=DefaultStemGenerator(min_cosine_similarity=0.75, max_edit_distance=0)),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.70, max_edit_distance=2, min_cosine_similarity_for_edit_distance=0.70
            ),
            training_params={"epochs": 8},
        ),
    ]

    corpus_name = "wiki-he-cbow"
    corpus_folder = get_path(corpus_name)
    training_params = {"vector_size": 150, "epochs": 7, "window": 7}
    default_stemming_params = {"min_cosine_similarity": 0.68}
    # trainer = FastTextStemmingTrainer(
    #     corpus_folder=corpus_folder, max_iterations=10, training_program=training_program
    # )
    trainer = Word2VecStemmingTrainer(
        corpus_folder=corpus_folder,
        max_iterations=8,
        completed_iterations=6,
        training_program=training_program,
        default_training_params=training_params,
        default_stem_generator_params=default_stemming_params,
    )

    trainer.train()


if __name__ == "__main__":
    main()
