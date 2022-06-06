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
    configure_logging(detailed_json=False, pretty_json=True, level="INFO")

    training_program = [
        IterationProgram(
            stem_generator=DefaultStemGenerator(min_cosine_similarity=0.85, max_edit_distance=0),
            iteration_kwargs={
                "remove_words_not_in_model": True,
            },
        ),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.85, max_edit_distance=1, min_cosine_similarity_for_edit_distance=0.82
            ),
        ),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.80, max_edit_distance=1, min_cosine_similarity_for_edit_distance=0.80
            )
        ),
        IterationProgram(
            stem_generator=DefaultStemGenerator(
                min_cosine_similarity=0.80, max_edit_distance=2, min_cosine_similarity_for_edit_distance=0.82
            ),
            training_params={"epochs": 8},
        ),
    ]

    corpus_name = "skv-ft-30"
    corpus_folder = get_path("hebrew", corpus_name)
    training_params = {"vector_size": 30, "epochs": 2, "window": 10, "skip_gram": True, "min_count": 5}
    default_stemming_params = {"min_cosine_similarity": 0.80, "min_cosine_similarity_for_edit_distance": 0.82}
    # trainer = Word2VecStemmingTrainer(
    trainer = FastTextStemmingTrainer(
        corpus_folder=corpus_folder,
        max_iterations=5,
        completed_iterations=0,
        training_program=training_program,
        default_training_params=training_params,
        default_stem_generator_params=default_stemming_params,
    )
    trainer.save_stem_dict()

    trainer.train()


if __name__ == "__main__":
    main()
