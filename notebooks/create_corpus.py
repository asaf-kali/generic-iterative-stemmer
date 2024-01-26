from generic_iterative_stemmer.training.base.create_corpus import (
    generate_wiki_corpus_file,
    hebrew_tokenizer_no_suffix,
)
from tests.utils.loader import get_path
from tests.utils.logging import configure_logging


def create_wiki_corpus(corpus_folder_name: str, language_code: str):
    configure_logging()
    articles_zip = get_path(corpus_folder_name, f"{language_code}wiki-latest-pages-articles.xml.bz2")
    corpus_out_path = get_path(corpus_folder_name, "corpus.txt")
    generate_wiki_corpus_file(
        articles_file_path=articles_zip,
        output_file_path=corpus_out_path,
        tokenizer_func=hebrew_tokenizer_no_suffix,
    )


if __name__ == "__main__":
    create_wiki_corpus(corpus_folder_name="hebrew", language_code="he")
