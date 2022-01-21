import os

TEST_RUNTIME_CORPUS_FOLDER = os.path.join("tests", "runtime_data")
CORPUS_SMALL = "corpus-small"
CORPUS_TINY = "corpus-tiny"


def get_test_src_corpus_path(corpus_name: str) -> str:
    return os.path.join("tests", "data", f"{corpus_name}.txt")


def get_runtime_file_path(*path: str) -> str:
    return os.path.join(TEST_RUNTIME_CORPUS_FOLDER, *path)
