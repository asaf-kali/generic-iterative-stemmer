import os

TEST_DATA_FOLDER = os.path.join("tests", "data")
TEST_RUNTIME_FOLDER = os.path.join("tests", "runtime_data")

CORPUS_SMALL = "corpus-small"
CORPUS_TINY = "corpus-tiny"


def get_test_src_corpus_path(corpus_name: str) -> str:
    return os.path.join(TEST_DATA_FOLDER, f"{corpus_name}.txt")


def get_runtime_file_path(*path: str) -> str:
    return os.path.join(TEST_RUNTIME_FOLDER, *path)


def get_large_stem_dict_path() -> str:
    return os.path.join(TEST_DATA_FOLDER, "large-stem-dict.json")
