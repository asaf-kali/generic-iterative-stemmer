import json
import logging
import os

from gensim.models import KeyedVectors

from ..errors import StemDictFileNotFoundError

log = logging.getLogger(__name__)


def get_model_path(base_folder: str) -> str:
    return os.path.join(base_folder, "model.kv")


def get_stem_dict_path_from_model_path(model_path: str) -> str:
    return f"{model_path}.stem-dict.json"


def get_stem_dict_path_from_iteration_folder(base_folder: str) -> str:
    model_path = get_model_path(base_folder)
    return get_stem_dict_path_from_model_path(model_path)


def save_stem_dict(stem_dict: dict, model_path: str):
    stem_dict_path = get_stem_dict_path_from_model_path(model_path)
    with open(stem_dict_path, "w") as file:
        serialized = json.dumps(stem_dict, indent=2, ensure_ascii=False)
        file.write(serialized)
    log.debug(f"Stem dict saved: {stem_dict_path}.")


def validate_stem_dict_is_reduced(stem_dict: dict):
    return set(stem_dict.keys()).intersection(set(stem_dict.values())) == set()


class StemmedKeyedVectors:
    def __init__(self, kv: KeyedVectors, stem_dict: dict):
        self.kv = kv
        self.stem_dict = stem_dict
        # Override get_vector
        self._inner_get_vector = self.kv.get_vector
        self.kv.get_vector = self.get_vector

    def __getattr__(self, item):
        # This allows StemmedKeyedVectors to act like its underling KeyedVectors
        return getattr(self.kv, item)

    def __getitem__(self, item):
        return self.kv.__getitem__(item)

    def __contains__(self, item) -> bool:
        if item in self.stem_dict:
            return True
        return self.kv.__contains__(item)

    def stem(self, word: str) -> str:
        return self.stem_dict.get(word, word)

    def get_vector(self, key, norm=False):
        stem = self.stem(key)
        return self._inner_get_vector(stem, norm=norm)

    def most_similar(self, *args, **kwargs):
        similarities = self.kv.most_similar(*args, **kwargs)
        stemmed_similarities = [(self.stem(word), score) for word, score in similarities]
        return stemmed_similarities

    @classmethod
    def load(cls, file_name: str, mmap=None):
        kv: KeyedVectors = KeyedVectors.load(fname=file_name, mmap=mmap)  # type: ignore
        stem_dict_path = get_stem_dict_path_from_model_path(file_name)
        try:
            with open(stem_dict_path) as file:
                stem_dict = json.load(file)
        except FileNotFoundError as e:
            raise StemDictFileNotFoundError() from e
        return StemmedKeyedVectors(kv=kv, stem_dict=stem_dict)

    def save(self, file_name: str, *args, **kwargs):
        self.kv.save(file_name, *args, **kwargs)
        save_stem_dict(self.stem_dict, file_name)
