import json
import logging
import os
from typing import Optional

import numpy as np
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
    log.debug(f"Stem dict saved: {model_path}.")


class StemmedKeyedVectors(KeyedVectors):
    def __init__(
        self,
        stem_dict: dict,
        vector_size: int,
        count: Optional[int] = 0,
        dtype: Optional[type] = np.float32,
        mapfile_path: Optional[str] = None,
    ):
        self.stem_dict = stem_dict
        # TODO: Validate stem_dict is reduced
        super().__init__(vector_size=vector_size, count=count, dtype=dtype, mapfile_path=mapfile_path)

    def __getitem__(self, item):
        stem = self.stem_dict.get(item, item)
        return super().__getitem__(stem)

    @classmethod
    def from_keyed_vectors(cls, stem_dict: dict, kv: KeyedVectors) -> "StemmedKeyedVectors":
        model = cls(stem_dict, vector_size=kv.vector_size)
        # This is pretty ugly, but it's working.
        for key, value in kv.__dict__.items():
            model.__dict__[key] = value
        log.debug("StemmedKeyedVectors loaded.")
        return model

    @classmethod
    def load(cls, fname: str, mmap=None):
        kv: KeyedVectors = super().load(fname=fname, mmap=mmap)  # type: ignore
        stem_dict_path = get_stem_dict_path_from_model_path(fname)
        try:
            with open(stem_dict_path) as file:
                stem_dict = json.load(file)
        except IOError as e:
            raise StemDictFileNotFoundError() from e
        return StemmedKeyedVectors.from_keyed_vectors(stem_dict=stem_dict, kv=kv)

    def save(self, fname: str, *args, **kwargs):
        super().save(fname, *args, **kwargs)
        save_stem_dict(self.stem_dict, fname)

    def similarity_unseen_docs(self, *args, **kwargs):
        # This is here just so that pycharm won't mark this class as abstract.
        return super().similarity_unseen_docs(*args, **kwargs)
