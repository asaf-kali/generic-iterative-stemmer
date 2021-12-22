import json
from typing import Optional

import numpy as np
from gensim.models import KeyedVectors

STEM_DICT_FILE_SUFFIX = "stem-dict.json"


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
        model = StemmedKeyedVectors(stem_dict, vector_size=kv.vector_size)
        # TODO: This is pretty ugly, but its working.
        for key, value in kv.__dict__.items():
            model.__dict__[key] = value
        return model

    @classmethod
    def load(cls, fname: str, mmap=None):
        kv: KeyedVectors = super().load(fname=fname, mmap=mmap)  # type: ignore
        stem_dict_path = f"{fname}.{STEM_DICT_FILE_SUFFIX}"
        with open(stem_dict_path) as file:
            stem_dict = json.load(file)
        return StemmedKeyedVectors.from_keyed_vectors(stem_dict=stem_dict, kv=kv)

    def similarity_unseen_docs(self, *args, **kwargs):
        # This is here just so that pycharm won't mark this class as abstract.
        return super().similarity_unseen_docs(*args, **kwargs)
