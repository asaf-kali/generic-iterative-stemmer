from typing import Optional

import numpy as np
from gensim.models import KeyedVectors


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
        item = self.stem_dict.get(item, item)
        return super().__getitem__(item)

    def similarity_unseen_docs(self, *args, **kwargs):
        # This is here just so that pycharm won't mark this class as abstract.
        return super().similarity_unseen_docs(*args, **kwargs)
