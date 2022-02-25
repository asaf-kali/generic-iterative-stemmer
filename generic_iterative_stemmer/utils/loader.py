import os

from gensim.models import KeyedVectors, Word2Vec, fasttext

from ..utils import get_logger
from . import Settings

log = get_logger(__name__)


def load_kv(kv_path: str) -> KeyedVectors:
    model: KeyedVectors = KeyedVectors.load(kv_path)  # type: ignore
    log.info("Vectors loaded")
    return model


def load_w2v_model(model_path: str) -> Word2Vec:
    model = Word2Vec.load(model_path)
    log.info("word2vec model loaded")
    return model


def load_ft_model(model_path: str):
    model = fasttext.load_facebook_model(model_path)
    log.info("fasttext model loaded")
    return model


def get_path(*path: str) -> str:
    return os.path.join(Settings.data_folder, *path)
