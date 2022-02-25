import os
from typing import Any, Callable

from dynaconf import Dynaconf
from dynaconf.base import LazySettings

from generic_iterative_stemmer.utils import get_logger


class classproperty:  # noqa
    """@classmethod + @property"""

    def __init__(self, f: Callable):
        self.f = classmethod(f)

    def __get__(self, *args):
        return self.f.__get__(*args)()  # type: ignore


class Settings:
    _settings: LazySettings = None

    @classmethod
    def reload(cls):
        log = get_logger(__name__)
        log.info("Loading settings")
        cls._settings = Dynaconf(**os.environ)

    @classmethod
    def get_attribute(cls, name: str, default: Any = None) -> Any:
        return getattr(cls.cache, name, default)

    @classproperty
    def cache(cls) -> LazySettings:
        if cls._settings is None:
            cls.reload()
        return cls._settings

    @classproperty
    def is_debug(cls) -> bool:
        return bool(cls.get_attribute("DEBUG", default=False))

    @classproperty
    def data_folder(cls) -> str:
        return cls.get_attribute("DATA_FOLDER_PATH", default="./data")
