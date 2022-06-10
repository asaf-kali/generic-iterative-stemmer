import logging
import os
from typing import Any, Callable

from dynaconf import Dynaconf
from dynaconf.base import LazySettings


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
        log = logging.getLogger(__name__)
        log.info("Loading settings")
        settings_files = ["settings.toml", "local.toml"]
        cls._settings = Dynaconf(environments=True, settings_files=settings_files, **os.environ)

    @classmethod
    def get(cls, name: str, default: Any = None) -> Any:
        return getattr(cls.cache, name, default)

    @classproperty
    def cache(cls) -> LazySettings:
        if cls._settings is None:
            cls.reload()
        return cls._settings

    @classproperty
    def is_debug(cls) -> bool:
        return bool(cls.get("DEBUG", default=False))

    @classproperty
    def data_folder(cls) -> str:
        rel_path = cls.get("DATA_FOLDER_PATH", default="~/.cache/language_data")
        abs_path = absolute_path(rel_path)
        return abs_path


def absolute_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
