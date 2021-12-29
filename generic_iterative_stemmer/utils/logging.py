import logging
import sys
from datetime import timedelta
from logging import Filter
from logging.config import dictConfig
from time import time


class LevelRangeFilter(Filter):
    def __init__(self, low=0, high=100):
        Filter.__init__(self)
        self.low = low
        self.high = high

    def filter(self, record):
        if self.low <= record.levelno < self.high:
            return True
        return False


log = logging.getLogger(__name__)


def get_logging_config(formatter: str = None, level: str = None) -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[%(asctime)s] %(message)s [%(name)s]",
                "datefmt": "%H:%M:%S",
            },
            "debug": {
                "format": "[%(asctime)s.%(msecs)03d] [%(levelname)-.4s]: %(message)s @@@ "
                "[%(threadName)s] [%(name)s:%(lineno)s]",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "test": {
                "format": "[%(asctime)s.%(msecs)03d] [%(levelname)-.4s]: %(message)s "
                "[%(threadName)s] [%(name)s:%(lineno)s]",
                "datefmt": "%H:%M:%S",
            },
        },
        "filters": {
            "std_filter": {"()": "generic_iterative_stemmer.utils.LevelRangeFilter", "high": logging.WARNING},
            "err_filter": {"()": "generic_iterative_stemmer.utils.LevelRangeFilter", "low": logging.WARNING},
        },
        "handlers": {
            "console_out": {
                "class": "logging.StreamHandler",
                "filters": ["std_filter"],
                "formatter": formatter or "simple",
                "stream": sys.stdout,
            },
            "console_err": {
                "class": "logging.StreamHandler",
                "filters": ["err_filter"],
                "formatter": formatter or "debug",
                "stream": sys.stderr,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "run.log",
                "formatter": formatter or "debug",
            },
        },
        "root": {"handlers": ["console_out", "console_err", "file"], "level": level or "DEBUG"},
        "loggers": {"gensim": {"level": "INFO"}, "smart_open": {"level": "WARN"}},
    }


def configure_logging(formatter: str = None, level: str = None):
    config = get_logging_config(formatter=formatter, level=level)
    dictConfig(config)
    log.debug("Logging configured")


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        finish = time()
        delta = timedelta(seconds=finish - start)
        log.info(f"Function '{func.__name__}' took {delta}.")
        return result

    return wrapper
