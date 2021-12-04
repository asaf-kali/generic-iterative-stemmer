import logging
import sys
from datetime import timedelta
from functools import wraps
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
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[%(name)s:%(lineno)s] %(message)s"},
        "debug": {
            "format": "[%(asctime)s.%(msecs)03d] [%(levelname)-.4s]: %(message)s @@@ "
                      "[%(threadName)s] [%(name)s:%(lineno)s]",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "filters": {
        "std_filter": {"()": "utils.logging.LevelRangeFilter", "high": logging.WARNING},
        "err_filter": {"()": "utils.logging.LevelRangeFilter", "low": logging.WARNING},
    },
    "handlers": {
        "console_out": {
            "class": "logging.StreamHandler",
            "filters": ["std_filter"],
            "formatter": "simple",
            "stream": sys.stdout,
        },
        "console_err": {
            "class": "logging.StreamHandler",
            "filters": ["err_filter"],
            "formatter": "debug",
            "stream": sys.stderr,
        },
    },
    "root": {"handlers": ["console_out", "console_err"], "level": "DEBUG"},
    "loggers": {
        "gensim": {"level": "WARN"},
        "smart_open": {"level": "WARN"}
    },
}


def configure_logging():
    dictConfig(LOGGING_CONFIG)
    log.debug("Logging configured")


def measure_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        finish = time()
        delta = timedelta(seconds=finish - start)
        log.info(f"Function '{f.__name__}' took {delta}.")
        return result

    return wrapper
