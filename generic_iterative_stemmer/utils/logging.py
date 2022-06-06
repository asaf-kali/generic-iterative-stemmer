import json
import logging
import sys
from datetime import datetime, timedelta
from logging import Filter, Formatter, Logger, LogRecord
from logging.config import dictConfig
from time import time


class ContextLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = {}

    def _log(self, *args, **kwargs) -> None:
        extra = kwargs.get("extra")
        kwargs["extra"] = {"extra": extra, "context": self.context}
        super()._log(*args, **kwargs)  # noqa

    def update_context(self, data: dict):
        self.context.update(data)

    def set_context(self, data: dict):
        self.context = data

    def reset_context(self):
        self.set_context({})


class ExtraDataFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        message = super().format(record)
        extra = getattr(record, "extra", None)
        if extra:
            message += f" extra: {extra}"
        return message


class JsonFormatter(Formatter):
    def __init__(self, detailed: bool = False, pretty_json: bool = False):
        super().__init__()
        self.detailed = detailed
        self.indent = 2 if pretty_json else None
        self.tz = datetime.now().astimezone().tzinfo

    def format(self, record: LogRecord) -> str:
        data: dict = {
            "datetime": datetime.fromtimestamp(record.created, self.tz).isoformat(sep=" ", timespec="milliseconds"),
            "message": record.msg or record.message,
        }
        extra = getattr(record, "extra", None)
        if extra:
            data["extra"] = extra
        if self.detailed:
            data.update(
                {
                    "level": record.levelname,
                    "func_name": record.funcName,
                    "module": record.module,
                    "file_path": record.pathname,
                    "line_number": record.lineno,
                    "process": record.process,
                    "thread": record.thread,
                    "process_name": record.processName,
                    "thread_name": record.threadName,
                    "exc_info": record.exc_info,
                    "level_code": record.levelno,
                    "timestamp": record.created,
                    "context": getattr(record, "context", None),
                }
            )
        return json.dumps(data, indent=self.indent, ensure_ascii=False)


class LevelRangeFilter(Filter):
    def __init__(self, low=0, high=100):
        Filter.__init__(self)
        self.low = low
        self.high = high

    def filter(self, record):
        if self.low <= record.levelno < self.high:
            return True
        return False


def get_logger(name: str) -> ContextLogger:
    return logging.getLogger(name)  # type: ignore


log = get_logger(__name__)


def configure_logging(formatter: str = None, level: str = None, detailed_json: bool = True, pretty_json: bool = False):
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "class": "generic_iterative_stemmer.utils.ExtraDataFormatter",
                "format": "[%(asctime)s] %(message)s [%(name)s]",
                "datefmt": "%H:%M:%S",
            },
            "debug": {
                "class": "generic_iterative_stemmer.utils.ExtraDataFormatter",
                "format": "[%(asctime)s.%(msecs)03d] [%(levelname)-.4s]: %(message)s @@@ "
                "[%(threadName)s] [%(name)s:%(lineno)s]",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "generic_iterative_stemmer.utils.JsonFormatter",
                "detailed": detailed_json,
                "pretty_json": pretty_json,
            },
            "test": {
                "class": "generic_iterative_stemmer.utils.ExtraDataFormatter",
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
                # "stream": sys.stdout,
                "stream": sys.stderr,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "run.log",
                "formatter": "json",
            },
        },
        "root": {"handlers": ["console_out", "console_err", "file"], "level": level or "DEBUG"},
        "loggers": {
            "gensim": {"level": "INFO"},
            "smart_open": {"level": "WARN"},
            "matplotlib": {"propagate": False},
        },
    }
    logging.setLoggerClass(ContextLogger)
    dictConfig(config)
    log.debug("Logging configured")


class MeasureTime:
    def __init__(self):
        self.start = self.end = 0

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()

    @property
    def delta(self) -> float:
        return round(self.duration.total_seconds(), 3)

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.end - self.start)
