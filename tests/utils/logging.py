import logging
from logging.config import dictConfig

from the_spymaster_util.logger import get_dict_config


def configure_logging(std_formatter: str = None):
    handlers = {
        "training_file": {
            "class": "logging.FileHandler",
            "filename": "train.log",
            "formatter": "json",
        },
    }
    loggers = {
        "gensim": {"level": "INFO"},
        "smart_open": {"level": "WARN"},
        "matplotlib": {"propagate": False},
    }
    dict_config = get_dict_config(
        std_formatter=std_formatter,
        extra_handlers=handlers,
        extra_loggers=loggers,
    )
    dictConfig(dict_config)
    logging.getLogger(__name__).debug("Logging configured")
