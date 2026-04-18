"""
log_setup.py - Central logging initialiser for tensorized_minhash.

Usage in entry-point scripts (main.py, demo workers, demo genome scripts):
    from log_setup import setup_logging    # when tensorized_minhash/ is on sys.path
    # or -
    from tensorized_minhash.log_setup import setup_logging  # when /app is on sys.path

setup_logging()    # INFO level, reads logging.conf
setup_logging(level=logging.DEBUG)    # override root level at runtime

Library modules (core/, data/, benchmarks/, spark/) do NOT call this.
They only do:

import logging
logger = logging.getLogger(__name__)
"""

import logging
import logging.config
from pathlib import Path

_CONF_PATH = Path(__file__).parent / "logging.conf"


def setup_logging(level: int | None = None) -> None:
    """
    Configure all loggers from logging.conf.

    Safe to call multiple times - fileConfig with disable_existing_loggers=False
    will not silently discard loggers that were already created before this call.

    Args:
        level: Optional override for the root logger level after loading the
        config (e.g. `logging.DEBUG` to turn on verbose output).
    """
    logging.config.fileConfig(_CONF_PATH, disable_existing_loggers=False)
    if level is not None:
        logging.getLogger().setLevel(level)