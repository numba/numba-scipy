import logging

logger = logging.getLogger("numba_scipy.special")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Change to logging.DEBUG for more verbosity

from . import overloads

_overloads.add_overloads()

def set_verbose_logging(enabled=True):
    level = logging.DEBUG if enabled else logging.INFO
    logger.setLevel(level)
    logger.info(f"Verbose logging {'enabled' if enabled else 'disabled'}")
