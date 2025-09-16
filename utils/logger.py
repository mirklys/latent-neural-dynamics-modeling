import logging
from pathlib import Path
from datetime import datetime as dt
from typing import Callable, Any
from functools import wraps

class Logger:
    def __init__(self, log_dir: str, name: str = "application"):
        self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        log_file_name = f"{name}_{dt.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_file = self.log_dir / log_file_name

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)


_LOGGER_INSTANCE = None

def setup_logger(log_dir: str, name: str = "application"):
    """Initializes the logger instance."""
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = Logger(log_dir, name)
    return _LOGGER_INSTANCE

def get_logger():
    """Returns the single logger instance."""
    if _LOGGER_INSTANCE is None:
        raise RuntimeError("Logger has not been initialized. Call setup_logger first.")
    return _LOGGER_INSTANCE
