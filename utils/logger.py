import logging
from pathlib import Path
from datetime import datetime as dt
from typing import Callable, Any
from functools import wraps


class MarkdownFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        lines = message.split("\n")
        # Look for the separator line, which is typically something like "|---|---|"
        if len(lines) > 2 and all(
            part.startswith("---") or part == ""
            for part in lines[1].strip().split("|")
            if part
        ):
            return "\n".join([lines[0]] + lines[2:])
        return message


class Logger:
    def __init__(self, log_dir: str, name: str = "application"):
        self.log_dir = Path(log_dir).resolve()
        print(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file_name = f"{Path(name).stem}_{dt.now().strftime('%Y%m%d_%H%M%S')}.md"
        self.log_file = self.log_dir / log_file_name
        print(self.log_file)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(console_formatter)
        self.logger.addHandler(ch)

        file_formatter = MarkdownFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(file_formatter)
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
