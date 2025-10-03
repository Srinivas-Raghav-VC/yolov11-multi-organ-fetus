"""
Centralized logging configuration for fetus-yolo.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "fetus-yolo",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    formatter = logging.Formatter(format_string)
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level); ch.setFormatter(formatter)
        logger.addHandler(ch)
    if log_file is not None:
        log_file = Path(log_file); log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        fh.setLevel(level); fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def get_logger(name: str = "fetus-yolo") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
