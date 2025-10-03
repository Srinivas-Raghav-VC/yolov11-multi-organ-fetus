"""
Safe I/O utilities for file operations.
"""
from pathlib import Path
from typing import Union
import cv2
import numpy as np

from .logging_config import get_logger
from .validation import validate_path, validate_image

logger = get_logger(__name__)


class FileOperationError(Exception):
    pass


def safe_read_image(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    try:
        path = validate_path(path, must_exist=True, must_be_file=True)
    except Exception as e:
        raise FileOperationError(f"Invalid image path: {e}")
    try:
        img = cv2.imread(str(path), flags)
    except Exception as e:
        raise FileOperationError(f"Failed to read image {path}: {e}")
    if img is None:
        raise FileOperationError(f"Failed to read image (returned None): {path}")
    try:
        validate_image(img)
    except Exception as e:
        raise FileOperationError(f"Invalid image {path}: {e}")
    return img
