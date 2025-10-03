"""
Input validation utilities (paths, images, numeric ranges).
"""
from pathlib import Path
from typing import Union, List, Optional, Sequence
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
MAX_PATH_LENGTH = 4096
MAX_IMAGE_SIZE = 10000


class ValidationError(Exception):
    pass


def validate_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allowed_parents: Optional[List[Path]] = None,
    create_if_missing: bool = False,
) -> Path:
    if path is None:
        raise ValidationError("Path cannot be None")
    path = Path(path)
    path_str = str(path)
    if len(path_str) > MAX_PATH_LENGTH:
        raise ValidationError("Path too long")
    if any(c in path_str for c in ('\x00', '\r', '\n')):
        raise ValidationError("Path contains suspicious characters")
    try:
        path = path.resolve()
    except Exception as e:
        raise ValidationError(f"Cannot resolve path {path}: {e}")
    if allowed_parents:
        allowed_parents = [Path(p).resolve() for p in allowed_parents]
        def _is_relative_to(child: Path, parent: Path) -> bool:
            try:
                child.relative_to(parent)
                return True
            except Exception:
                return False
        if not any(_is_relative_to(path, p) for p in allowed_parents):
            raise ValidationError("Path outside allowed directories")
    if must_exist and not path.exists():
        raise ValidationError(f"Path does not exist: {path}")
    if path.exists():
        if must_be_file and not path.is_file():
            raise ValidationError(f"Not a file: {path}")
        if must_be_dir and not path.is_dir():
            raise ValidationError(f"Not a directory: {path}")
    elif create_if_missing and must_be_dir:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    return path


def validate_image(image: np.ndarray, min_size: int = 1, max_size: int = MAX_IMAGE_SIZE, channels: Optional[List[int]] = None) -> None:
    if image is None:
        raise ValidationError("Image is None")
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in [2, 3]:
        raise ValidationError("Invalid image array")
    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        raise ValidationError("Image too small")
    if h > max_size or w > max_size:
        raise ValidationError("Image too large")
    if channels is not None:
        c = 1 if image.ndim == 2 else image.shape[2]
        if c not in channels:
            raise ValidationError("Unexpected channel count")


def validate_model_weights(weights_path: Union[str, Path], expected_extensions: Sequence[str] = ('.pt', '.pth', '.onnx', '.engine')) -> Path:
    path = validate_path(weights_path, must_exist=True, must_be_file=True)
    if path.suffix.lower() not in expected_extensions:
        raise ValidationError("Invalid model extension")
    if path.stat().st_size == 0:
        raise ValidationError("Weights file is empty")
    return path


def validate_class_names(names: Union[List[str], dict], expected_count: Optional[int] = None, expected_names: Optional[Sequence[str]] = None) -> dict:
    if isinstance(names, list):
        names_dict = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        names_dict = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValidationError("names must be list or dict")
    if expected_count is not None and len(names_dict) != expected_count:
        raise ValidationError("Unexpected class count")
    if expected_names is not None and set(names_dict.values()) != set(expected_names):
        raise ValidationError("Class names mismatch")
    if len(set(names_dict.values())) != len(names_dict):
        raise ValidationError("Duplicate class names")
    if set(names_dict.keys()) != set(range(len(names_dict))):
        raise ValidationError("Class IDs must be consecutive from 0")
    return names_dict


def validate_numeric_range(value: Union[int, float], name: str, min_val: Optional[Union[int, float]] = None, max_val: Optional[Union[int, float]] = None, allow_negative: bool = True) -> None:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"{name} must be numeric")
    if not allow_negative and value < 0:
        raise ValidationError(f"{name} cannot be negative")
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}")
