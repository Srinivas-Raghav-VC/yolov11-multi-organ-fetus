"""Fetus-YOLO Python package.

This package provides a clean, importable API for dataset utilities,
training, inference, and evaluation.
"""

from .data.dataset_utils import (
    DatasetValidationError,
    validate_dataset_yaml,
    inspect_dataset,
    ensure_dataset_ready,
    format_reports,
)

__all__ = [
    "DatasetValidationError",
    "validate_dataset_yaml",
    "inspect_dataset",
    "ensure_dataset_ready",
    "format_reports",
]

__version__ = "0.1.0"
