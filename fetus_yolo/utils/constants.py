"""
Shared constants for the fetus-yolo project.

Centralized single source of truth.
"""
from typing import Sequence, Tuple

CLASS_NAMES: Sequence[str] = ("abdomen", "head", "arm", "legs")

CLASS_ID_ABDOMEN = 0
CLASS_ID_HEAD = 1
CLASS_ID_ARM = 2
CLASS_ID_LEGS = 3

TINY_CLASS_IDS = (CLASS_ID_ARM, CLASS_ID_LEGS)

ONNX_PROVIDERS = [
    'TensorrtExecutionProvider',
    'CUDAExecutionProvider',
    'DmlExecutionProvider',
    'OpenVINOExecutionProvider',
    'CoreMLExecutionProvider',
    'CPUExecutionProvider',
]

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
MODEL_EXTENSIONS = ('.pt', '.pth', '.onnx', '.engine', '.tflite')

DEFAULT_LETTERBOX_COLOR: Tuple[int, int, int] = (114, 114, 114)
EPSILON = 1e-6
MIN_BOX_SIZE = 1e-6
MAX_COORD = 1.0
MIN_COORD = 0.0

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_IMG_SIZE = 896
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 0.25

DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(message)s"
)
