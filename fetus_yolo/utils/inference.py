"""
Inference utilities for YOLO models.
"""
from typing import Tuple, List, Union
import numpy as np
import cv2
from .constants import DEFAULT_LETTERBOX_COLOR, EPSILON


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = DEFAULT_LETTERBOX_COLOR,
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
) -> Tuple[np.ndarray, Union[float, Tuple[float, float]], Tuple[int, int]]:
    if img is None or img.size == 0:
        raise ValueError("Input image is None or empty")
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {h}x{w}")
    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / float(w), new_shape[0] / float(h))
    dw /= 2; dh /= 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    left = int(np.floor(dw)); right = int(np.ceil(dw))
    top = int(np.floor(dh)); bottom = int(np.ceil(dh))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    if boxes is None or scores is None:
        raise ValueError("boxes and scores cannot be None")
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("boxes and scores length mismatch")
    if boxes.shape[1] != 4:
        raise ValueError(f"boxes must be (N,4), got {boxes.shape}")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0,1]")
    boxes = boxes.astype(np.float32); scores = scores.astype(np.float32)
    order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_rest - inter + EPSILON
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


def reverse_letterbox(boxes: np.ndarray, original_shape: Tuple[int, int], ratio: Union[float, Tuple[float, float]], padding: Tuple[int, int]) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    pad_left, pad_top = padding
    orig_h, orig_w = original_shape
    if isinstance(ratio, tuple):
        rx, ry = float(ratio[0]), float(ratio[1])
    else:
        rx = ry = float(ratio)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / max(rx, EPSILON)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / max(ry, EPSILON)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
    return boxes


def decode_dets(det: np.ndarray, conf: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if det is None or det.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    det = np.asarray(det)
    if det.shape[-1] == 6:
        boxes = det[:, :4].astype(np.float32)
        scores = det[:, 4].astype(np.float32)
        clses = det[:, 5].astype(np.int32)
        m = scores > float(conf)
        return boxes[m], scores[m], clses[m]
    xywh = det[:, 0:4].astype(np.float32)
    obj = det[:, 4].astype(np.float32)
    cls_scores = det[:, 5:].astype(np.float32)
    if cls_scores.size == 0:
        scores = obj
        clses = np.zeros_like(obj, dtype=np.int32)
    else:
        best = cls_scores.argmax(axis=1); best_p = cls_scores.max(axis=1)
        scores = obj * best_p; clses = best.astype(np.int32)
    x, y, w, h = xywh.T
    x1 = x - w / 2.0; y1 = y - h / 2.0
    x2 = x + w / 2.0; y2 = y + h / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    m = scores > float(conf)
    return boxes[m], scores[m], clses[m]
