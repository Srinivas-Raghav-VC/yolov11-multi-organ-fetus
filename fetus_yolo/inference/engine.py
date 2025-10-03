from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fetus_yolo.utils.inference import letterbox, nms, reverse_letterbox, decode_dets
from fetus_yolo.utils import constants


@dataclass
class Detections:
    boxes: np.ndarray  # (N,4) xyxy
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,)
    time_ms: float


class InferenceEngine:
    def __init__(
        self,
        onnx: Optional[Path] = None,
        engine: Optional[Path] = None,
        providers: Optional[Sequence[str]] = None,
        names: Optional[Sequence[str]] = None,
        device: str = "auto",
    ) -> None:
        self.names = list(names) if names else list(constants.CLASS_NAMES)
        self.device = device
        self._is_onnx = False
        self._is_trt = False
        self._sess = None
        self._inp = None
        self._out = None
        self._trt_model = None

        if engine is not None:
            from ultralytics import YOLO as ULYOLO

            self._trt_model = ULYOLO(str(engine))
            self._is_trt = True
            return

        if onnx is None:
            raise ValueError("Provide either onnx or engine path")
        import onnxruntime as ort

        avail = []
        try:
            avail = ort.get_available_providers()
        except Exception:
            avail = []
        pr = providers or constants.ONNX_PROVIDERS
        pr = [p for p in pr if p in avail]
        self._sess = ort.InferenceSession(str(Path(onnx)), providers=pr or None)
        self._inp = self._sess.get_inputs()[0].name
        self._out = self._sess.get_outputs()[0].name
        self._is_onnx = True

    def _run_onnx(self, img_bgr: np.ndarray, imgsz: int, conf: float, iou: float) -> Detections:
        img, r, (pad_left, pad_top) = letterbox(img_bgr, (imgsz, imgsz))
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
        x = x[None]
        t0 = time.time()
        pred = self._sess.run([self._out], {self._inp: x})[0]
        dt = (time.time() - t0) * 1000.0
        det = pred[0] if pred.ndim == 3 else pred
        boxes, scores, clses = decode_dets(det, conf)
        if boxes.size:
            boxes = reverse_letterbox(boxes, img_bgr.shape[:2], r, (pad_left, pad_top))
            keep = nms(boxes, scores, iou)
            boxes, scores, clses = boxes[keep], scores[keep], clses[keep]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            clses = np.zeros((0,), dtype=np.int32)
        return Detections(boxes, scores, clses, dt)

    def _run_trt(self, img_bgr: np.ndarray, imgsz: int, conf: float, iou: float) -> Detections:
        t0 = time.time()
        res = self._trt_model.predict(source=img_bgr, conf=float(conf), iou=float(iou), verbose=False, device=self.device)[0]
        dt = (time.time() - t0) * 1000.0
        if res is not None and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = res.boxes.conf.cpu().numpy().astype(np.float32)
            clses = res.boxes.cls.cpu().numpy().astype(np.int32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            clses = np.zeros((0,), dtype=np.int32)
        return Detections(boxes, scores, clses, dt)

    def infer_global(self, img_bgr: np.ndarray, imgsz: int, conf: float, iou: float) -> Detections:
        if self._is_trt:
            return self._run_trt(img_bgr, imgsz, conf, iou)
        return self._run_onnx(img_bgr, imgsz, conf, iou)

    def infer_tta(
        self,
        img_bgr: np.ndarray,
        imgsz: int,
        conf: float,
        iou: float,
        scales: Sequence[float] = (1.0, 0.83),
        flips: Sequence[str] = ("none", "h"),
    ) -> Detections:
        H, W = img_bgr.shape[:2]
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_clses: List[np.ndarray] = []
        t0 = time.time()
        for s in scales:
            resized = cv2.resize(img_bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR)
            for flip in flips:
                img = resized
                if flip == "h":
                    img = cv2.flip(img, 1)
                elif flip == "v":
                    img = cv2.flip(img, 0)
                det = self.infer_global(img, imgsz, conf, iou)
                boxes, scores, clses = det.boxes, det.scores, det.classes
                if boxes.size:
                    if flip == "h":
                        boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]
                    if flip == "v":
                        boxes[:, [1, 3]] = img.shape[0] - boxes[:, [3, 1]]
                    if s != 1.0:
                        boxes[:, [0, 2]] /= s
                        boxes[:, [1, 3]] /= s
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_clses.append(clses)
        if all_boxes:
            all_boxes = np.concatenate(all_boxes, 0)
            all_scores = np.concatenate(all_scores, 0)
            all_clses = np.concatenate(all_clses, 0)
            keep = nms(all_boxes, all_scores, iou)
            all_boxes, all_scores, all_clses = all_boxes[keep], all_scores[keep], all_clses[keep]
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_scores = np.zeros((0,), dtype=np.float32)
            all_clses = np.zeros((0,), dtype=np.int32)
        dt = (time.time() - t0) * 1000.0
        return Detections(all_boxes, all_scores, all_clses, dt)

    @staticmethod
    def _gen_tiles(h: int, w: int, tile: int, overlap: float):
        step = max(1, int(round(tile * (1.0 - overlap))))
        xs = list(range(0, max(1, w - tile + 1), step))
        ys = list(range(0, max(1, h - tile + 1), step))
        if not xs or xs[-1] != max(0, w - tile):
            xs.append(max(0, w - tile))
        if not ys or ys[-1] != max(0, h - tile):
            ys.append(max(0, h - tile))
        for y in ys:
            for x in xs:
                y2 = min(h, y + tile)
                x2 = min(w, x + tile)
                yield y, x, y2, x2

    def infer_tiled(
        self,
        img_bgr: np.ndarray,
        imgsz: int,
        tile: int,
        overlap: float,
        conf: float,
        iou: float,
    ) -> Detections:
        H, W = img_bgr.shape[:2]
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_clses: List[np.ndarray] = []
        t0 = time.time()
        for (y1, x1, y2, x2) in self._gen_tiles(H, W, tile, overlap):
            tile_img = img_bgr[y1:y2, x1:x2]
            det = self.infer_global(tile_img, imgsz, conf, iou)
            boxes, scores, clses = det.boxes, det.scores, det.classes
            if not boxes.size:
                continue
            # Clip, shift to global
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, x2 - x1 - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, y2 - y1 - 1)
            boxes[:, [0, 2]] += x1
            boxes[:, [1, 3]] += y1
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_clses.append(clses)
        if all_boxes:
            all_boxes = np.concatenate(all_boxes, 0)
            all_scores = np.concatenate(all_scores, 0)
            all_clses = np.concatenate(all_clses, 0)
            keep = nms(all_boxes, all_scores, iou)
            all_boxes, all_scores, all_clses = all_boxes[keep], all_scores[keep], all_clses[keep]
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_scores = np.zeros((0,), dtype=np.float32)
            all_clses = np.zeros((0,), dtype=np.int32)
        dt = (time.time() - t0) * 1000.0
        return Detections(all_boxes, all_scores, all_clses, dt)

    def infer_adaptive(
        self,
        img_bgr: np.ndarray,
        imgsz: int,
        tile: int,
        overlap: float,
        conf: float,
        iou: float,
        min_det: int = 1,
        tiny_classes: Sequence[int] = (2, 3),
        min_tiny: int = 1,
        hist: int = 30,
        class_thresholds: Optional[Sequence[float]] = None,
        _tiny_hist: Optional[List[int]] = None,
    ) -> Tuple[Detections, List[int]]:
        det_g = self.infer_global(img_bgr, imgsz, conf, iou)
        tiny_now = int(np.sum(np.isin(det_g.classes, np.array(tiny_classes, dtype=int))))
        tiny_hist = _tiny_hist or []
        tiny_hist.append(tiny_now)
        if len(tiny_hist) > hist:
            tiny_hist.pop(0)
        tiny_mean = float(np.mean(tiny_hist)) if tiny_hist else 0.0

        need_tile = False
        if max(img_bgr.shape[:2]) > tile:
            if len(det_g.boxes) < min_det:
                need_tile = True
            elif tiny_mean < min_tiny:
                need_tile = True
            elif class_thresholds is not None and len(class_thresholds) > 0:
                for c in tiny_classes:
                    c = int(c)
                    if c < len(class_thresholds):
                        thr = float(class_thresholds[c])
                        c_scores = det_g.scores[det_g.classes == c]
                        mean_s = float(np.mean(c_scores)) if c_scores.size else 0.0
                        if mean_s < thr:
                            need_tile = True
                            break

        if not need_tile:
            return det_g, tiny_hist

        det_t = self.infer_tiled(img_bgr, imgsz, tile, overlap, conf, iou)
        if det_g.boxes.size and det_t.boxes.size:
            mb = np.concatenate([det_g.boxes, det_t.boxes], 0)
            ms = np.concatenate([det_g.scores, det_t.scores], 0)
            mc = np.concatenate([det_g.classes, det_t.classes], 0)
            keep = nms(mb, ms, iou)
            mb, ms, mc = mb[keep], ms[keep], mc[keep]
            return Detections(mb, ms, mc, det_g.time_ms + det_t.time_ms), tiny_hist
        return (det_t if det_t.boxes.size else det_g), tiny_hist
