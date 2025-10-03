"""
Unified inference CLI for ONNX/TensorRT with modes: global, tta, tiled, adaptive.
"""
import argparse
from pathlib import Path
import time
import sys

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fetus_yolo.inference import InferenceEngine
from fetus_yolo.utils import constants


def draw(img, boxes, scores, clses, names):
    for (x1, y1, x2, y2), c, s in zip(boxes, clses, scores):
        color = (0, 255, 0) if int(c) != 3 else (0, 165, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{names[int(c)]} {float(s):.2f}",
                    (int(x1), max(0, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--onnx', type=str)
    grp.add_argument('--engine', type=str)
    ap.add_argument('--source', default='0', help='0 webcam, or path to image/video/folder')
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--tile', type=int, default=640)
    ap.add_argument('--overlap', type=float, default=0.25)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--mode', choices=['global', 'tta', 'tiled', 'adaptive'], default='adaptive')
    ap.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.83])
    ap.add_argument('--flips', type=str, nargs='+', default=['none', 'h'])
    ap.add_argument('--min-det', type=int, default=1)
    ap.add_argument('--tiny-classes', type=int, nargs='*', default=[2, 3])
    ap.add_argument('--min-tiny', type=int, default=1)
    ap.add_argument('--hist', type=int, default=30)
    ap.add_argument('--class-thresholds', type=float, nargs='*')
    ap.add_argument('--names', nargs='+', default=list(constants.CLASS_NAMES))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    args.conf = max(0.0, min(1.0, float(args.conf)))
    args.iou = max(0.0, min(1.0, float(args.iou)))
    args.overlap = max(0.0, min(0.99, float(args.overlap)))
    args.imgsz = max(32, int(args.imgsz))
    args.tile = max(1, int(args.tile))

    engine = InferenceEngine(
        onnx=Path(args.onnx) if args.onnx else None,
        engine=Path(args.engine) if args.engine else None,
        providers=constants.ONNX_PROVIDERS,
        names=args.names,
        device=args.device,
    )

    s = args.source
    tiny_hist = []

    def run_frame(frame):
        if args.mode == 'global':
            det = engine.infer_global(frame, args.imgsz, args.conf, args.iou)
            return det
        if args.mode == 'tta':
            return engine.infer_tta(frame, args.imgsz, args.conf, args.iou, args.scales, args.flips)
        if args.mode == 'tiled':
            return engine.infer_tiled(frame, args.imgsz, args.tile, args.overlap, args.conf, args.iou)
        det, tiny_hist_updated = engine.infer_adaptive(
            frame,
            args.imgsz,
            args.tile,
            args.overlap,
            args.conf,
            args.iou,
            args.min_det,
            tuple(args.tiny_classes),
            args.min_tiny,
            args.hist,
            args.class_thresholds,
            tiny_hist,
        )
        tiny_hist[:] = tiny_hist_updated
        return det

    cap = None
    paths = []
    if s == '0':
        cap = cv2.VideoCapture(0)
        get_frame = lambda: cap.read()[1]
    else:
        p = Path(s)
        if p.is_dir():
            exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            paths = sorted([x for x in p.rglob('*') if x.suffix.lower() in exts])
        elif p.is_file():
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                paths = [p]
            else:
                cap = cv2.VideoCapture(str(p))
                get_frame = lambda: cap.read()[1]
        else:
            raise SystemExit(f'Invalid source: {s}')

    if cap is not None:
        while True:
            frame = get_frame()
            if frame is None:
                break
            det = run_frame(frame)
            msg = f"{args.mode} {det.time_ms:.1f} ms"
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            frame = draw(frame, det.boxes, det.scores, det.classes, args.names)
            cv2.imshow('Detections', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"Warning: failed to read image: {p}")
                continue
            det = run_frame(img)
            print(f"{p.name}: det={len(det.boxes)} time={det.time_ms:.1f} ms")
            if args.show:
                img = draw(img, det.boxes, det.scores, det.classes, args.names)
                cv2.imshow('Detections', img)
                cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
