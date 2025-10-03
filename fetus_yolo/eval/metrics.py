from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class EvaluationError(Exception):
    pass


def extract_per_class_ap(res, num_classes: int) -> Optional[List[float]]:
    """Prefer Ultralytics metrics.box.maps (per-class mAP50-95).
    Fallback to per-class ap50/ap arrays if present. Returns None if unavailable.
    """
    try:
        box = getattr(res, 'box', None)
        if box is not None:
            maps = getattr(box, 'maps', None)
            if isinstance(maps, (list, tuple, np.ndarray)) and len(maps) == num_classes:
                return [float(v) for v in np.array(maps).flatten()]
            for attr in ('ap50', 'ap'):
                vals = getattr(box, attr, None)
                if isinstance(vals, (list, tuple, np.ndarray)) and len(vals) == num_classes:
                    return [float(v) for v in np.array(vals).flatten()]
        return None
    except Exception:
        return None


def extract_aggregate_metrics(res) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    results_dict = getattr(res, 'results_dict', None)
    if results_dict:
        for key in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']:
            if key in results_dict:
                try:
                    metrics[key] = float(results_dict[key])
                except Exception:
                    pass
    box = getattr(res, 'box', None)
    if box:
        for attr, name in [('map50', 'mAP50'), ('map', 'mAP50-95'), ('mp', 'precision'), ('mr', 'recall')]:
            val = getattr(box, attr, None)
            if val is not None:
                try:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        val = float(np.mean(val))
                    metrics[name] = float(val)
                except Exception:
                    pass
    return metrics


def compute_confusion_matrix(model, data_yaml: Path, conf: float = 0.25):
    res = model.val(data=str(data_yaml), conf=conf, plots=False)
    cm = getattr(res, 'confusion_matrix', None)
    if cm is not None and getattr(cm, 'matrix', None) is not None:
        return np.array(cm.matrix)
    return None


def evaluate_with_coco(pred_json: Path, gt_json: Path, num_classes: int = 4) -> Dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        raise EvaluationError(f"pycocotools not available: {e}")

    if not pred_json.exists() or not gt_json.exists():
        raise EvaluationError("Prediction or GT JSON not found for COCO eval")

    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    results = {
        'mAP50-95': float(coco_eval.stats[0]),
        'mAP50': float(coco_eval.stats[1]),
        'mAP75': float(coco_eval.stats[2]),
        'mAP_small': float(coco_eval.stats[3]),
        'mAP_medium': float(coco_eval.stats[4]),
        'mAP_large': float(coco_eval.stats[5]),
    }

    per_class_ap50 = []
    for cid in range(num_classes):
        e = COCOeval(coco_gt, coco_dt, 'bbox')
        e.params.catIds = [cid]
        e.evaluate(); e.accumulate()
        per_class_ap50.append(float(e.stats[1]))
    results['per_class_ap50'] = per_class_ap50
    return results
