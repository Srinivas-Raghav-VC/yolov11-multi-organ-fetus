"""
Unified evaluation CLI (per-class, aggregate, confusion matrix, optional COCO eval).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fetus_yolo.eval import (
    extract_aggregate_metrics,
    extract_per_class_ap,
    evaluate_with_coco,
    compute_confusion_matrix,
    EvaluationError,
)


def main():
    ap = argparse.ArgumentParser(description='Unified evaluation')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--img', type=int, default=896)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--save-dir', default='results/eval')
    ap.add_argument('--num-classes', type=int, default=4)
    ap.add_argument('--coco', action='store_true', help='attempt COCO eval if files available')
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    res = model.val(data=args.data, imgsz=args.img, conf=args.conf, split='val', save_json=True, plots=True, device=args.device)

    # Aggregate
    agg = extract_aggregate_metrics(res)
    # Per-class
    per_class = extract_per_class_ap(res, args.num_classes)
    # Confusion matrix
    cm = compute_confusion_matrix(model, Path(args.data), args.conf)

    results = {'aggregate': agg}
    if per_class is not None:
        results['per_class_ap'] = per_class

    # Optional COCO
    if args.coco:
        try:
            pred_json = Path('runs/detect/val/predictions.json')
            gt_json = Path(args.data).parent / 'annotations.json'
            if pred_json.exists() and gt_json.exists():
                results['coco_eval'] = evaluate_with_coco(pred_json, gt_json, args.num_classes)
        except EvaluationError as e:
            print(f"COCO eval skipped: {e}")
        except Exception as e:
            print(f"COCO eval error: {e}")

    # Save JSON
    (save_dir / 'evaluation_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    # Save CSV summary
    rows = []
    row = {'metric': 'mAP50-95', 'value': agg.get('mAP50-95', agg.get('metrics/mAP50-95(B)', None))}
    rows.append(row)
    rows.append({'metric': 'mAP50', 'value': agg.get('mAP50', agg.get('metrics/mAP50(B)', None))})
    rows.append({'metric': 'precision', 'value': agg.get('precision', agg.get('metrics/precision(B)', None))})
    rows.append({'metric': 'recall', 'value': agg.get('recall', agg.get('metrics/recall(B)', None))})
    if per_class is not None:
        for i, v in enumerate(per_class):
            rows.append({'metric': f'class_{i}_ap', 'value': v})
    pd.DataFrame(rows).to_csv(save_dir / 'evaluation_summary.csv', index=False)

    # Confusion matrix save
    if cm is not None:
        npy_path = save_dir / 'confusion_matrix.npy'
        import numpy as np
        np.save(str(npy_path), cm)

    print(f"Saved evaluation to {save_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
