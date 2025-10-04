"""
Train RF-DETR on this project's YOLO-format dataset.

Steps:
- Validate dataset YAML and integrity (uses existing dataset_utils).
- Convert YOLO labels to COCO JSON per split (train/valid/test).
- Launch rfdetr training (Base/Nano/Small/Medium) with configurable params.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import shutil
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fetus_yolo.data.dataset_utils import (
    EXPECTED_CLASS_NAMES,
    DatasetInfo,
    DatasetValidationError,
    ensure_dataset_ready,
    gather_files,
    img_to_label_path,
    check_label_file,
)


def _yolo_to_coco_bbox(x: float, y: float, w: float, h: float, iw: int, ih: int) -> Tuple[float, float, float, float]:
    cx = x * iw
    cy = y * ih
    bw = w * iw
    bh = h * ih
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x1 = float(min(x1, iw - 1))
    y1 = float(min(y1, ih - 1))
    bw = float(min(bw, iw - x1))
    bh = float(min(bh, ih - y1))
    return [x1, y1, bw, bh]


def convert_yolo_to_coco(info: DatasetInfo, out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    # Map dataset splits to RF-DETR expected names
    split_map = {"train": "train", "val": "valid", "test": "test"}

    categories = []
    for cid, name in sorted(info.names.items()):
        categories.append({"id": int(cid), "name": str(name)})

    ann_id = 1
    img_id = 1

    for split, (img_dir, lbl_dir) in info.splits.items():
        if split not in split_map:
            continue
        target_split = split_map[split]
        dst_dir = out_root / target_split
        dst_dir.mkdir(parents=True, exist_ok=True)
        images: List[Dict] = []
        annotations: List[Dict] = []

        img_files, _ = gather_files(img_dir, lbl_dir)
        for img_path in img_files:
            # copy to split folder for relative file_name portability
            dst_img = dst_dir / img_path.name
            if not dst_img.exists():
                try:
                    shutil.copy2(str(img_path), str(dst_img))
                except Exception:
                    img_tmp = cv2.imread(str(img_path))
                    if img_tmp is not None:
                        cv2.imwrite(str(dst_img), img_tmp)

            img = cv2.imread(str(dst_img))
            if img is None:
                continue
            ih, iw = int(img.shape[0]), int(img.shape[1])
            images.append({
                "id": img_id,
                "file_name": dst_img.name,
                "width": iw,
                "height": ih,
            })

            lbl_path = img_to_label_path(img_path, img_dir, lbl_dir)
            if lbl_path.exists():
                _, lines = check_label_file(lbl_path, class_count=len(info.names))
                for cls, x, y, w, h in lines:
                    bbox = _yolo_to_coco_bbox(x, y, w, h, iw, ih)
                    area = float(bbox[2] * bbox[3])
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [float(v) for v in bbox],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1
            img_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        (dst_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    return out_root


def get_model(size: str):
    try:
        from rfdetr import RFDETRBase as _Base
    except Exception as e:  # pragma: no cover - defensive
        raise SystemExit(f"rfdetr not available: {e}")

    # Attempt to resolve size-specific classes; fallback to Base
    model_cls = _Base
    if size.lower() != "base":
        try:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
            mapping = {
                "nano": RFDETRNano,
                "small": RFDETRSmall,
                "medium": RFDETRMedium,
            }
            model_cls = mapping.get(size.lower(), _Base)
        except Exception:
            model_cls = _Base
    return model_cls()


def train_rfdetr(coco_dir: Path, out_dir: Path, size: str, epochs: int, batch_size: int,
                 grad_accum_steps: int, lr: float, resolution: int, device: str | None):
    model = get_model(size)
    kw = {
        "dataset_dir": str(coco_dir),
        "output_dir": str(out_dir),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "lr": float(lr),
        "resolution": int(resolution),
    }
    if device:
        kw["device"] = device
    model.train(**kw)


def main():
    ap = argparse.ArgumentParser(description="Train RF-DETR on YOLO-format dataset")
    ap.add_argument("--data", default="fetus-yolo/data_yolo/fetal_fp23.yaml")
    ap.add_argument("--size", choices=["base", "nano", "small", "medium"], default="base")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--coco-dir", default="datasets/coco_rfdetr")
    ap.add_argument("--out", default="runs/rfdetr")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = (REPO_ROOT / data_yaml).resolve()

    try:
        info, reports = ensure_dataset_ready(data_yaml, REPO_ROOT, run_integrity=True)
        if reports:
            from fetus_yolo.data.dataset_utils import format_reports
            print(format_reports(reports))
    except DatasetValidationError as exc:
        raise SystemExit(str(exc))

    coco_dir = convert_yolo_to_coco(info, Path(args.coco_dir).resolve())
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rfdetr(
        coco_dir=coco_dir,
        out_dir=out_dir,
        size=args.size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resolution=args.resolution,
        device=args.device,
    )

    print(f"RF-DETR training finished. Outputs in: {out_dir}")
    print(f"COCO dataset written to: {coco_dir}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
