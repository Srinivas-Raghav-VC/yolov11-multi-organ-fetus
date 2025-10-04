"""
Unified sanity training CLI for YOLOv11 and RF-DETR.

Features:
- Validates YOLO-format dataset (existing dataset_utils) and converts to COCO for RF-DETR.
- Trains a quick sanity run for YOLOv11 (ultralytics) and/or RF-DETR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import shutil
import os
import re
import zipfile
import cv2
import yaml

try:
    import requests
except Exception:
    requests = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fetus_yolo.data.dataset_utils import (
    DatasetInfo,
    DatasetValidationError,
    ensure_dataset_ready,
    gather_files,
    img_to_label_path,
    check_label_file,
)


def _yolo_to_coco_bbox(x: float, y: float, w: float, h: float, iw: int, ih: int):
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
    split_map = {"train": "train", "val": "valid", "test": "test"}
    categories = [{"id": int(cid), "name": str(name)} for cid, name in sorted(info.names.items())]
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
            # copy image to COCO split folder for portability (relative file_name)
            dst_img = dst_dir / img_path.name
            if not dst_img.exists():
                try:
                    shutil.copy2(str(img_path), str(dst_img))
                except Exception:
                    # fallback to cv2 imread/write if copy fails
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
        coco = {"images": images, "annotations": annotations, "categories": categories}
        (dst_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return out_root


def _drive_file_id(url: str) -> str | None:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)/", url)
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None


def _gdrive_download(url: str, out_path: Path):
    if requests is None:
        raise SystemExit("requests not installed. Run: pip install requests")
    file_id = _drive_file_id(url)
    if not file_id:
        raise SystemExit("Cannot parse Google Drive file id from URL")
    sess = requests.Session()
    base = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    resp = sess.get(base, params=params, stream=True)
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if token:
        params["confirm"] = token
        resp = sess.get(base, params=params, stream=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(32768):
            if chunk:
                f.write(chunk)


def _extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    return out_dir


def _find_yolo_root(search_dir: Path) -> Path | None:
    candidates = []
    for root, dirs, files in os.walk(search_dir):
        p = Path(root)
        img_train = p / 'images' / 'train'
        lbl_train = p / 'labels' / 'train'
        if img_train.exists() and lbl_train.exists():
            candidates.append(p)
    return min(candidates, key=lambda x: len(str(x))) if candidates else None


def _write_yolo_yaml(root: Path, out_yaml: Path, names: List[str]):
    data = {
        'path': root.as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': names,
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def train_yolo_sanity(model_cfg: str, data_yaml: Path, imgsz: int, epochs: int, device: str, project: Path, name: str):
    from ultralytics import YOLO
    m = YOLO(model_cfg)
    m.train(
        data=str(data_yaml),
        imgsz=int(imgsz),
        epochs=int(epochs),
        batch='auto',
        rect=True,
        multi_scale=True,
        amp=True,
        device=device,
        project=str(project),
        name=name,
        pretrained=True,
        seed=0,
        deterministic=True,
        verbose=True,
        resume=False,
        cache=False,
        plots=True,
        save_period=0,
    )
    return project / name / 'weights' / 'best.pt'


def get_rfdetr(size: str):
    from rfdetr import RFDETRBase
    if size.lower() == 'nano':
        try:
            from rfdetr import RFDETRNano
            return RFDETRNano()
        except Exception:
            return RFDETRBase()
    if size.lower() == 'small':
        try:
            from rfdetr import RFDETRSmall
            return RFDETRSmall()
        except Exception:
            return RFDETRBase()
    if size.lower() == 'medium':
        try:
            from rfdetr import RFDETRMedium
            return RFDETRMedium()
        except Exception:
            return RFDETRBase()
    return RFDETRBase()


def train_rfdetr_sanity(coco_dir: Path, out_dir: Path, size: str, epochs: int, batch_size: int,
                        grad_accum_steps: int, resolution: int, device: str | None):
    model = get_rfdetr(size)
    kw = {
        "dataset_dir": str(coco_dir),
        "output_dir": str(out_dir),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "resolution": int(resolution),
    }
    if device:
        kw["device"] = device
    model.train(**kw)


def main():
    ap = argparse.ArgumentParser(description="Unified sanity trainer for YOLOv11 and RF-DETR")
    ap.add_argument('--data', default='fetus-yolo/data_yolo/fetal_fp23.yaml')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--do', choices=['both', 'yolo', 'rfdetr'], default='both')
    ap.add_argument('--mode', choices=['sanity', 'real'], default='sanity', help='Preset config: sanity (quick) or real (full)')
    ap.add_argument('--fetch-url', default=None, help='Optional dataset .zip URL (Google Drive supported) to auto-download & prepare')
    ap.add_argument('--interactive', action='store_true', help='Prompt for choices (menu)')
    # YOLO
    ap.add_argument('--yolo-model', default=None)
    ap.add_argument('--yolo-img', type=int, default=None)
    ap.add_argument('--yolo-epochs', type=int, default=None)
    # RF-DETR
    ap.add_argument('--rfdetr-size', choices=['nano', 'small', 'medium', 'base'], default=None)
    ap.add_argument('--rfdetr-epochs', type=int, default=None)
    ap.add_argument('--rfdetr-batch', type=int, default=None)
    ap.add_argument('--rfdetr-accum', type=int, default=None)
    ap.add_argument('--rfdetr-res', type=int, default=None)
    # IO
    ap.add_argument('--runs-yolo', default='runs/detect')
    ap.add_argument('--runs-rfdetr', default='runs/rfdetr')
    ap.add_argument('--coco-dir', default='datasets/coco_rfdetr')
    args = ap.parse_args()

    # Interactive menu if requested or no extra args provided
    if args.interactive or len(sys.argv) == 1:
        print("Select training target: \n  1) YOLOv11\n  2) RF-DETR\n  3) Both")
        sel = (input("Choice [3]: ").strip() or '3')
        if sel == '1':
            args.do = 'yolo'
        elif sel == '2':
            args.do = 'rfdetr'
        else:
            args.do = 'both'

        md = input("Mode (sanity/real) [sanity]: ").strip().lower()
        args.mode = md if md in ('sanity', 'real') else 'sanity'

        dv = input("Device (auto/0) [auto]: ").strip()
        args.device = dv if dv else 'auto'

        if args.do in ('yolo', 'both'):
            ym_default = 'yolo11n-p2.yaml' if args.mode == 'sanity' else 'yolo11s-p2.yaml'
            yi_default = 640 if args.mode == 'sanity' else 1024
            ye_default = 10 if args.mode == 'sanity' else 200
            ym = input(f"YOLO model (yolo11n-p2.yaml / yolo11s-p2.yaml) [{ym_default}]: ").strip()
            args.yolo_model = ym or ym_default
            yi = input(f"YOLO image size [{yi_default}]: ").strip()
            args.yolo_img = int(yi) if yi else yi_default
            ye = input(f"YOLO epochs [{ye_default}]: ").strip()
            args.yolo_epochs = int(ye) if ye else ye_default

        if args.do in ('rfdetr', 'both'):
            rs_default = 'nano' if args.mode == 'sanity' else 'small'
            rr_default = 560
            re_default = 10 if args.mode == 'sanity' else 100
            rb_default = 4
            ra_default = 4
            rs = input(f"RF-DETR size (nano/small/medium/base) [{rs_default}]: ").strip().lower()
            args.rfdetr_size = rs or rs_default
            rr = input(f"RF-DETR resolution (divisible by 56) [{rr_default}]: ").strip()
            args.rfdetr_res = int(rr) if rr else rr_default
            re = input(f"RF-DETR epochs [{re_default}]: ").strip()
            args.rfdetr_epochs = int(re) if re else re_default
            rb = input(f"RF-DETR batch per GPU [{rb_default}]: ").strip()
            args.rfdetr_batch = int(rb) if rb else rb_default
            ra = input(f"RF-DETR grad accumulation [{ra_default}]: ").strip()
            args.rfdetr_accum = int(ra) if ra else ra_default

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = (REPO_ROOT / data_yaml).resolve()

    # Auto-fetch dataset if requested or if YAML missing and URL provided
    if args.fetch_url or not data_yaml.exists():
        url = args.fetch_url
        if not url and (args.interactive or len(sys.argv) == 1):
            url = input("Dataset .zip URL (Google Drive) [leave blank to skip]: ").strip() or None
        if url:
            zip_path = (REPO_ROOT / 'datasets' / 'auto' / 'dataset.zip').resolve()
            _gdrive_download(url, zip_path)
            extracted = _extract_zip(zip_path, zip_path.parent / 'extracted')
            yolo_root = _find_yolo_root(extracted)
            if not yolo_root:
                raise SystemExit("YOLO dataset structure not found after extraction (expected images/ and labels/).")
            auto_yaml = (REPO_ROOT / 'datasets' / 'auto' / 'auto.yaml').resolve()
            _write_yolo_yaml(yolo_root, auto_yaml, names=["abdomen", "head", "arm", "legs"])
            data_yaml = auto_yaml

    try:
        info, reports = ensure_dataset_ready(data_yaml, REPO_ROOT, run_integrity=True)
        if reports:
            from fetus_yolo.data.dataset_utils import format_reports
            print(format_reports(reports))
    except DatasetValidationError as exc:
        raise SystemExit(str(exc))

    # Resolve per-mode defaults if not provided
    yolo_model = args.yolo_model or ('yolo11n-p2.yaml' if args.mode == 'sanity' else 'yolo11s-p2.yaml')
    yolo_img = args.yolo_img if args.yolo_img is not None else (640 if args.mode == 'sanity' else 1024)
    yolo_epochs = args.yolo_epochs if args.yolo_epochs is not None else (10 if args.mode == 'sanity' else 200)

    rfdetr_size = args.rfdetr_size or ('nano' if args.mode == 'sanity' else 'small')
    rfdetr_res = args.rfdetr_res if args.rfdetr_res is not None else 560
    rfdetr_epochs = args.rfdetr_epochs if args.rfdetr_epochs is not None else (10 if args.mode == 'sanity' else 100)
    rfdetr_batch = args.rfdetr_batch if args.rfdetr_batch is not None else 4
    rfdetr_accum = args.rfdetr_accum if args.rfdetr_accum is not None else 4

    if args.do in ('rfdetr', 'both'):
        try:
            import rfdetr  # noqa: F401
        except Exception as e:
            raise SystemExit(f"RF-DETR not installed. Install with: pip install rfdetr\nDetails: {e}")
        coco_dir = convert_yolo_to_coco(info, Path(args.coco_dir).resolve())
        out_dir = Path(args.runs_rfdetr).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        train_rfdetr_sanity(
            coco_dir=coco_dir,
            out_dir=out_dir,
            size=rfdetr_size,
            epochs=rfdetr_epochs,
            batch_size=rfdetr_batch,
            grad_accum_steps=rfdetr_accum,
            resolution=rfdetr_res,
            device=None if args.device == 'auto' else args.device,
        )
        print(f"RF-DETR {args.mode} training done. Outputs: {out_dir}")

    if args.do in ('yolo', 'both'):
        try:
            from ultralytics import YOLO  # noqa: F401
        except Exception as e:
            raise SystemExit(f"Ultralytics not installed. Install with: pip install ultralytics\nDetails: {e}")
        yolo_runs = Path(args.runs_yolo).resolve()
        yolo_runs.mkdir(parents=True, exist_ok=True)
        best = train_yolo_sanity(
            model_cfg=yolo_model,
            data_yaml=data_yaml,
            imgsz=yolo_img,
            epochs=yolo_epochs,
            device=args.device,
            project=yolo_runs,
            name=f'{args.mode}_yolo11',
        )
        print(f"YOLOv11 {args.mode} training done. Best: {best}")

    return 0


if __name__ == '__main__':
    import sys as _sys
    _sys.exit(main())
