"""
Unified training CLI for Fetus-YOLO.

Modes:
  - proposed: single best recipe (YOLO11s-P2, high-res)
  - baselines: train a suite of comparable configs
  - dual: train nano+small models
  - quick: fast sanity check (10 epochs)
Options:
  --balanced/--class-weights for class-balanced sampling
  --boost (legs/arms multipliers) to create oversampled train list (optional)
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fetus_yolo.data.dataset_utils import (
    DatasetValidationError,
    ensure_dataset_ready,
    format_reports,
)
from fetus_yolo.train import BalancedDetectionTrainer


def ensure_boosted_yaml(data_yaml: Path, legs_mult: int, arms_mult: int, repo_root: Path, ds_root: Path) -> Path:
    boosted_list = ds_root / 'train_boosted.txt'
    import subprocess

    subprocess.check_call([
        sys.executable, str(repo_root / 'scripts' / 'make_oversampled_train.py'),
        '--data-root', str(ds_root),
        '--legs-mult', str(legs_mult), '--arms-mult', str(arms_mult),
        '--out', str(boosted_list)
    ])
    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    path_field = cfg.get('path')
    boosted_yaml = data_yaml.parent / (data_yaml.stem + '_boosted.yaml')
    cfg_boost = dict(cfg)
    if path_field:
        cfg_boost['train'] = 'train_boosted.txt'
    else:
        cfg_boost['train'] = boosted_list.as_posix()
    boosted_yaml.write_text(yaml.safe_dump(cfg_boost, sort_keys=False), encoding='utf-8')
    return boosted_yaml


def train_one(model_cfg: str, name: str, data: str, hyp: str, img: int, epochs: int, device: str,
              project: str, save_period: int, balanced: bool, class_weights: list | None):
    if balanced:
        cw_map = None if class_weights is None else {i: float(v) for i, v in enumerate(class_weights)}
        trainer = BalancedDetectionTrainer(overrides={
            'model': model_cfg,
            'data': data,
            'imgsz': img,
            'epochs': epochs,
            'device': device,
            'project': project,
            'name': name,
            'rect': True,
            'multi_scale': True,
            'amp': True,
            'seed': 0,
            'deterministic': True,
            'verbose': True,
            'resume': False,
            'cache': False,
            'plots': True,
            'save_period': save_period,
            'hyp': hyp,
        }, class_weights=cw_map)
        trainer.train()
        # For consistency with YOLO object, fetch results via eval
        model = YOLO(model_cfg)
        val_results = model.val(data=data, imgsz=img, conf=0.25)
    else:
        model = YOLO(model_cfg)
        model.train(
            data=data,
            imgsz=img,
            epochs=epochs,
            batch='auto',
            rect=True,
            multi_scale=True,
            amp=True,
            device=device,
            project=project,
            name=name,
            pretrained=True,
            hyp=hyp,
            seed=0,
            deterministic=True,
            verbose=True,
            resume=False,
            cache=False,
            plots=True,
            save_period=save_period,
        )
        val_results = model.val(data=data, imgsz=img, conf=0.25)

    # Extract metrics
    box = getattr(val_results, 'box', None)
    row = {
        'name': name,
        'model_cfg': model_cfg,
        'imgsz': img,
    }
    if box:
        row.update({
            'mAP50': float(box.map50) if hasattr(box, 'map50') else None,
            'mAP50-95': float(box.map) if hasattr(box, 'map') else None,
            'precision': float(box.mp) if hasattr(box, 'mp') else None,
            'recall': float(box.mr) if hasattr(box, 'mr') else None,
        })
        maps = getattr(box, 'maps', None)
        if isinstance(maps, (list, tuple, np.ndarray)):
            arr = np.array(maps).flatten()
            row['per_class_ap'] = [float(x) for x in arr]
    return row


def main():
    ap = argparse.ArgumentParser(description='Unified training CLI')
    ap.add_argument('--mode', choices=['proposed', 'baselines', 'dual', 'quick'], default='proposed')
    ap.add_argument('--data', default='fetus-yolo/data_yolo/fetal_fp23.yaml')
    ap.add_argument('--img', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--boost', action='store_true')
    ap.add_argument('--legs-mult', type=int, default=4)
    ap.add_argument('--arms-mult', type=int, default=2)
    ap.add_argument('--balanced', action='store_true')
    ap.add_argument('--class-weights', nargs='+', type=float)
    ap.add_argument('--project', default='runs/detect')
    ap.add_argument('--save-dir', default='results/train')
    ap.add_argument('--yes', action='store_true')
    args = ap.parse_args()

    repo_root = REPO_ROOT
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data).resolve() if Path(args.data).exists() else (repo_root / args.data)
    try:
        info, reports = ensure_dataset_ready(data_yaml, repo_root, run_integrity=True)
    except DatasetValidationError as exc:
        raise SystemExit(str(exc))
    if reports:
        print(format_reports(reports))

    data_used = str(data_yaml)
    if args.boost:
        data_used = str(ensure_boosted_yaml(data_yaml, args.legs_mult, args.arms_mult, repo_root, info.dataset_root))

    hyp_custom = repo_root / 'configs' / 'hyp_ultrasound_smallobj.yaml'
    hyp_str = str(hyp_custom) if hyp_custom.exists() else None

    if args.mode == 'quick':
        args.epochs = 10

    if args.mode in ['baselines'] and not args.yes:
        resp = input('Proceed with baseline training? [y/N]: ')
        if resp.lower() != 'y':
            print('Aborted')
            return 1

    experiments = []
    if args.mode == 'proposed' or args.mode == 'quick':
        experiments.append(('yolo11s-p2.yaml', 'proposed_full', args.img))
    elif args.mode == 'dual':
        experiments.extend([
            ('yolo11n-p2.yaml', 'yolo11n_ultra_smallobj', args.img),
            ('yolo11s-p2.yaml', 'yolo11s_ultra_smallobj', args.img),
        ])
    elif args.mode == 'baselines':
        experiments.extend([
            ('yolo11s-p2.yaml', 'proposed_full', args.img),
            ('yolo11s.yaml', 'baseline_no_p2', args.img),
            ('yolo11s-p2.yaml', 'baseline_lowres', 640),
            ('yolo11s-p2.yaml', 'baseline_no_boost', args.img),
            ('yolov8s.yaml', 'baseline_yolov8', args.img),
        ])

    rows = []
    for model_cfg, name, img in experiments:
        data_for_run = data_used
        if name == 'baseline_no_boost' and args.boost:
            data_for_run = str(data_yaml)  # revert to original
        row = train_one(model_cfg, name, data_for_run, hyp_str, img, args.epochs, args.device, args.project,
                        save_period=0, balanced=args.balanced, class_weights=args.class_weights)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / f'train_results_{args.mode}.csv', index=False)
    (save_dir / f'train_results_{args.mode}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f"Saved training results to {save_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

<COPY>
