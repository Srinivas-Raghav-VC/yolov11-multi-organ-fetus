"""
Prepare a YOLO dataset YAML by auto-detecting common folder layouts.

Detects both patterns:
 A) <root>/images/{train,val|valid,test} and <root>/labels/{train,val|valid,test}
 B) <root>/{train,val|valid,test}/images and <root>/{train,val|valid,test}/labels

Usage:
  python scripts/prepare_yolo_yaml.py --root /path/to/extracted --out datasets/auto/auto.yaml \
    --names abdomen head arm legs
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

try:
    import yaml
except Exception as e:
    raise SystemExit("PyYAML not installed. Please install with: pip install PyYAML")


def find_yolo_layout(base: Path):
    """Return tuple (layout_type, root_path, val_key, has_test) or None.
    layout_type: 'A' or 'B' as described in module docstring.
    val_key: 'val' or 'valid'
    has_test: bool
    """
    candidates = []
    for dp, _, _ in os.walk(base):
        p = Path(dp)
        # Pattern A
        if (p / 'images').is_dir() and (p / 'labels').is_dir():
            if (p / 'images' / 'train').exists() and (p / 'labels' / 'train').exists():
                val_key = None
                if (p / 'images' / 'valid').exists() and (p / 'labels' / 'valid').exists():
                    val_key = 'valid'
                elif (p / 'images' / 'val').exists() and (p / 'labels' / 'val').exists():
                    val_key = 'val'
                if val_key:
                    has_test = (p / 'images' / 'test').exists() and (p / 'labels' / 'test').exists()
                    candidates.append(('A', p, val_key, has_test))
        # Pattern B
        if (p / 'train' / 'images').exists() and (p / 'train' / 'labels').exists():
            val_key = None
            if (p / 'valid' / 'images').exists() and (p / 'valid' / 'labels').exists():
                val_key = 'valid'
            elif (p / 'val' / 'images').exists() and (p / 'val' / 'labels').exists():
                val_key = 'val'
            if val_key:
                has_test = (p / 'test' / 'images').exists() and (p / 'test' / 'labels').exists()
                candidates.append(('B', p, val_key, has_test))

    if not candidates:
        return None
    # Choose shortest path (most top-level)
    return sorted(candidates, key=lambda x: len(str(x[1])))[0]


def write_yaml(layout, out_path: Path, class_names: list[str]):
    layout_type, root, val_key, has_test = layout
    data = {
        'path': root.as_posix(),
        'names': class_names,
    }
    if layout_type == 'A':
        data['train'] = 'images/train'
        data['val'] = f'images/{val_key}'
        if has_test:
            data['test'] = 'images/test'
    else:  # 'B'
        data['train'] = 'train/images'
        data['val'] = f'{val_key}/images'
        if has_test:
            data['test'] = 'test/images'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Prepare YOLO YAML by auto-detecting dataset layout')
    ap.add_argument('--root', required=True, help='Extracted dataset root to scan')
    ap.add_argument('--out', required=True, help='Path to write YAML (e.g., datasets/auto/auto.yaml)')
    ap.add_argument('--names', nargs='+', required=True, help='Class names in order')
    args = ap.parse_args()

    base = Path(args.root).resolve()
    if not base.exists():
        raise SystemExit(f'Root does not exist: {base}')

    layout = find_yolo_layout(base)
    if not layout:
        print('No YOLO layout found under:', base)
        print('Expected either:')
        print(' A) <root>/images/{train,val|valid,test} and <root>/labels/{train,val|valid,test}')
        print(' B) <root>/{train,val|valid,test}/images and <root>/{train,val|valid,test}/labels')
        return 1

    out = write_yaml(layout, Path(args.out).resolve(), args.names)
    print('Wrote YAML:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
