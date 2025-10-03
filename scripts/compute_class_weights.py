#!/usr/bin/env python3
"""
Compute class weights for handling class imbalance.
"""
import argparse
import json
import yaml
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np


def read_yolo_labels(label_file: Path) -> List[int]:
    class_ids = []
    try:
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    class_ids.append(int(float(parts[0])))
                except Exception:
                    continue
    except Exception:
        pass
    return class_ids


def count_class_instances(label_dir: Path, num_classes: int = 4) -> Dict[int, int]:
    class_counts = Counter()
    total_labels = 0
    empty_labels = 0
    for label_file in label_dir.rglob('*.txt'):
        class_ids = read_yolo_labels(label_file)
        if not class_ids:
            empty_labels += 1
        else:
            class_counts.update(class_ids)
            total_labels += 1
    print(f"\nDataset Statistics:")
    print(f"  Total label files: {total_labels + empty_labels}")
    print(f"  Non-empty labels: {total_labels}")
    print(f"  Empty labels: {empty_labels}")
    print(f"  Total instances: {sum(class_counts.values())}")
    return {i: class_counts.get(i, 0) for i in range(num_classes)}


def compute_inverse_frequency_weights(class_counts: Dict[int, int]) -> Dict[int, float]:
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = 0.0 if count == 0 else total / (num_classes * count)
    max_weight = max(weights.values()) if weights.values() else 1.0
    return {k: v / max_weight for k, v in weights.items()}


def compute_inverse_sqrt_weights(class_counts: Dict[int, int]) -> Dict[int, float]:
    total = sum(class_counts.values())
    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = 0.0 if count == 0 else float(np.sqrt(total / count))
    max_weight = max(weights.values()) if weights.values() else 1.0
    return {k: v / max_weight for k, v in weights.items()}


def compute_effective_samples_weights(class_counts: Dict[int, int], beta: float = 0.9999) -> Dict[int, float]:
    weights = {}
    for class_id, count in class_counts.items():
        if count == 0:
            weights[class_id] = 0.0
        else:
            effective_num = (1.0 - beta) / (1.0 - np.power(beta, count))
            weights[class_id] = 1.0 / effective_num
    max_weight = max(weights.values()) if weights.values() else 1.0
    return {k: v / max_weight for k, v in weights.items()}


def main():
    parser = argparse.ArgumentParser(description='Compute class weights for handling imbalance')
    parser.add_argument('--data-root', default='data_yolo', help='Dataset root')
    parser.add_argument('--split', default='train', help='Split to analyze')
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--class-names', nargs='+', default=['abdomen', 'head', 'arm', 'legs'])
    parser.add_argument('--strategy', choices=['inverse_frequency', 'inverse_sqrt', 'effective_samples', 'all'], default='all')
    parser.add_argument('--beta', type=float, default=0.9999)
    parser.add_argument('--output', default='class_weights.yaml')
    parser.add_argument('--format', choices=['yaml', 'json'], default='yaml')
    args = parser.parse_args()

    label_dir = Path(args.data_root) / 'labels' / args.split
    if not label_dir.exists():
        print(f"❌ Label directory not found: {label_dir}")
        return 1

    counts = count_class_instances(label_dir, args.num_classes)
    print("\nClass Counts:")
    for i in range(args.num_classes):
        print(f"  {args.class_names[i] if i < len(args.class_names) else i}: {counts[i]}")

    strategies = {}
    if args.strategy in ('all', 'inverse_frequency'):
        strategies['inverse_frequency'] = compute_inverse_frequency_weights(counts)
    if args.strategy in ('all', 'inverse_sqrt'):
        strategies['inverse_sqrt'] = compute_inverse_sqrt_weights(counts)
    if args.strategy in ('all', 'effective_samples'):
        strategies['effective_samples'] = compute_effective_samples_weights(counts, args.beta)

    output_data = {
        'class_counts': counts,
        'class_names': args.class_names,
        'strategies': {k: {int(kk): float(vv) for kk, vv in v.items()} for k, v in strategies.items()},
        'recommended': 'effective_samples' if max(counts.values()) / max(1, min([c for c in counts.values() if c > 0], default=1)) > 10 else 'inverse_frequency'
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == 'yaml':
        out.write_text(yaml.safe_dump(output_data, sort_keys=False), encoding='utf-8')
    else:
        out.write_text(json.dumps(output_data, indent=2), encoding='utf-8')
    print(f"✅ Saved: {out}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
