#!/usr/bin/env python3
"""
K-Fold Cross-Validation for robust evaluation.

This script performs stream-based k-fold cross-validation to:
1. Verify results are not due to lucky train/val split
2. Compute mean and standard deviation of metrics
3. Test statistical significance of improvements
4. Provide confidence intervals

Usage:
  python scripts/cross_validate.py \
    --model yolo11s-p2.yaml \
    --data-root data_yolo \
    --n-folds 5 \
    --epochs 200 \
    --img 1024 \
    --device 0 \
    --save-dir results/cv

Timeline: n_folds × epochs (e.g., 5 folds × 200 epochs = ~1 week on single GPU)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Statistical tests unavailable.")


def get_unique_streams(data_root: Path) -> List[str]:
    """
    Get unique stream names from dataset.
    
    Assumes directory structure: data_root/labels/train/stream_name/file.txt
    """
    labels_dir = data_root / 'labels' / 'train'
    
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    streams = set()
    
    # Look for subdirectories
    for item in labels_dir.iterdir():
        if item.is_dir():
            streams.add(item.name)
    
    # If no subdirectories, look for files directly
    if not streams:
        # Files are directly in train/, no stream structure
        # In this case, we'll split by file groups
        print("⚠️  No stream-based directory structure found")
        print("   Will use random file-based splits instead")
        return None
    
    return sorted(list(streams))


def create_fold_dataset(train_streams: List[str], val_streams: List[str],
                        data_root: Path, fold_idx: int,
                        original_yaml: Path) -> Path:
    """
    Create fold-specific dataset YAML.
    
    Returns:
        Path to created fold YAML
    """
    # Read original YAML for reference
    with open(original_yaml) as f:
        base_config = yaml.safe_load(f)
    
    # Create fold-specific config
    fold_config = {
        'path': str(data_root.absolute()),
        'names': base_config.get('names', {0: 'abdomen', 1: 'head', 2: 'arm', 3: 'legs'}),
    }
    
    # Set train and val paths
    # If streams are in subdirectories
    if train_streams:
        fold_config['train'] = [f'images/train/{s}' for s in train_streams]
        fold_config['val'] = [f'images/train/{s}' for s in val_streams]
    else:
        # Fall back to original structure
        fold_config['train'] = 'images/train'
        fold_config['val'] = 'images/val'
    
    # Save fold YAML
    fold_yaml = data_root / f'fold{fold_idx}_data.yaml'
    with open(fold_yaml, 'w') as f:
        yaml.dump(fold_config, f, default_flow_style=False)
    
    return fold_yaml


def train_fold(model_cfg: str, data_yaml: Path, fold_idx: int,
               epochs: int, imgsz: int, device: str,
               hyp: str = None, seed: int = 42,
               project: str = 'runs/detect') -> Dict:
    """
    Train a single fold.
    
    Returns:
        Dictionary with fold results
    """
    fold_name = f'fold{fold_idx}'
    
    print("\n" + "="*80)
    print(f"TRAINING FOLD {fold_idx + 1}")
    print("="*80)
    
    start_time = datetime.now()
    
    try:
        model = YOLO(model_cfg)
        
        # Train
        model.train(
            data=str(data_yaml),
            imgsz=imgsz,
            epochs=epochs,
            batch='auto',
            rect=True,
            multi_scale=True,
            amp=True,
            device=device,
            project=project,
            name=fold_name,
            pretrained=True,
            hyp=hyp,
            seed=seed + fold_idx,  # Different seed per fold
            deterministic=True,
            verbose=True,
            resume=False,
            cache=False,
            plots=True,
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        print(f"\nEvaluating fold {fold_idx}...")
        val_results = model.val(data=str(data_yaml), imgsz=imgsz, conf=0.25)
        
        # Extract metrics (prefer metrics.box.maps)
        box = getattr(val_results, 'box', None)
        
        results = {
            'fold': fold_idx,
            'status': 'success',
            'training_time_seconds': duration,
            'training_time_hours': duration / 3600,
        }
        
        if box:
            results['mAP50'] = float(box.map50) if hasattr(box, 'map50') else None
            results['mAP50-95'] = float(box.map) if hasattr(box, 'map') else None
            results['precision'] = float(box.mp) if hasattr(box, 'mp') else None
            results['recall'] = float(box.mr) if hasattr(box, 'mr') else None
            
            # Per-class AP
            maps = getattr(box, 'maps', None)
            if isinstance(maps, (list, tuple, np.ndarray)):
                arr = np.array(maps).flatten()
                results['per_class_ap'] = [float(x) for x in arr]
        
        print(f"✅ Fold {fold_idx} completed: mAP50 = {results.get('mAP50', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Fold {fold_idx} FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'fold': fold_idx,
            'status': 'failed',
            'error': str(e),
        }


def compute_statistics(fold_results: List[Dict], metric_name: str) -> Dict:
    """
    Compute statistics for a metric across folds.
    
    Returns:
        Dictionary with mean, std, min, max, median, ci_lower, ci_upper
    """
    values = [r[metric_name] for r in fold_results 
              if r.get('status') == 'success' and metric_name in r and r[metric_name] is not None]
    
    if not values:
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None,
            'n': 0,
        }
    
    values = np.array(values)
    
    stats_dict = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)),  # Sample std
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'n': len(values),
    }
    
    # 95% confidence interval
    if len(values) > 1:
        if HAS_SCIPY:
            ci = stats.t.interval(
                0.95,
                len(values) - 1,
                loc=stats_dict['mean'],
                scale=stats.sem(values)
            )
            stats_dict['ci_lower'] = float(ci[0])
            stats_dict['ci_upper'] = float(ci[1])
        else:
            # Approximate CI without scipy
            sem = stats_dict['std'] / np.sqrt(len(values))
            margin = 1.96 * sem  # Approximate for large n
            stats_dict['ci_lower'] = stats_dict['mean'] - margin
            stats_dict['ci_upper'] = stats_dict['mean'] + margin
    
    return stats_dict


def compare_to_baseline(your_results: List[Dict], baseline_results: List[Dict],
                       metric_name: str = 'mAP50') -> Dict:
    """
    Statistical comparison to baseline using paired t-test.
    
    Returns:
        Dictionary with test results
    """
    your_values = [r[metric_name] for r in your_results 
                   if r.get('status') == 'success' and metric_name in r]
    baseline_values = [r[metric_name] for r in baseline_results 
                      if r.get('status') == 'success' and metric_name in r]
    
    if not your_values or not baseline_values:
        return {'error': 'Insufficient data for comparison'}
    
    if len(your_values) != len(baseline_values):
        return {'error': 'Mismatched fold counts'}
    
    your_values = np.array(your_values)
    baseline_values = np.array(baseline_values)
    
    # Compute differences
    differences = your_values - baseline_values
    mean_diff = np.mean(differences)
    
    comparison = {
        'your_mean': float(np.mean(your_values)),
        'baseline_mean': float(np.mean(baseline_values)),
        'mean_difference': float(mean_diff),
        'relative_improvement_%': float((mean_diff / np.mean(baseline_values)) * 100),
    }
    
    # Paired t-test
    if HAS_SCIPY and len(differences) > 1:
        t_stat, p_value = stats.ttest_rel(your_values, baseline_values)
        
        comparison['t_statistic'] = float(t_stat)
        comparison['p_value'] = float(p_value)
        comparison['significant_at_0.05'] = bool(p_value < 0.05)
        comparison['significant_at_0.01'] = bool(p_value < 0.01)
        
        # Effect size (Cohen's d for paired samples)
        d = mean_diff / np.std(differences, ddof=1)
        comparison['cohens_d'] = float(d)
        
        # Interpretation
        if abs(d) < 0.2:
            effect_size_interpretation = 'negligible'
        elif abs(d) < 0.5:
            effect_size_interpretation = 'small'
        elif abs(d) < 0.8:
            effect_size_interpretation = 'medium'
        else:
            effect_size_interpretation = 'large'
        
        comparison['effect_size_interpretation'] = effect_size_interpretation
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='K-fold cross-validation for robust evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', default='yolo11s-p2.yaml', help='Model configuration')
    parser.add_argument('--data-root', default='data_yolo', help='Dataset root')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--img', type=int, default=1024, help='Image size')
    parser.add_argument('--device', default='0', help='CUDA device')
    parser.add_argument('--hyp', default=None, help='Custom hyperparameters YAML')
    parser.add_argument('--save-dir', default='results/cv', help='Save directory')
    parser.add_argument('--project', default='runs/detect', help='Ultralytics project dir')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--quick-test', action='store_true', help='Quick test (3 folds, 10 epochs)')
    parser.add_argument('--yes', action='store_true', help='Auto-confirm prompts')
    args = parser.parse_args()
    
    if args.quick_test:
        print("\n⚠️  QUICK TEST MODE")
        args.n_folds = 3
        args.epochs = 10
    
    # Setup
    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.hyp and not Path(args.hyp).exists():
        # Try repo root
        repo_root = Path(__file__).parent.parent
        hyp_path = repo_root / 'configs' / args.hyp
        if hyp_path.exists():
            args.hyp = str(hyp_path)
    
    print("="*80)
    print("CROSS-VALIDATION SETUP")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Data root: {data_root}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Image size: {args.img}")
    print(f"Device: {args.device}")
    print(f"Hyperparameters: {args.hyp if args.hyp else 'default'}")
    print(f"Save directory: {save_dir}")
    print("="*80)
    
    # Get streams for splitting
    print("\nDiscovering dataset structure...")
    streams = get_unique_streams(data_root)
    
    if streams is None:
        print("❌ Cannot perform stream-based cross-validation")
        print("   Dataset does not have stream-based directory structure")
        return 1
    
    print(f"Found {len(streams)} streams:")
    for i, stream in enumerate(streams[:10]):
        print(f"  {i+1}. {stream}")
    if len(streams) > 10:
        print(f"  ... and {len(streams) - 10} more")
    
    if len(streams) < args.n_folds:
        print(f"\n⚠️  WARNING: Only {len(streams)} streams for {args.n_folds} folds")
        print(f"   Reducing to {len(streams)} folds")
        args.n_folds = len(streams)
    
    # Prepare k-fold split
    streams = np.array(streams)
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Find original data YAML
    original_yaml = data_root / 'fetal_fp23.yaml'
    if not original_yaml.exists():
        original_yaml = list(data_root.glob('*.yaml'))[0] if list(data_root.glob('*.yaml')) else None
    
    print(f"\nEstimated time: ~{args.n_folds * 8} hours (assuming 8h per fold)")
    
    if not args.quick_test and not args.yes:
        response = input("\nProceed with cross-validation? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted")
            return 1
    
    # Run k-fold CV
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(streams)):
        train_streams = streams[train_idx].tolist()
        val_streams = streams[val_idx].tolist()
        
        print(f"\nFold {fold_idx + 1}/{args.n_folds}")
        print(f"  Train streams: {len(train_streams)}")
        print(f"  Val streams: {len(val_streams)}")
        
        # Create fold-specific dataset
        fold_yaml = create_fold_dataset(
            train_streams, val_streams,
            data_root, fold_idx, original_yaml
        )
        
        # Train fold
        result = train_fold(
            model_cfg=args.model,
            data_yaml=fold_yaml,
            fold_idx=fold_idx,
            epochs=args.epochs,
            imgsz=args.img,
            device=args.device,
            hyp=args.hyp,
            seed=args.seed,
            project=args.project,
        )
        
        fold_results.append(result)
        
        # Save intermediate results
        intermediate_file = save_dir / 'fold_results_intermediate.json'
        with open(intermediate_file, 'w') as f:
            json.dump(fold_results, f, indent=2)
    
    # Compute aggregate statistics
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    
    successful_folds = [r for r in fold_results if r.get('status') == 'success']
    print(f"\nSuccessful folds: {len(successful_folds)}/{len(fold_results)}")
    
    if len(successful_folds) == 0:
        print("❌ No successful folds")
        return 1
    
    # Aggregate metrics
    aggregate_stats = {}
    
    for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        metric_stats = compute_statistics(fold_results, metric)
        aggregate_stats[metric] = metric_stats
        
        if metric_stats['mean'] is not None:
            print(f"\n{metric}:")
            print(f"  Mean: {metric_stats['mean']:.4f}")
            print(f"  Std:  {metric_stats['std']:.4f}")
            print(f"  Min:  {metric_stats['min']:.4f}")
            print(f"  Max:  {metric_stats['max']:.4f}")
            if 'ci_lower' in metric_stats:
                print(f"  95% CI: [{metric_stats['ci_lower']:.4f}, {metric_stats['ci_upper']:.4f}]")
    
    # Per-class analysis
    print("\n" + "="*80)
    print("PER-CLASS CROSS-VALIDATION")
    print("="*80)
    
    class_names = ['abdomen', 'head', 'arm', 'legs']
    
    # Extract per-class APs from all folds
    per_class_by_fold = []
    for result in successful_folds:
        if 'per_class_ap' in result:
            per_class_by_fold.append(result['per_class_ap'])
    
    if per_class_by_fold:
        per_class_by_fold = np.array(per_class_by_fold)  # Shape: (n_folds, n_classes)
        
        for class_idx, class_name in enumerate(class_names):
            if class_idx < per_class_by_fold.shape[1]:
                class_values = per_class_by_fold[:, class_idx]
                
                print(f"\n{class_name}:")
                print(f"  Mean AP50: {np.mean(class_values):.4f} ± {np.std(class_values, ddof=1):.4f}")
                print(f"  Min:  {np.min(class_values):.4f}")
                print(f"  Max:  {np.max(class_values):.4f}")
    
    # Save results
    results_file = save_dir / 'cv_results.json'
    output_data = {
        'config': {
            'model': args.model,
            'n_folds': args.n_folds,
            'epochs': args.epochs,
            'imgsz': args.img,
            'seed': args.seed,
        },
        'fold_results': fold_results,
        'aggregate_statistics': aggregate_stats,
        'successful_folds': len(successful_folds),
        'total_folds': len(fold_results),
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # CSV summary
    csv_file = save_dir / 'cv_results.csv'
    fold_df = pd.DataFrame([r for r in fold_results if r.get('status') == 'success'])
    fold_df.to_csv(csv_file, index=False)
    print(f"✅ CSV saved to: {csv_file}")
    
    # Generate report
    report_file = save_dir / 'cv_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-VALIDATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Model: {args.model}\n")
        f.write(f"Folds: {args.n_folds}\n")
        f.write(f"Epochs per fold: {args.epochs}\n")
        f.write(f"Image size: {args.img}\n")
        f.write(f"Successful folds: {len(successful_folds)}/{len(fold_results)}\n\n")
        
        f.write("AGGREGATE RESULTS\n")
        f.write("-"*80 + "\n")
        for metric, stats in aggregate_stats.items():
            if stats['mean'] is not None:
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                if 'ci_lower' in stats:
                    f.write(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")
    
    print(f"✅ Report saved to: {report_file}")
    
    print("\n" + "="*80)
    print("✅ CROSS-VALIDATION COMPLETE")
    print("="*80)
    
    # Final summary
    if aggregate_stats.get('mAP50', {}).get('mean'):
        mAP50_stats = aggregate_stats['mAP50']
        print(f"\nFinal Result: mAP50 = {mAP50_stats['mean']:.4f} ± {mAP50_stats['std']:.4f}")
        
        if 'ci_lower' in mAP50_stats:
            print(f"95% Confidence Interval: [{mAP50_stats['ci_lower']:.4f}, {mAP50_stats['ci_upper']:.4f}]")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

<COPY>
