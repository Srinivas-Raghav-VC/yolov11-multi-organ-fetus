from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ultralytics.models.yolo.detect import DetectionTrainer


def compute_image_weights(labels: List[np.ndarray], class_weights: Dict[int, float], nc: int) -> np.ndarray:
    w = np.zeros(len(labels), dtype=np.float32)
    default = 0.0
    for i, lab in enumerate(labels):
        if lab is None or len(lab) == 0:
            w[i] = default
            continue
        clses = lab[:, 0].astype(int)
        w[i] = float(sum(class_weights.get(int(c), 0.0) for c in np.unique(clses)))
        if w[i] == 0.0:
            w[i] = default
    if (w > 0).any():
        w = w / (w.max() + 1e-9)
        w = 0.1 + 0.9 * w
    else:
        w[:] = 1.0
    return w


class BalancedDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, class_weights: Dict[int, float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights or {}

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode=mode, batch_size=batch_size)
        if mode != "train":
            return DataLoader(
                dataset,
                batch_size=min(batch_size, 32),
                shuffle=False,
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=getattr(self, 'collate_fn', None) or getattr(dataset, 'collate_fn', None),
            )
        labels = getattr(dataset, 'labels', None)
        if labels is None:
            sampler = None
            shuffle = True
        else:
            nc = getattr(self.data, 'nc', None) or len(self.data.get('names', [])) or 4
            img_weights = compute_image_weights(labels, self._class_weights, nc)
            sampler = WeightedRandomSampler(weights=torch.from_numpy(img_weights), num_samples=len(img_weights), replacement=True)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=getattr(self, 'collate_fn', None) or getattr(dataset, 'collate_fn', None),
        )
