from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


EXPECTED_CLASS_NAMES: Sequence[str] = ("abdomen", "head", "arm", "legs")


class DatasetValidationError(RuntimeError):
    """Raised when the dataset definition does not match FPUS23 expectations."""


@dataclass(slots=True)
class SplitReport:
    split: str
    image_count: int
    labels_present: int
    missing_labels: List[Path]
    empty_labels: int
    problem_files: List[Tuple[Path, List[str]]]

    @property
    def has_failures(self) -> bool:
        return bool(self.missing_labels or self.problem_files)


@dataclass(slots=True)
class DatasetInfo:
    data_yaml: Path
    dataset_root: Path
    names: Dict[int, str]
    splits: Dict[str, Tuple[Path, Path]]  # split -> (images_dir, labels_dir)


def read_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_dataset_root(data_yaml: Path, cfg: Dict, repo_root: Path) -> Path:
    path_value = cfg.get("path")
    if path_value:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            local = (data_yaml.parent / candidate).resolve()
            repo = (repo_root / candidate).resolve()
            if local.exists():
                return local
            if repo.exists():
                return repo
        return candidate.resolve()
    return data_yaml.parent.resolve()


def _ensure_dir(path: Path, kind: str, context: str) -> None:
    if not path.exists():
        raise DatasetValidationError(f"{kind} directory missing for {context}: {path}")
    if not path.is_dir():
        raise DatasetValidationError(f"{kind} path is not a directory for {context}: {path}")


def _labels_dir_for_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1:])
    return images_dir.parent / "labels" / images_dir.name


def validate_dataset_yaml(data_yaml: Path, repo_root: Path, *, expected_names: Sequence[str] = EXPECTED_CLASS_NAMES) -> DatasetInfo:
    if not data_yaml.exists():
        raise DatasetValidationError(f"Dataset YAML not found: {data_yaml}")

    cfg = read_yaml(data_yaml)
    dataset_root = _resolve_dataset_root(data_yaml, cfg, repo_root)

    names_field = cfg.get("names")
    if isinstance(names_field, dict):
        names = {int(k): str(v) for k, v in names_field.items()}
    elif isinstance(names_field, list):
        names = {idx: str(val) for idx, val in enumerate(names_field)}
    else:
        raise DatasetValidationError("Dataset YAML must define class names as list or mapping")

    expected_set = set(expected_names)
    actual_set = set(names.values())
    if actual_set != expected_set:
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        raise DatasetValidationError(
            "Dataset class names do not match FPUS23 expected classes. "
            f"Missing={sorted(missing)} Extra={sorted(extra)}"
        )

    splits: Dict[str, Tuple[Path, Path]] = {}
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if not rel:
            continue
        img_path = Path(rel)
        images_dir = img_path if img_path.is_absolute() else (dataset_root / img_path)
        images_dir = images_dir.resolve()
        labels_dir = _labels_dir_for_images(images_dir)
        _ensure_dir(images_dir, "Image", split)
        _ensure_dir(labels_dir, "Label", split)
        splits[split] = (images_dir, labels_dir)

    if not splits:
        raise DatasetValidationError("Dataset YAML defines no splits")

    return DatasetInfo(data_yaml=data_yaml.resolve(), dataset_root=dataset_root, names=names, splits=splits)


def gather_files(img_dir: Path, lbl_dir: Path) -> Tuple[List[Path], List[Path]]:
    img_files: List[Path] = []
    for root, _, files in os.walk(img_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                img_files.append((Path(root) / fn).resolve())
    img_files.sort()

    lbl_files: List[Path] = []
    for root, _, files in os.walk(lbl_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                lbl_files.append((Path(root) / fn).resolve())
    lbl_files.sort()
    return img_files, lbl_files


def img_to_label_path(img_path: Path, img_root: Path, lbl_root: Path) -> Path:
    rel = img_path.relative_to(img_root)
    return (lbl_root / rel).with_suffix(".txt")


def check_label_file(path: Path, class_count: int) -> Tuple[List[str], List[Tuple[int, float, float, float, float]]]:
    problems: List[str] = []
    lines: List[Tuple[int, float, float, float, float]] = []
    try:
        txt = path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover - defensive
        problems.append(f"read_error:{exc}")
        return problems, lines

    if not txt:
        return problems, lines

    for idx, line in enumerate(txt.splitlines()):
        parts = line.strip().split()
        if len(parts) != 5:
            problems.append(f"L{idx + 1}:bad_cols:{len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception:
            problems.append(f"L{idx + 1}:parse_error")
            continue
        if not (0 <= cls < class_count):
            problems.append(f"L{idx + 1}:bad_class:{cls}")
        for name, value in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not (0.0 <= value <= 1.0):
                problems.append(f"L{idx + 1}:out_of_range:{name}={value}")
        if w <= 0 or h <= 0:
            problems.append(f"L{idx + 1}:nonpos_wh:w={w},h={h}")
        lines.append((cls, x, y, w, h))
    return problems, lines


def run_split_checks(split: str, img_dir: Path, lbl_dir: Path, class_count: int) -> SplitReport:
    imgs, lbls = gather_files(img_dir, lbl_dir)
    missing: List[Path] = []
    empties = 0
    present = 0
    problems: List[Tuple[Path, List[str]]] = []

    for img in imgs:
        lbl = img_to_label_path(img, img_dir, lbl_dir)
        if not lbl.exists():
            missing.append(lbl)
            continue
        present += 1
        errs, lines = check_label_file(lbl, class_count)
        if not lines:
            empties += 1
        if errs:
            problems.append((lbl, errs[:5]))

    return SplitReport(
        split=split,
        image_count=len(imgs),
        labels_present=present,
        missing_labels=missing,
        empty_labels=empties,
        problem_files=problems,
    )


def inspect_dataset(data_yaml: Path, repo_root: Path, *, run_integrity: bool = True) -> Tuple[DatasetInfo, List[SplitReport]]:
    info = validate_dataset_yaml(data_yaml, repo_root)
    reports: List[SplitReport] = []
    if run_integrity:
        class_count = len(info.names)
        for split, (img_dir, lbl_dir) in info.splits.items():
            reports.append(run_split_checks(split, img_dir, lbl_dir, class_count))
    return info, reports


def ensure_dataset_ready(
    data_yaml: Path,
    repo_root: Path,
    *,
    run_integrity: bool = True,
    expected_names: Sequence[str] = EXPECTED_CLASS_NAMES,
) -> Tuple[DatasetInfo, List[SplitReport]]:
    info, reports = inspect_dataset(data_yaml, repo_root, run_integrity=run_integrity)
    expected_set = set(expected_names)
    actual_set = set(info.names.values())
    if actual_set != expected_set:
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        raise DatasetValidationError(
            "Class names mismatch detected. "
            f"Missing={sorted(missing)} Extra={sorted(extra)}"
        )

    if run_integrity:
        failures = [r for r in reports if r.has_failures]
        if failures:
            summaries = ", ".join(
                f"{r.split}: missing={len(r.missing_labels)} issues={len(r.problem_files)}" for r in failures
            )
            raise DatasetValidationError(
                "Dataset integrity check failed. "
                f"Resolve missing or malformed labels before training. Details: {summaries}"
            )
    return info, reports


def format_reports(reports: Iterable[SplitReport]) -> str:
    lines = []
    for rep in reports:
        lines.append(
            f"[{rep.split}] images={rep.image_count} labels_present={rep.labels_present} "
            f"missing={len(rep.missing_labels)} empty={rep.empty_labels} problems={len(rep.problem_files)}"
        )
        if rep.missing_labels:
            lines.append(f"  example_missing: {rep.missing_labels[0]}")
        for path, issues in rep.problem_files[:3]:
            joined = ", ".join(issues)
            lines.append(f"  {path} -> {joined}")
    return "\n".join(lines)
