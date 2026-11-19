"""XD-Violence filename labels and official test interval annotations.

Raw XD-Violence media is not distributed with this repository. This module
expects the official videos and annotation text file to be placed locally.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

XD_LABEL_NAMES = {
    "A": "Normal",
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car accident",
    "G": "Explosion",
}
TARGET_VIOLENCE_CODES = {"B1", "B2", "B4", "B5"}
EXCLUDED_HAZARD_CODES = {"B6", "G"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_xd_filename(filename: str) -> tuple[int | None, list[str], str]:
    """Return (binary label, category names, disposition) for an XD filename.

    Violent categories take precedence in multi-label names. Normal videos are
    label 0. Car-accident and explosion-only videos are excluded from this
    human-violence task rather than mislabeled as normal.
    """
    stem = Path(filename).stem
    marker = "_label_"
    if marker not in stem:
        raise ValueError(f"XD filename has no '_label_' tag: {filename}")

    raw_tags = stem.rsplit(marker, 1)[1]
    tags = {
        part for part in raw_tags.replace("_", "-").split("-") if part and part != "0"
    }
    unknown = tags - XD_LABEL_NAMES.keys()
    if unknown:
        raise ValueError(f"Unknown XD label code(s) {sorted(unknown)} in {filename}")
    if not tags:
        raise ValueError(f"No XD label code found in {filename}")

    categories = sorted(XD_LABEL_NAMES[tag] for tag in tags)
    if tags & TARGET_VIOLENCE_CODES:
        return 1, categories, "include_violence"
    if tags == {"A"}:
        return 0, ["Normal"], "include_normal"
    if tags <= EXCLUDED_HAZARD_CODES:
        return None, categories, "exclude_non_human_hazard"
    raise ValueError(f"Unsupported XD label combination {sorted(tags)} in {filename}")


def scan_video_directory(
    video_dir: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Scan XD videos, read metadata, and emit portable project-relative paths."""
    video_dir = video_dir.resolve()
    project_root = project_root.resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    paths = sorted(
        p
        for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported video files found under {video_dir}")

    for path in paths:
        label, categories, disposition = parse_xd_filename(path.name)
        counts[disposition] += 1
        if label is None:
            continue

        try:
            relative_path = path.resolve().relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"XD videos must be inside the project folder for portable manifests. "
                f"Move {path} under {project_root / 'data' / 'raw'} or choose another project root."
            ) from exc

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"OpenCV could not open XD video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0 or total_frames <= 0:
            raise ValueError(f"Invalid FPS or frame count for XD video: {path}")

        records.append(
            {
                "source_video_id": path.resolve()
                .relative_to(project_root)
                .with_suffix("")
                .as_posix(),
                "video_path": relative_path,
                "label": int(label),
                "category": "+".join(categories),
                "source_categories": categories,
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": round(total_frames / fps, 4),
            }
        )
    return records, counts


def stratified_video_split(
    records: list[dict[str, Any]],
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split training videos into train/validation while preserving labels."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for record in records:
        by_label[int(record["label"])].append(record)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for label, group in by_label.items():
        if len(group) < 2:
            raise ValueError(
                f"Need at least two training videos for each class; label {label} has {len(group)}."
            )
        rng.shuffle(group)
        n_validation = max(1, round(len(group) * validation_ratio))
        if n_validation >= len(group):
            n_validation = len(group) - 1
        validation.extend(group[:n_validation])
        train.extend(group[n_validation:])

    rng.shuffle(train)
    rng.shuffle(validation)
    _assert_disjoint(train, validation)
    return train, validation


def parse_xd_test_annotations(annotation_path: Path) -> dict[str, list[list[int]]]:
    """Parse the official XD annotation format into inclusive frame intervals.

    Each row starts with ``v=<video filename>`` and is followed by pairs of
    start/end frame indices, for example ``v=clip_label_B1-0-0 523 610``.
    """
    annotations: dict[str, list[list[int]]] = {}
    if not annotation_path.is_file():
        raise FileNotFoundError(f"XD test annotations not found: {annotation_path}")

    for line_number, raw_line in enumerate(
        annotation_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = raw_line.split()
        if not fields:
            continue
        key = fields[0].removeprefix("v=")
        values: list[int] = []
        try:
            values = [int(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(
                f"Invalid frame index on annotation line {line_number}"
            ) from exc
        if len(values) % 2:
            raise ValueError(f"Unpaired frame index on annotation line {line_number}")
        intervals = []
        for start, end in zip(values[::2], values[1::2]):
            if start < 0 or end < start:
                raise ValueError(
                    f"Invalid interval ({start}, {end}) on line {line_number}"
                )
            intervals.append([start, end])
        annotations[Path(key).name] = intervals
    return annotations


def attach_test_events(
    test_records: list[dict[str, Any]],
    annotation_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    annotations = parse_xd_test_annotations(annotation_path)
    lookup: dict[str, dict[str, Any]] = {}
    for record in test_records:
        filename = Path(record["video_path"]).name
        keys = {filename.lower(), Path(filename).stem.lower()}
        for key in tuple(keys):
            without_video_prefix = key.removeprefix("v=")
            keys.add(without_video_prefix)
            keys.add(f"v={without_video_prefix}")
        for key in keys:
            lookup[key] = record

    unmatched: list[str] = []
    attached: set[str] = set()
    for filename, intervals in annotations.items():
        record = lookup.get(filename.lower()) or lookup.get(Path(filename).stem.lower())
        if record is None:
            unmatched.append(filename)
            continue
        if record["label"] == 0:
            raise ValueError(
                f"XD event intervals were provided for a normal video: {filename}"
            )
        record["violent_intervals_frames"] = intervals
        attached.add(record["source_video_id"])

    missing_positive = [
        record["video_path"]
        for record in test_records
        if record["label"] == 1 and record["source_video_id"] not in attached
    ]
    if missing_positive:
        raise ValueError(
            "Some positive test videos have no matched official intervals. "
            "Check filename alignment; examples: " + ", ".join(missing_positive[:5])
        )
    return test_records, unmatched


def _assert_disjoint(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    left_ids = {record["source_video_id"] for record in left}
    right_ids = {record["source_video_id"] for record in right}
    overlap = left_ids & right_ids
    if overlap:
        raise ValueError(f"Source video leakage detected: {sorted(overlap)[:5]}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
