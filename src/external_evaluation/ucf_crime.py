"""UCF-Crime test-manifest parsing and event-level scoring helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable

import cv2
import numpy as np
import yaml

VIOLENCE_CATEGORIES = {"Abuse", "Assault", "Fighting", "Shooting"}
NORMAL_CATEGORIES = {"Normal", "Normal_Videos_event", "Testing_Normal_Videos_Anomaly"}
MANIFEST_FIELDS = (
    "video_id",
    "relative_path",
    "video_path",
    "category",
    "target_label",
    "split",
    "annotation_status",
    "file_exists",
    "fps",
    "duration_s",
    "ground_truth_intervals_json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_official_test_split(path: str | Path) -> list[str]:
    entries = []
    seen = set()
    for line_no, raw in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        value = value.split()[0].replace("\\", "/")
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"Unsafe path in UCF test split on line {line_no}: {value!r}"
            )
        normalized = relative.as_posix().removeprefix("./")
        if normalized in seen:
            raise ValueError(f"Duplicate video in UCF test split: {normalized}")
        seen.add(normalized)
        entries.append(normalized)
    if not entries:
        raise ValueError(f"No videos found in UCF test split: {path}")
    return entries


def read_temporal_annotations(path: str | Path) -> dict[str, dict]:
    """Read the official text rows: name, category, then inclusive frame pairs."""
    annotations = {}
    for line_no, raw in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 4 or (len(fields) - 2) % 2:
            raise ValueError(f"Malformed temporal annotation on line {line_no}")
        name, category = Path(fields[0]).name, fields[1]
        try:
            values = [int(value) for value in fields[2:]]
        except ValueError as exc:
            raise ValueError(f"Non-integer frame annotation on line {line_no}") from exc
        intervals = []
        for start_frame, end_frame in zip(values[::2], values[1::2]):
            if start_frame < 0 and end_frame < 0:
                continue
            if start_frame < 1 or end_frame < start_frame:
                raise ValueError(f"Invalid frame interval on annotation line {line_no}")
            intervals.append({"start_frame": start_frame, "end_frame": end_frame})
        key = name.casefold()
        if key in annotations:
            raise ValueError(f"Duplicate temporal annotation for {name}")
        annotations[key] = {"category": category, "intervals": intervals}
    return annotations


def _video_info(path: Path) -> tuple[float | None, float | None]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None, None
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            return None, None
        duration = frames / fps if math.isfinite(frames) and frames > 0 else None
        return fps, duration
    finally:
        capture.release()


def build_ucf_test_manifest(
    videos_root: str | Path,
    split_file: str | Path,
    temporal_annotations_file: str | Path,
    mapping_file: str | Path = "configs/behavior/dataset_mapping.yaml",
) -> tuple[list[dict], dict]:
    """Resolve only official test-split paths and map target violence labels."""
    root = Path(videos_root).expanduser().resolve()
    mapping = yaml.safe_load(Path(mapping_file).read_text(encoding="utf-8"))
    labels = mapping["ucf_crime"]["label_mapping"]
    annotations = read_temporal_annotations(temporal_annotations_file)
    entries = read_official_test_split(split_file)
    rows = []
    missing = []
    seen_video_ids = set()

    for relative in entries:
        relative_path = PurePosixPath(relative)
        video_path = (root / Path(*relative_path.parts)).resolve()
        try:
            video_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Video path escapes the dataset root: {relative}"
            ) from exc
        name = video_path.name
        annotation = annotations.get(name.casefold())
        path_category = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        if (
            annotation
            and path_category not in NORMAL_CATEGORIES
            and path_category.casefold() != annotation["category"].casefold()
        ):
            raise ValueError(
                f"Split category {path_category!r} disagrees with annotation "
                f"category {annotation['category']!r} for {name}"
            )
        if (
            annotation
            and name.casefold().startswith("normal")
            and annotation["category"].casefold() != "normal"
        ):
            raise ValueError(
                f"Normal-video annotation has an unexpected category for {name}"
            )
        if path_category in NORMAL_CATEGORIES or name.casefold().startswith("normal"):
            category = "Normal_Videos_event"
        else:
            category = annotation["category"] if annotation else path_category
        label_info = labels.get(category)
        if category == "Normal_Videos_event":
            target_label = 0
        elif isinstance(label_info, dict):
            target_label = label_info.get("target_class")
        else:
            target_label = None

        video_id = video_path.stem
        if video_id in seen_video_ids:
            raise ValueError(f"Duplicate video ID in official split: {video_id}")
        seen_video_ids.add(video_id)
        exists = video_path.is_file()
        fps, duration = _video_info(video_path) if exists else (None, None)
        is_target_positive = target_label == 1
        frame_intervals = annotation["intervals"] if annotation else []
        intervals = []
        if is_target_positive and annotation and fps:
            intervals = [
                {
                    "start_s": max(0.0, (item["start_frame"] - 1) / fps),
                    "end_s": item["end_frame"] / fps,
                    "category": category,
                }
                for item in frame_intervals
            ]
        if is_target_positive:
            if intervals:
                annotation_status = "annotated"
            elif annotation and frame_intervals:
                annotation_status = "fps_unavailable_for_temporal_annotation"
            else:
                annotation_status = "missing_temporal_annotation"
        elif target_label == 0:
            annotation_status = "negative_video"
        else:
            annotation_status = "excluded_by_label_mapping"

        if not exists:
            missing.append(relative)
        rows.append(
            {
                "video_id": video_id,
                "relative_path": relative,
                "video_path": str(video_path),
                "category": category,
                "target_label": "" if target_label is None else int(target_label),
                "split": "official_test",
                "annotation_status": annotation_status,
                "file_exists": str(exists).lower(),
                "fps": "" if fps is None else round(fps, 6),
                "duration_s": "" if duration is None else round(duration, 6),
                "ground_truth_intervals_json": json.dumps(
                    intervals, separators=(",", ":")
                ),
            }
        )

    metadata = {
        "dataset": "UCF-Crime",
        "split": "official anomaly-detection test list",
        "videos_root": str(root),
        "split_file_sha256": sha256_file(split_file),
        "temporal_annotations_sha256": sha256_file(temporal_annotations_file),
        "mapping_file_sha256": sha256_file(mapping_file),
        "total_split_entries": len(entries),
        "missing_video_files": missing,
        "target_violence_categories": sorted(VIOLENCE_CATEGORIES),
        "negative_categories_follow_mapping": True,
        "note": "UCF-Crime video paths are kept on the local machine; no dataset videos are copied into this repository.",
    }
    return rows, metadata


def write_manifest(rows: list[dict], path: str | Path, metadata: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label_text = row.get("target_label", "").strip()
            row["target_label"] = int(label_text) if label_text else None
            row["file_exists"] = row.get("file_exists", "false").lower() == "true"
            for field in ("fps", "duration_s"):
                value = row.get(field, "").strip()
                row[field] = float(value) if value else None
            row["ground_truth_intervals"] = json.loads(
                row.pop("ground_truth_intervals_json", "[]") or "[]"
            )
            rows.append(row)
    return rows


def load_window_predictions(path: str | Path) -> dict[str, list[dict]]:
    predictions: dict[str, list[dict]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed prediction JSON on line {line_no}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Prediction row on line {line_no} must be an object")
            video_id = row.get("video_id")
            probability = row.get("violence_probability")
            if (
                not isinstance(video_id, str)
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
            ):
                raise ValueError(f"Invalid prediction row on line {line_no}")
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability on line {line_no}")
            if row.get("model_status") != "valid":
                continue
            for field in ("window_start_s", "window_end_s"):
                value = row.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"Invalid {field} on line {line_no}")
            if row["window_start_s"] < 0 or row["window_end_s"] < row["window_start_s"]:
                raise ValueError(f"Reversed prediction window on line {line_no}")
            for field in ("inference_latency_ms", "capture_to_score_ms"):
                value = row.get(field)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"Invalid {field} on line {line_no}")
            predictions.setdefault(video_id, []).append(row)
    for rows in predictions.values():
        rows.sort(key=lambda item: (item["window_start_s"], item["window_end_s"]))
    return predictions


def events_from_windows(
    windows: Iterable[dict], threshold: float, *, merge_gap_s: float = 1.0
) -> list[dict]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not math.isfinite(merge_gap_s) or merge_gap_s < 0:
        raise ValueError("merge_gap_s must be a finite nonnegative number")
    active = [row for row in windows if row["violence_probability"] >= threshold]
    events: list[dict] = []
    for row in active:
        start, end = float(row["window_start_s"]), float(row["window_end_s"])
        score = float(row["violence_probability"])
        if events and start <= events[-1]["end_s"] + merge_gap_s:
            events[-1]["end_s"] = max(events[-1]["end_s"], end)
            events[-1]["peak_score"] = max(events[-1]["peak_score"], score)
        else:
            events.append({"start_s": start, "end_s": end, "peak_score": score})
    return events


def _iou(left: dict, right: dict) -> float:
    intersection = max(
        0.0, min(left["end_s"], right["end_s"]) - max(left["start_s"], right["start_s"])
    )
    union = max(left["end_s"], right["end_s"]) - min(left["start_s"], right["start_s"])
    return intersection / union if union > 0 else 0.0


def _binary_metrics(samples: list[tuple[int, float, str]], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for label, score, _ in samples:
        predicted = score >= threshold
        if label == 1 and predicted:
            tp += 1
        elif label == 0 and predicted:
            fp += 1
        elif label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = None
    if len({label for label, _, _ in samples}) == 2:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score([s[0] for s in samples], [s[1] for s in samples]))
        except (ImportError, ValueError):
            pass
    return {
        "scored_videos": len(samples),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(samples) if samples else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auc,
        "threshold": threshold,
    }


def evaluate_ucf_run(
    manifest: list[dict],
    predictions: dict[str, list[dict]],
    *,
    threshold: float,
    merge_gap_s: float = 1.0,
    event_iou_threshold: float = 0.1,
    run_metadata: dict | None = None,
) -> dict:
    """Score video classification and event intervals without tuning on test data."""
    if (
        not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or not math.isfinite(event_iou_threshold)
        or not 0 <= event_iou_threshold <= 1
        or not math.isfinite(merge_gap_s)
        or merge_gap_s < 0
    ):
        raise ValueError("thresholds must be finite and in range")

    scored, per_video, predicted_by_video = [], [], {}
    inference_ms, capture_to_score_ms = [], []
    for item in manifest:
        video_id = item["video_id"]
        windows = predictions.get(video_id, [])
        valid = [w for w in windows if w.get("model_status") == "valid"]
        if valid:
            score = max(float(w["violence_probability"]) for w in valid)
            predicted_by_video[video_id] = events_from_windows(
                valid, threshold, merge_gap_s=merge_gap_s
            )
            if item["target_label"] in (0, 1):
                scored.append((item["target_label"], score, video_id))
            inference_ms.extend(
                float(w.get("inference_latency_ms", 0.0)) for w in valid
            )
            capture_to_score_ms.extend(
                float(w.get("capture_to_score_ms", 0.0)) for w in valid
            )
        else:
            score = None
            predicted_by_video[video_id] = []
        per_video.append(
            {
                "video_id": video_id,
                "category": item["category"],
                "target_label": item["target_label"],
                "annotation_status": item["annotation_status"],
                "scored": bool(valid),
                "max_violence_probability": score,
                "prediction": None
                if score is None or item["target_label"] not in (0, 1)
                else int(score >= threshold),
                "predicted_events": predicted_by_video[video_id],
            }
        )

    event_truth = []
    event_predictions = []
    duration_negative_s = 0.0
    duration_normal_s = 0.0
    unannotated_positive = sum(
        1
        for item in manifest
        if item["file_exists"]
        and item["target_label"] == 1
        and item["annotation_status"] != "annotated"
    )
    run_metadata = run_metadata or {}
    failed_video_ids = {
        item.get("video_id")
        for item in run_metadata.get("failed_videos", [])
        if isinstance(item, dict) and isinstance(item.get("video_id"), str)
    }
    failed_video_ids.update(
        item
        for item in run_metadata.get("missing_video_ids", [])
        if isinstance(item, str)
    )
    event_scored_video_ids = {
        video_id
        for video_id, windows in predictions.items()
        if any(window.get("model_status") == "valid" for window in windows)
    } - failed_video_ids
    for item in manifest:
        if not item["file_exists"] or item["target_label"] not in (0, 1):
            continue
        video_id = item["video_id"]
        if video_id not in event_scored_video_ids:
            continue
        if item["target_label"] == 1:
            if item["annotation_status"] != "annotated":
                continue
            event_truth.extend(
                {**interval, "video_id": video_id}
                for interval in item["ground_truth_intervals"]
            )
        else:
            duration = float(item["duration_s"] or 0.0)
            duration_negative_s += duration
            if item["category"] in NORMAL_CATEGORIES:
                duration_normal_s += duration
        event_predictions.extend(
            {**event, "video_id": video_id}
            for event in predicted_by_video.get(video_id, [])
        )

    candidates_by_prediction = {}
    for pi, pred in enumerate(event_predictions):
        for gi, truth in enumerate(event_truth):
            if pred["video_id"] == truth["video_id"]:
                overlap = _iou(pred, truth)
                if overlap >= event_iou_threshold:
                    candidates_by_prediction.setdefault(pi, []).append((overlap, gi))

    # Use augmenting paths so an early high-IoU pair cannot reduce the number
    # of valid one-to-one matches for later events.
    truth_to_prediction = {}

    def assign(pi: int, visited: set[int]) -> bool:
        candidates = sorted(candidates_by_prediction.get(pi, []), reverse=True)
        for _, gi in candidates:
            if gi in visited:
                continue
            visited.add(gi)
            previous = truth_to_prediction.get(gi)
            if previous is None or assign(previous, visited):
                truth_to_prediction[gi] = pi
                return True
        return False

    for pi in range(len(event_predictions)):
        assign(pi, set())

    matches = []
    for gi, pi in sorted(truth_to_prediction.items()):
        pred, truth = event_predictions[pi], event_truth[gi]
        overlap = _iou(pred, truth)
        matches.append(
            {
                "video_id": truth["video_id"],
                "ground_truth_start_s": truth["start_s"],
                "predicted_start_s": pred["start_s"],
                "temporal_iou": overlap,
                "detection_delay_s": max(0.0, pred["start_s"] - truth["start_s"]),
            }
        )

    event_tp = len(matches)
    event_fp = len(event_predictions) - event_tp
    event_fn = len(event_truth) - event_tp
    event_precision = event_tp / (event_tp + event_fp) if event_tp + event_fp else 0.0
    event_recall = event_tp / (event_tp + event_fn) if event_tp + event_fn else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    category_by_video = {item["video_id"]: item["category"] for item in manifest}
    label_by_video = {item["video_id"]: item["target_label"] for item in manifest}
    false_alarms_normal = sum(
        1
        for event in event_predictions
        if category_by_video.get(event["video_id"]) in NORMAL_CATEGORIES
    )
    false_alarms_negative = sum(
        1 for event in event_predictions if label_by_video.get(event["video_id"]) == 0
    )
    false_alarms_positive_off_event = event_fp - false_alarms_negative
    false_alarms_per_normal_hour = (
        false_alarms_normal / (duration_normal_s / 3600) if duration_normal_s else None
    )
    false_alarms_per_negative_hour = (
        false_alarms_negative / (duration_negative_s / 3600)
        if duration_negative_s
        else None
    )

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def percentile95(values: list[float]) -> float | None:
        return float(np.percentile(values, 95)) if values else None

    return {
        "schema_version": 1,
        "dataset": "UCF-Crime",
        "split": "official anomaly-detection test split",
        "evaluation_scope": "Violence versus non-violence according to configs/behavior/dataset_mapping.yaml; no model training or threshold selection is performed on this split.",
        "video_metrics": {
            **_binary_metrics(scored, threshold),
            "unscored_in_scope_videos": sum(
                1
                for item in per_video
                if item["target_label"] in (0, 1) and not item["scored"]
            ),
            "excluded_categories": sorted(
                {item["category"] for item in manifest if item["target_label"] is None}
            ),
        },
        "event_metrics": {
            "temporal_iou_threshold": event_iou_threshold,
            "merge_gap_s": merge_gap_s,
            "annotated_ground_truth_events": len(event_truth),
            "predicted_events": len(event_predictions),
            "true_positive": event_tp,
            "false_positive": event_fp,
            "false_negative": event_fn,
            "precision": event_precision,
            "recall": event_recall,
            "f1": event_f1,
            "false_alarms_on_normal_videos": false_alarms_normal,
            "false_alarms_on_all_negative_videos": false_alarms_negative,
            "false_alarms_on_positive_videos_outside_ground_truth": false_alarms_positive_off_event,
            "false_alarms_per_normal_hour": false_alarms_per_normal_hour,
            "false_alarms_per_negative_hour": false_alarms_per_negative_hour,
            "mean_detection_delay_s": mean(
                [match["detection_delay_s"] for match in matches]
            ),
            "unannotated_positive_videos_excluded": unannotated_positive,
            "unscored_annotated_positive_videos": sum(
                1
                for item in manifest
                if item["target_label"] == 1
                and item["annotation_status"] == "annotated"
                and item["video_id"] not in event_scored_video_ids
            ),
            "matches": matches,
        },
        "runtime_metrics": {
            "mean_model_inference_ms": mean(inference_ms),
            "p95_model_inference_ms": percentile95(inference_ms),
            "mean_capture_to_score_ms": mean(capture_to_score_ms),
            "p95_capture_to_score_ms": percentile95(capture_to_score_ms),
            **run_metadata,
        },
        "per_video": per_video,
        "limitations": [
            "UCF-Crime supplies video-level and temporal event annotations, not the bounding boxes needed to score weapon-object detection precision or recall.",
            "This temporal classifier evaluation does not establish tracking ID accuracy or campus generalization.",
            "Unscored videos are missing predictions, not valid negative predictions.",
            "Anomaly classes mapped to non-violence remain negative for this behavior model; they are not scored as violence events.",
        ],
    }
