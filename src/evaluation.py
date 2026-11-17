"""Small, reusable helpers for detector experiments and threshold evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_report_paths(
    outputs: list[Path], inputs: list[Path], *, overwrite: bool = False
) -> None:
    """Protect experiment inputs and previously recorded results before inference."""
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Report and prediction cache must use different output paths.")
    for index, path in enumerate(resolved_outputs):
        if path.exists() and any(
            other.exists() and path.samefile(other)
            for other in resolved_outputs[:index]
        ):
            raise ValueError(
                "Report and prediction cache must use different output files."
            )
    protected = {path.resolve() for path in inputs}
    for path in resolved_outputs:
        if path in protected or (
            path.exists()
            and any(source.exists() and path.samefile(source) for source in protected)
        ):
            raise ValueError(f"An output would overwrite an experiment input: {path}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Output path is not a file: {path}")
            if not overwrite:
                raise FileExistsError(
                    f"Report already exists: {path}. Choose another path or use --overwrite."
                )


def json_ready(value):
    """Convert NumPy values to ordinary JSON values; use null for unavailable metrics."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resolved_dataset_config(data_path: Path) -> dict:
    """Use one path convention for our tools and the Ultralytics validator."""
    config = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must be a mapping.")
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = data_path.parent / root
    return {**config, "path": str(root.resolve())}


def prepare_dataset_yaml(data_path: Path, destination_dir: Path) -> Path:
    """Write an absolute-root copy; never change the user's original dataset YAML.

    Ultralytics resolves relative roots differently from our config-relative
    convention. A stable resolved copy also remains available after training.
    """
    config = resolved_dataset_config(data_path)
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    destination_dir.mkdir(parents=True, exist_ok=True)
    path = destination_dir / f"{data_path.stem}_{digest}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def load_split(data_path: Path, split: str) -> tuple[dict, list[dict]]:
    """Read this project's YOLO directory layout and require matching label files.

    Relative dataset roots are resolved from the YAML's directory. The generated
    grouped data.yaml uses an absolute root and is also safe for Ultralytics.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    config = resolved_dataset_config(data_path)
    if not isinstance(config.get(split), str):
        raise ValueError(
            "Expected a YAML mapping with a single directory for the requested split."
        )
    root = Path(config["path"])
    names = config["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names
    names = {int(k): str(v) for k, v in names.items()}
    if not names or set(names) != set(range(len(names))):
        raise ValueError("Dataset classes must have consecutive IDs starting at zero.")
    image_dir = (root / config[split]).resolve()
    label_dir = root / "labels" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    entries = []
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative = path.relative_to(image_dir)
            label = label_dir / relative.with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing label file: {label}")
            entries.append(
                {"image_path": path, "label_path": label, "image": relative.as_posix()}
            )
    if not entries:
        raise ValueError(f"No images found in {image_dir}")
    return {
        "names": names,
        "root": root,
        "image_dir": image_dir,
        "split": split,
    }, entries


def split_fingerprint(data_path: Path, entries: list[dict]) -> dict:
    """Hash file contents as well as names so changed images/labels invalidate an experiment."""
    digest = hashlib.sha256()
    raw_names = resolved_dataset_config(data_path)["names"]
    names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else raw_names
    names = {int(key): str(value) for key, value in names.items()}
    digest.update(json.dumps(names, sort_keys=True).encode("utf-8"))
    for entry in sorted(entries, key=lambda item: item["image"]):
        image_hash = file_sha256(entry["image_path"])
        label_hash = file_sha256(entry["label_path"])
        digest.update(f"{entry['image']}\0{image_hash}\0{label_hash}\n".encode("utf-8"))
    return {
        "dataset_yaml": str(data_path),
        "dataset_yaml_sha256": file_sha256(data_path),
        "split_content_sha256": digest.hexdigest(),
        "images": len(entries),
        "class_names": names,
    }


def read_ground_truth(
    label_path: Path, width: int, height: int, names: dict
) -> list[dict]:
    objects = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            fields = [float(value) for value in line.split()]
            if len(fields) != 5 or not all(math.isfinite(value) for value in fields):
                raise ValueError("expected five finite values")
            class_value, x, y, box_w, box_h = fields
            if not class_value.is_integer() or int(class_value) not in names:
                raise ValueError("unknown class ID")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
                raise ValueError("invalid normalized box")
            # The grouped dataset should already be clean. A tiny rounding error
            # at the border is harmless; genuinely invalid boxes stop evaluation.
            x1, y1, x2, y2 = x - box_w / 2, y - box_h / 2, x + box_w / 2, y + box_h / 2
            if min(x1, y1) < -1e-6 or max(x2, y2) > 1 + 1e-6:
                raise ValueError("box extends outside image")
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: {exc}") from exc
        objects.append(
            {
                "class_id": int(class_value),
                "bbox": [
                    max(0, x1) * width,
                    max(0, y1) * height,
                    min(1, x2) * width,
                    min(1, y2) * height,
                ],
            }
        )
    return objects


def collect_predictions(
    model,
    entries: list[dict],
    names: dict,
    *,
    confidence: float,
    imgsz: int,
    device: str | None,
    batch: int = 1,
    nms_iou: float = 0.7,
    max_det: int = 300,
    warmup_runs: int = 2,
) -> list[dict]:
    """Infer once; the returned records can be reused for every confidence threshold."""
    model_names = model.names
    model_names = (
        dict(enumerate(model_names)) if isinstance(model_names, list) else model_names
    )
    if {int(k): str(v) for k, v in model_names.items()} != names:
        raise ValueError("Model class names do not match this dataset.")
    if batch < 1:
        raise ValueError("batch must be positive")
    for _ in range(warmup_runs):
        model.predict(
            source=str(entries[0]["image_path"]),
            conf=confidence,
            imgsz=imgsz,
            device=device,
            iou=nms_iou,
            max_det=max_det,
            verbose=False,
            save=False,
        )
    records = []
    # Keep the list small: Ultralytics decodes a supplied path list into memory.
    for offset in range(0, len(entries), batch):
        chunk = entries[offset : offset + batch]
        predictions = model.predict(
            source=[str(entry["image_path"]) for entry in chunk],
            stream=True,
            conf=confidence,
            imgsz=imgsz,
            device=device,
            iou=nms_iou,
            max_det=max_det,
            verbose=False,
            save=False,
        )
        count = 0
        for entry, result in zip(chunk, predictions, strict=True):
            if Path(result.path).resolve() != entry["image_path"].resolve():
                raise ValueError(
                    f"Prediction path mismatch: expected {entry['image_path']}, got {result.path}"
                )
            height, width = result.orig_shape
            boxes = []
            if result.boxes is not None:
                coordinates = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for bbox, score, class_id in zip(
                    coordinates, scores, classes, strict=True
                ):
                    if (
                        not math.isfinite(float(class_id))
                        or not float(class_id).is_integer()
                    ):
                        raise ValueError(
                            f"Invalid predicted class ID in {entry['image']}: {class_id}"
                        )
                    box = {
                        "bbox": bbox.tolist(),
                        "confidence": float(score),
                        "class_id": int(class_id),
                    }
                    validate_object(box, names, prediction=True)
                    boxes.append(box)
            records.append(
                {
                    "image": entry["image"],
                    "image_path": str(entry["image_path"]),
                    "width": width,
                    "height": height,
                    "ground_truth": read_ground_truth(
                        entry["label_path"], width, height, names
                    ),
                    "predictions": boxes,
                    "model_timing_ms": dict(result.speed),
                }
            )
            count += 1
        if (offset + count) % 50 == 0 or offset + count == len(entries):
            print(f"Predicted {offset + count}/{len(entries)} images", flush=True)
    return records


def timing_summary(records: list[dict]) -> dict:
    """Report warmed model pipeline timing; this is not camera-to-display latency."""
    times = []
    for record in records:
        timing = record.get("model_timing_ms", {})
        values = [timing.get(key) for key in ("preprocess", "inference", "postprocess")]
        if all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in values
        ):
            times.append(sum(values))
    return {
        "description": "Ultralytics preprocessing + inference + postprocessing, after two warm-up predictions, batch 1. Excludes image loading and report writing.",
        "images": len(times),
        "mean_ms": float(np.mean(times)) if times else None,
        "median_ms": float(np.median(times)) if times else None,
        "p95_ms": float(np.percentile(times, 95)) if times else None,
    }


def box_iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection *= max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    predictions: list[dict],
    ground_truth: list[dict],
    iou: float = 0.5,
    class_aware: bool = True,
) -> tuple[list, list, list]:
    """Greedy confidence-order matching, within each class, with one match per GT.

    A wrong-class box is both a false positive for its prediction and a missed
    ground-truth object. Duplicate boxes cannot claim the same target twice.
    """
    if not 0 < iou <= 1:
        raise ValueError("matching IoU must be in (0, 1]")
    unmatched = set(range(len(ground_truth)))
    matches, false_positives = [], []
    order = sorted(
        range(len(predictions)), key=lambda index: -predictions[index]["confidence"]
    )
    for index in order:
        prediction = predictions[index]
        candidates = [
            target
            for target in sorted(unmatched)
            if not class_aware
            or ground_truth[target]["class_id"] == prediction["class_id"]
        ]
        best = max(
            candidates,
            key=lambda target: box_iou(
                prediction["bbox"], ground_truth[target]["bbox"]
            ),
            default=None,
        )
        if (
            best is not None
            and box_iou(prediction["bbox"], ground_truth[best]["bbox"]) >= iou
        ):
            matches.append((index, best))
            unmatched.remove(best)
        else:
            false_positives.append(index)
    return matches, false_positives, sorted(unmatched)


def count_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def validate_object(obj: dict, names: dict, *, prediction: bool = False) -> None:
    """Invalid outputs must stop evaluation, not disappear as apparent negatives."""
    class_id = obj["class_id"]
    if (
        isinstance(class_id, bool)
        or not isinstance(class_id, numbers.Integral)
        or class_id not in names
    ):
        raise ValueError(f"Invalid class ID: {class_id!r}")
    bbox = obj["bbox"]
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        raise ValueError("Bounding boxes must contain four finite coordinates.")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("Bounding boxes must have positive width and height.")
    if prediction:
        confidence = float(obj["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Prediction confidence must be finite and in [0, 1].")


def evaluate_at_threshold(
    records: list[dict],
    names: dict,
    threshold: float,
    iou: float = 0.5,
    include_images: bool = True,
) -> dict:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be in [0, 1]")
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in names}
    class_ids = sorted(names)
    matrix_index = {class_id: index for index, class_id in enumerate(class_ids)}
    background = len(class_ids)
    confusion = np.zeros((background + 1, background + 1), dtype=int)
    negative_images = negative_fp_images = images_with_fp = images_with_fn = 0
    images = []
    for record in records:
        ground_truth = record["ground_truth"]
        for obj in ground_truth:
            validate_object(obj, names)
        for obj in record["predictions"]:
            validate_object(obj, names, prediction=True)
        # Ultralytics retains scores strictly above its confidence cutoff.
        predictions = [p for p in record["predictions"] if p["confidence"] > threshold]
        matches, fp, fn = match_detections(predictions, ground_truth, iou)
        for prediction, _ in matches:
            counts[predictions[prediction]["class_id"]]["tp"] += 1
        for prediction in fp:
            counts[predictions[prediction]["class_id"]]["fp"] += 1
        for target in fn:
            counts[ground_truth[target]["class_id"]]["fn"] += 1
        images_with_fp += bool(fp)
        images_with_fn += bool(fn)
        if not ground_truth:
            negative_images += 1
            negative_fp_images += bool(predictions)
        # This separate geometric matching shows wrong-class boxes off diagonal.
        # The P/R/F1 counts above always use class-aware matching.
        geometry_matches, geometry_fp, geometry_fn = match_detections(
            predictions, ground_truth, iou, class_aware=False
        )
        for prediction, target in geometry_matches:
            confusion[
                matrix_index[predictions[prediction]["class_id"]],
                matrix_index[ground_truth[target]["class_id"]],
            ] += 1
        for prediction in geometry_fp:
            confusion[
                matrix_index[predictions[prediction]["class_id"]], background
            ] += 1
        for target in geometry_fn:
            confusion[background, matrix_index[ground_truth[target]["class_id"]]] += 1
        if include_images:
            images.append(
                {
                    "image": record["image"],
                    "image_path": record.get("image_path"),
                    "ground_truth": ground_truth,
                    "predictions": predictions,
                    "matches": matches,
                    "false_positive_indices": fp,
                    "missed_ground_truth_indices": fn,
                }
            )
    per_class = [
        {"class_id": class_id, "class_name": names[class_id], **count_metrics(**values)}
        for class_id, values in sorted(counts.items())
    ]
    supported = [row for row in per_class if row["tp"] + row["fn"] > 0]
    total = {key: sum(row[key] for row in per_class) for key in ("tp", "fp", "fn")}
    result = {
        "confidence_threshold": threshold,
        "matching_iou": iou,
        "matching_method": "Class-aware, descending-confidence, one-to-one greedy IoU matching.",
        "aggregate": {
            **count_metrics(**total),
            "macro_f1": float(np.mean([row["f1"] for row in supported]))
            if supported
            else None,
            "macro_recall": float(np.mean([row["recall"] for row in supported]))
            if supported
            else None,
            "macro_classes": [row["class_id"] for row in supported],
            "images": len(records),
            "images_with_false_positives": images_with_fp,
            "images_with_missed_objects": images_with_fn,
            "negative_images": negative_images,
            "negative_images_with_false_positives": negative_fp_images,
            "negative_image_false_positive_rate": negative_fp_images / negative_images
            if negative_images
            else None,
        },
        "per_class": per_class,
        "confusion_matrix": {
            "counts": confusion.tolist(),
            "labels": [names[key] for key in class_ids] + ["background"],
            "rows": "predicted",
            "columns": "ground truth",
            "matching_method": "Class-agnostic, descending-confidence, one-to-one greedy IoU matching for diagnosis.",
        },
    }
    if include_images:
        result["images"] = images
    return result


def select_threshold(
    records: list[dict],
    names: dict,
    thresholds: list[float],
    *,
    split: str,
    iou: float = 0.5,
) -> tuple[dict, list[dict]]:
    """Pick macro F1 on validation only; tie-break by recall, then lower threshold."""
    if split != "val":
        raise ValueError("Threshold selection is allowed on validation data only.")
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    results = [
        evaluate_at_threshold(records, names, value, iou, include_images=False)
        for value in sorted(set(thresholds))
    ]
    if any(result["aggregate"]["macro_f1"] is None for result in results):
        raise ValueError(
            "Validation data needs at least one labelled object to select a threshold."
        )
    best = max(
        results,
        key=lambda result: (
            result["aggregate"]["macro_f1"],
            result["aggregate"]["macro_recall"],
            -result["confidence_threshold"],
        ),
    )
    return best, results
