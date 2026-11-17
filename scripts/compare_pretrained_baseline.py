"""Compare pretrained and fine-tuned YOLO11n on their two common object classes.

All validation images are retained, including images with no knife/baseball bat.
This is a descriptive fixed-threshold baseline, not a five-class mAP comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (
    evaluate_at_threshold,
    file_sha256,
    load_split,
    read_ground_truth,
    repo_path,
    split_fingerprint,
    validate_object,
    validate_report_paths,
)

COMMON_NAMES = {1: "knife", 2: "baseball bat"}
COCO_TO_PROJECT = {43: 1, 34: 2}
PROJECT_TO_PROJECT = {1: 1, 2: 2}


def validate_class_mapping(model_names, mapping):
    for source_id, project_id in mapping.items():
        if model_names.get(source_id) != COMMON_NAMES[project_id]:
            raise ValueError(
                f"Class {source_id} must mean {COMMON_NAMES[project_id]!r}"
            )


def remap_prediction(bbox, confidence, source_id, mapping):
    if not math.isfinite(float(source_id)) or not float(source_id).is_integer():
        raise ValueError("Predicted class ID must be a finite integer")
    source_id = int(source_id)
    if source_id not in mapping:
        raise ValueError(f"Model returned unrequested class {source_id}")
    prediction = {
        "class_id": mapping[source_id],
        "bbox": [float(v) for v in bbox],
        "confidence": float(confidence),
    }
    validate_object(prediction, COMMON_NAMES, prediction=True)
    return prediction


def filtered_ground_truth(label_path, width, height, dataset_names):
    # Parse with the original full class map first: invalid excluded labels must
    # not silently disappear, and other-class-only images remain in the sample.
    objects = read_ground_truth(label_path, width, height, dataset_names)
    return [obj for obj in objects if obj["class_id"] in COMMON_NAMES]


def collect_common_predictions(
    model, entries, dataset_names, mapping, *, device, batch
):
    validate_class_mapping(model.names, mapping)
    records = []
    for offset in range(0, len(entries), batch):
        chunk = entries[offset : offset + batch]
        predictions = model.predict(
            source=[str(entry["image_path"]) for entry in chunk],
            batch=batch,
            classes=sorted(mapping),
            conf=0.25,
            imgsz=640,
            rect=False,
            device=device,
            iou=0.7,
            max_det=300,
            augment=False,
            verbose=False,
            save=False,
        )
        for entry, result in zip(chunk, predictions, strict=True):
            if Path(result.path).resolve() != entry["image_path"].resolve():
                raise ValueError("Result path does not match the expected image")
            height, width = result.orig_shape
            boxes = []
            if result.boxes is not None:
                for bbox, confidence, source_id in zip(
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                    strict=True,
                ):
                    boxes.append(remap_prediction(bbox, confidence, source_id, mapping))
            records.append(
                {
                    "image": entry["image"],
                    "image_path": str(entry["image_path"]),
                    "width": width,
                    "height": height,
                    "ground_truth": filtered_ground_truth(
                        entry["label_path"], width, height, dataset_names
                    ),
                    "predictions": boxes,
                }
            )
        if len(records) % 100 == 0 or len(records) == len(entries):
            print(f"Predicted {len(records)}/{len(entries)} images", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("configs/dataset_grouped.yaml")
    )
    parser.add_argument("--pretrained", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument(
        "--finetuned", type=Path, default=Path("models/dangerous_items_yolo11n.pt")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/baseline/common_classes.json")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/baseline/common_class_predictions.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    data, output, cache = (
        repo_path(args.data),
        repo_path(args.output),
        repo_path(args.predictions),
    )
    pretrained, finetuned = repo_path(args.pretrained), repo_path(args.finetuned)
    info, entries = load_split(data, "val")
    if any(info["names"].get(key) != value for key, value in COMMON_NAMES.items()):
        parser.error("Dataset knife/baseball-bat class IDs do not match the project")
    inputs = [data, pretrained, finetuned] + [
        entry[key] for entry in entries for key in ("image_path", "label_path")
    ]
    validate_report_paths([output, cache], inputs, overwrite=args.overwrite)
    provenance = split_fingerprint(data, entries)

    import torch
    import ultralytics
    from ultralytics import YOLO

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Descriptive validation comparison at a common fixed operating point for knife and baseball bat only.",
        "split": "val",
        "provenance": provenance,
        "class_names": COMMON_NAMES,
        "settings": {
            "confidence": 0.25,
            "image_size": 640,
            "rect": False,
            "nms_iou": 0.7,
            "matching_iou": 0.5,
            "max_detections": 300,
            "prediction_batch": args.batch,
            "device": args.device,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "models": [],
        "limitations": [
            "Only two labels overlap. No pretrained claim is made for machete, rifle or gun.",
            "Validation was already used to select the fine-tuned checkpoint and deployment settings; this is descriptive, not an unbiased held-out test.",
            "Shared confidence 0.25 is fixed for this comparison, not tuned; the deployed item threshold remains 0.45.",
            "Images containing only excluded classes count as negative images for this restricted two-class task.",
            "COCO and project annotation conventions may differ; this is an application baseline, not a causal estimate of training benefit.",
            "No AP or five-class mAP is computed here.",
        ],
    }
    caches = {}
    for name, path, mapping in (
        ("general_pretrained_yolo11n", pretrained, COCO_TO_PROJECT),
        ("selected_finetuned_yolo11n", finetuned, PROJECT_TO_PROJECT),
    ):
        print(f"Evaluating {name}", flush=True)
        model = YOLO(str(path))
        records = collect_common_predictions(
            model, entries, info["names"], mapping, device=args.device, batch=args.batch
        )
        metrics = evaluate_at_threshold(
            records, COMMON_NAMES, 0.25, 0.5, include_images=False
        )
        report["models"].append(
            {
                "name": name,
                "checkpoint": str(path),
                "checkpoint_sha256": file_sha256(path),
                "source_to_project_class_ids": mapping,
                "verified_source_class_names": {
                    key: model.names[key] for key in mapping
                },
                "fixed_operating_point": metrics,
            }
        )
        caches[name] = records
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "provenance": provenance,
                "settings": report["settings"],
                "models": caches,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    report["prediction_cache"] = {"path": str(cache), "sha256": file_sha256(cache)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for model in report["models"]:
        print(model["name"], json.dumps(model["fixed_operating_point"]["aggregate"]))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
