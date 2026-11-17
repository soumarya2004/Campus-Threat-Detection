"""Report standard mAP and true fixed-threshold precision/recall separately.

Example:
    python scripts/evaluate_detector.py --model runs/.../best.pt --split test \
        --threshold-file reports/experiments/validation_threshold.json \
        --output reports/experiments/grouped_test.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (
    collect_predictions,
    evaluate_at_threshold,
    file_sha256,
    json_ready,
    load_split,
    prepare_dataset_yaml,
    repo_path,
    split_fingerprint,
    timing_summary,
    validate_report_paths,
)


def read_threshold_decision(
    path: Path, model_hash: str, image_size: int
) -> tuple[float, dict]:
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("split") != "val":
        raise ValueError("Threshold decision must come from validation data.")
    if decision["provenance"]["model_sha256"] != model_hash:
        raise ValueError("Threshold was tuned for a different model checkpoint.")
    if decision["inference"]["image_size"] != image_size:
        raise ValueError("Evaluation image size must match threshold tuning.")
    threshold = float(decision["selected_threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold in decision file.")
    return threshold, decision


def verify_threshold_context(decision: dict, data_path: Path) -> None:
    """A threshold belongs to a model, preprocessing settings and validation set."""
    _, entries = load_split(data_path, "val")
    current = split_fingerprint(data_path, entries)
    if (
        current["split_content_sha256"]
        != decision["provenance"]["split_content_sha256"]
    ):
        raise ValueError(
            "Validation images, labels or class names changed after threshold tuning."
        )
    for key, expected in {
        "nms_iou": 0.7,
        "max_det": 300,
        "prediction_batch": 1,
    }.items():
        if decision["inference"].get(key) != expected:
            raise ValueError(f"Threshold tuning {key} does not match this evaluator.")
    cache_confidence = decision["inference"].get("cache_confidence")
    if (
        not isinstance(cache_confidence, (int, float))
        or not 0 <= cache_confidence <= decision["selected_threshold"]
    ):
        raise ValueError(
            "Threshold prediction cache was incomplete below the selected threshold."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/dangerous_items_grouped_v1/data.yaml"),
    )
    parser.add_argument("--split", choices=["val", "test"], required=True)
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument(
        "--threshold",
        type=float,
        help="Frozen operating threshold; never optimize on the test split.",
    )
    threshold_group.add_argument(
        "--threshold-file", type=Path, help="JSON produced by tune_threshold.py."
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for standard mAP validation; operating-point predictions use one image at a time.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU required for a fixed-threshold true positive.",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-predictions", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing report/cache files."
    )
    args = parser.parse_args()
    if args.imgsz < 32 or args.batch < 1 or args.workers < 0 or not 0 < args.iou <= 1:
        parser.error("Invalid image size, batch, workers or matching IoU.")
    model_path, data_path, output = (
        repo_path(args.model),
        repo_path(args.data),
        repo_path(args.output),
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model_hash = file_sha256(model_path)
    decision = None
    if args.threshold_file:
        threshold, decision = read_threshold_decision(
            repo_path(args.threshold_file), model_hash, args.imgsz
        )
        if args.iou != decision["matching_iou"]:
            parser.error("Matching IoU must match threshold tuning.")
        verify_threshold_context(decision, data_path)
    else:
        threshold = args.threshold
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        parser.error("confidence threshold must be in [0, 1]")
    info, entries = load_split(data_path, args.split)
    cache = repo_path(args.save_predictions) if args.save_predictions else None
    protected = [model_path, data_path]
    if args.threshold_file:
        protected.append(repo_path(args.threshold_file))
    protected.extend(
        entry[key] for entry in entries for key in ("image_path", "label_path")
    )
    validate_report_paths(
        [output] + ([cache] if cache else []), protected, overwrite=args.overwrite
    )
    provenance = split_fingerprint(data_path, entries)
    provenance.update({"model_path": str(model_path), "model_sha256": model_hash})
    resolved_data = prepare_dataset_yaml(
        data_path, REPO_ROOT / "outputs" / "experiment_configs"
    )
    provenance["resolved_dataset_yaml"] = str(resolved_data)
    provenance["resolved_dataset_yaml_sha256"] = file_sha256(resolved_data)

    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    # Standard AP needs low-confidence predictions across the confidence curve.
    # The library's summary P/R is not precision/recall at our frozen threshold.
    metrics = model.val(
        data=str(resolved_data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=args.plots,
        project=str(REPO_ROOT / "runs" / "detect" / "evaluations"),
        name=output.stem,
        exist_ok=False,
        verbose=False,
    )
    standard = {
        "summary": metrics.results_dict,
        "per_class": metrics.summary(),
        "speed_ms_per_image": metrics.speed,
        "artifact_directory": str(metrics.save_dir),
        "interpretation": "mAP integrates confidence thresholds. Library summary precision/recall are its curve-selected values, not the frozen operating point below.",
    }
    records = collect_predictions(
        model,
        entries,
        info["names"],
        confidence=max(threshold, 1e-6),
        imgsz=args.imgsz,
        device=args.device,
    )
    fixed = evaluate_at_threshold(records, info["names"], threshold, args.iou)
    report = {
        "split": args.split,
        "provenance": provenance,
        "ultralytics_version": ultralytics.__version__,
        "settings": {
            "image_size": args.imgsz,
            "device": args.device,
            "map_batch": args.batch,
            "prediction_batch": 1,
            "nms_iou": 0.7,
            "max_det": 300,
            "map_minimum_confidence": 0.001,
            "matching_iou": args.iou,
            "confidence_threshold": threshold,
        },
        "threshold_selection": {
            "source": str(repo_path(args.threshold_file))
            if args.threshold_file
            else "Explicit frozen CLI value; selection provenance not provided.",
            "source_sha256": file_sha256(repo_path(args.threshold_file))
            if args.threshold_file
            else None,
            "criterion": decision.get("selection_criterion") if decision else None,
            "validation_content_sha256": decision["provenance"]["split_content_sha256"]
            if decision
            else None,
        },
        "standard_detection_metrics": standard,
        "fixed_operating_point": fixed,
        "warmed_prediction_timing": timing_summary(records),
        "limitations": [
            "Image detection metrics do not establish campus threat recognition or video event accuracy.",
            "Negative-image false-positive rate is undefined (null) if this split contains no negative images.",
            "Class-aware fixed-threshold matching is explicit and may differ from the library's matching implementation.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_ready(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                json_ready({"provenance": provenance, "records": records}),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            json_ready(
                {
                    "standard": standard["summary"],
                    "fixed_operating_point": fixed["aggregate"],
                }
            ),
            indent=2,
        )
    )
    print(f"Evaluation saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
