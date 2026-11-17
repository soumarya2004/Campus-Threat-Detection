"""Choose a confidence threshold on validation data only, then save the decision.

This does not edit the application's detector config. Review the JSON result and
freeze the chosen threshold before evaluating the untouched test split.
"""

from __future__ import annotations

import argparse
import json
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
    repo_path,
    select_threshold,
    split_fingerprint,
    timing_summary,
    validate_report_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/dangerous_items_grouped_v1/data.yaml"),
    )
    parser.add_argument(
        "--split",
        choices=["val"],
        default="val",
        help="Tuning the test split is deliberately disallowed.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[round(0.1 + i * 0.05, 2) for i in range(15)],
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU required for a true-positive match."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/validation_threshold.json"),
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        help="Optional low-confidence prediction cache JSON.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing report/cache files."
    )
    args = parser.parse_args()
    if any(not 0 < value <= 1 for value in args.thresholds):
        parser.error("thresholds must be in (0, 1]")
    if args.imgsz < 32 or not 0 < args.iou <= 1:
        parser.error("imgsz must be at least 32 and matching IoU must be in (0, 1]")
    model_path, data_path = repo_path(args.model), repo_path(args.data)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    info, entries = load_split(data_path, args.split)
    output = repo_path(args.output)
    cache = repo_path(args.save_predictions) if args.save_predictions else None
    protected = [model_path, data_path]
    protected.extend(
        entry[key] for entry in entries for key in ("image_path", "label_path")
    )
    validate_report_paths(
        [output] + ([cache] if cache else []), protected, overwrite=args.overwrite
    )
    provenance = split_fingerprint(data_path, entries)
    provenance.update(
        {"model_path": str(model_path), "model_sha256": file_sha256(model_path)}
    )
    cache_confidence = min(0.001, min(args.thresholds) / 2)
    inference = {
        "image_size": args.imgsz,
        "device": args.device,
        "nms_iou": 0.7,
        "max_det": 300,
        "prediction_batch": 1,
        "cache_confidence": cache_confidence,
    }

    import ultralytics
    from ultralytics import YOLO

    records = collect_predictions(
        YOLO(str(model_path)),
        entries,
        info["names"],
        confidence=cache_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )
    best, candidates = select_threshold(
        records, info["names"], args.thresholds, split=args.split, iou=args.iou
    )
    report = {
        "split": "val",
        "purpose": "Choose once on validation, freeze before test evaluation.",
        "selection_criterion": "Highest macro F1 over classes with validation ground truth; ties prefer macro recall, then lower confidence.",
        "selected_threshold": best["confidence_threshold"],
        "matching_iou": args.iou,
        "provenance": provenance,
        "inference": inference,
        "ultralytics_version": ultralytics.__version__,
        "candidates": candidates,
        "warmed_prediction_timing": timing_summary(records),
        "selected_operating_point": evaluate_at_threshold(
            records, info["names"], best["confidence_threshold"], args.iou
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_ready(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                json_ready(
                    {
                        "provenance": provenance,
                        "inference": inference,
                        "records": records,
                    }
                ),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    print(f"Selected confidence: {best['confidence_threshold']:.2f}")
    print(json.dumps(best["aggregate"], indent=2))
    print(f"Saved validation decision: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
