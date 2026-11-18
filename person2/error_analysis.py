"""Summarize video-level false positives and misses from an evaluation report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("evaluation_unit") != "source_video":
        raise ValueError(
            "Expected a source-video evaluation report from the XD MIL evaluator"
        )
    predictions = report.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Evaluation report has no per-video predictions")

    false_positives = []
    false_negatives = []
    true_positives = 0
    true_negatives = 0
    fp_categories: Counter[str] = Counter()
    fn_categories: Counter[str] = Counter()

    for row in predictions:
        label = int(row["label"])
        predicted = int(row["prediction"])
        record = {
            "source_video_id": row["source_video_id"],
            "video_path": row["video_path"],
            "category": row["category"],
            "label": label,
            "prediction": predicted,
            "violence_probability": float(row["violence_probability"]),
        }
        if label == 0 and predicted == 1:
            false_positives.append(record)
            fp_categories[record["category"]] += 1
        elif label == 1 and predicted == 0:
            false_negatives.append(record)
            fn_categories[record["category"]] += 1
        elif label == 1:
            true_positives += 1
        else:
            true_negatives += 1

    return {
        "evaluation_unit": "source_video",
        "threshold": report.get("threshold"),
        "counts": {
            "videos": len(predictions),
            "true_positives": true_positives,
            "false_positives": len(false_positives),
            "true_negatives": true_negatives,
            "false_negatives": len(false_negatives),
        },
        "false_positives_by_category": dict(fp_categories),
        "false_negatives_by_category": dict(fn_categories),
        "false_positive_examples": false_positives[:20],
        "false_negative_examples": false_negatives[:20],
        "limitations": [
            "This report is video-level; it does not calculate event detection delay.",
            "False alarm causes require manual review and are not inferred from labels.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("logs/xd_test_evaluation.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/xd_error_analysis.json")
    )
    args = parser.parse_args()
    if not args.report.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {args.report}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    audit = analyze_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["counts"], indent=2))
    print("Saved error analysis to:", args.output)


if __name__ == "__main__":
    main()
