"""Check that the shipped weights, defaults and evaluation evidence agree.

This runs without loading PyTorch or requiring a dataset. It catches accidental
checkpoint/threshold changes before a demo or handoff; it is not an accuracy test.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_release(root=REPO_ROOT):
    root = Path(root).resolve()

    def read_json(relative):
        return json.loads((root / relative).read_text(encoding="utf-8"))

    config = yaml.safe_load(
        (root / "configs/detector.yaml").read_text(encoding="utf-8")
    )
    card = read_json("models/model_card.json")
    threshold = read_json(card["selection"]["confidence"])
    resolution = read_json(card["selection"]["resolution"])
    report = read_json(card["test"]["report"])
    checks = []

    def compare(name, actual, expected):
        checks.append({"check": name, "passed": actual == expected})
        if actual != expected:
            checks[-1].update(actual=actual, expected=expected)

    configured_model = (root / config["model"]).resolve()
    card_model = (root / card["checkpoint"]).resolve()
    compare("Configured checkpoint", str(configured_model), str(card_model))
    digest = sha256(configured_model)
    compare("Checkpoint SHA-256", digest, card["checkpoint_sha256"])
    compare("Test checkpoint", report["provenance"]["model_sha256"], digest)
    compare(
        "Threshold-selection checkpoint",
        threshold["provenance"]["model_sha256"],
        digest,
    )
    compare("Resolution-selection checkpoint", resolution["model_sha256"], digest)
    compare("Threshold selected on validation", threshold["split"], "val")
    compare("Resolution selected on validation", resolution["selection_split"], "val")
    compare("Final evaluation split", report["split"], "test")
    compare("Default confidence", config["confidence"], threshold["selected_threshold"])
    compare(
        "Model-card confidence", card["inference"]["confidence"], config["confidence"]
    )
    compare(
        "Test confidence",
        report["settings"]["confidence_threshold"],
        config["confidence"],
    )
    compare(
        "Fixed-metric confidence",
        report["fixed_operating_point"]["confidence_threshold"],
        config["confidence"],
    )
    compare(
        "Default image size", config["image_size"], resolution["selected_image_size"]
    )
    for name, settings in (
        ("Model card", card["inference"]),
        ("Threshold selection", threshold["inference"]),
        ("Test", report["settings"]),
    ):
        compare(f"{name} image size", settings["image_size"], config["image_size"])
        compare(f"{name} NMS IoU", settings["nms_iou"], 0.7)
    compare("Model-card detection limit", card["inference"]["max_detections"], 300)
    compare("Test detection limit", report["settings"]["max_det"], 300)
    compare("Threshold detection limit", threshold["inference"]["max_det"], 300)
    compare(
        "Validation split identity",
        resolution["validation_content_sha256"],
        threshold["provenance"]["split_content_sha256"],
    )
    compare(
        "Test selection identity",
        report["threshold_selection"]["validation_content_sha256"],
        threshold["provenance"]["split_content_sha256"],
    )
    compare(
        "Threshold evidence hash",
        sha256(root / card["selection"]["confidence"]),
        report["threshold_selection"]["source_sha256"],
    )
    compare(
        "Dataset manifest hash",
        sha256(root / card["dataset"]["manifest"]),
        card["dataset"]["manifest_sha256"],
    )
    classes = [report["provenance"]["class_names"][str(i)] for i in range(5)]
    compare("Class order", card["classes"], classes)
    compare(
        "Card fixed-threshold metrics",
        card["test"]["fixed_threshold_metrics"],
        report["fixed_operating_point"]["aggregate"],
    )
    compare(
        "Card per-class metrics",
        card["test"]["per_class"],
        report["fixed_operating_point"]["per_class"],
    )
    metrics = report["standard_detection_metrics"]["summary"]
    compare("Card mAP50", card["test"]["map50"], metrics["metrics/mAP50(B)"])
    compare("Card mAP50-95", card["test"]["map50_95"], metrics["metrics/mAP50-95(B)"])
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    try:
        result = check_release(args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: incomplete or invalid release evidence: {exc}", file=sys.stderr)
        return 1
    for item in result["checks"]:
        if not item["passed"]:
            print(
                f"FAIL: {item['check']}: {item['actual']!r}; expected {item['expected']!r}"
            )
    print(
        f"{'PASS' if result['passed'] else 'FAIL'}: {len(result['checks'])} model-release checks"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
