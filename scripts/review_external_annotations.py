"""Validate frozen external labels and render raw/label-only review sheets.

No model is loaded and no annotations are changed. The output records integrity
checks; visual review decisions belong in a separate, dated review record.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "machete", 1: "knife", 2: "baseball bat", 3: "rifle", 4: "gun"}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_records(manifest_path, dataset):
    """Check image/label identity, dimensions, boxes, and YOLO serialization."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = manifest["records"]
    if not records:
        raise ValueError("Annotation manifest contains no records")
    seen = set()
    for record in records:
        name = record["image"]
        if (
            Path(name).name != name
            or "/" in name
            or "\\" in name
            or name.casefold() in seen
        ):
            raise ValueError(
                f"Image name must be unique and contain no directory: {name}"
            )
        seen.add(name.casefold())
        image_path = dataset / "images/test" / name
        label_path = dataset / "labels/test" / Path(name).with_suffix(".txt")
        if file_hash(image_path) != record["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {name}")
        if file_hash(label_path) != record["label_sha256"]:
            raise ValueError(f"Label hash mismatch: {name}")
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        if (width, height) != (record["width"], record["height"]):
            raise ValueError(f"Image dimensions disagree with manifest: {name}")
        annotations = record["annotations"]
        if not annotations and not record["explicit_background_review"]:
            raise ValueError(f"Empty label has no recorded background review: {name}")
        rows = [
            line.split() for line in label_path.read_text().splitlines() if line.strip()
        ]
        if len(rows) != len(annotations):
            raise ValueError(f"Label row count disagrees with manifest: {name}")
        for annotation, row in zip(annotations, rows):
            class_id = annotation["class_id"]
            if type(class_id) is not int or class_id not in CLASS_NAMES:
                raise ValueError(f"Unsupported class ID: {name}")
            box = annotation["bbox_xyxy"]
            if len(box) != 4 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in box
            ):
                raise ValueError(f"Invalid box coordinates: {name}")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Box outside image or empty: {name}")
            expected = [
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
            if len(row) != 5 or row[0] != str(class_id):
                raise ValueError(f"YOLO class/column mismatch: {name}")
            values = [float(value) for value in row[1:]]
            if not all(
                math.isfinite(actual) and abs(actual - wanted) <= 1e-7
                for actual, wanted in zip(values, expected)
            ):
                raise ValueError(f"YOLO coordinates disagree with manifest: {name}")
    return records


def render_sheet(records, dataset, output, *, labels):
    # Keep portrait frames tall enough to inspect the scene, without stretching.
    portrait = records[0]["height"] > records[0]["width"]
    cell_width, cell_height = (360, 640) if portrait else (640, 360)
    columns = 3 if portrait else 2
    rows = math.ceil(len(records) / columns)
    caption_height = 34
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * (cell_height + caption_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, record in enumerate(records):
        image_path = dataset / "images/test" / record["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if labels:
            overlay = ImageDraw.Draw(image)
            for annotation in record["annotations"]:
                overlay.rectangle(
                    annotation["bbox_xyxy"], outline=(0, 220, 255), width=3
                )
        image.thumbnail((cell_width, cell_height))
        x = index % columns * cell_width
        y = index // columns * (cell_height + caption_height)
        sheet.paste(image, (x + (cell_width - image.width) // 2, y))
        suffix = " | labels" if labels else " | raw"
        draw.text(
            (x + 6, y + cell_height + 4),
            record["image"] + suffix,
            fill="black",
            font=font,
        )
    sheet.save(output, quality=95)


def ambiguity_diagnostic(records, evaluation):
    """Partition saved matches using the original flags; never alter labels."""
    images = evaluation["fixed_operating_point"]["images"]
    by_image = {item["image"]: item for item in images}
    if len(by_image) != len(images) or set(by_image) != {
        record["image"] for record in records
    }:
        raise ValueError("Evaluation images differ from the annotation manifest")
    groups = {"unflagged_positive": [], "ambiguous_positive": [], "background": []}
    for record in records:
        item = by_image[record["image"]]
        if len(item["ground_truth"]) != len(record["annotations"]):
            raise ValueError(f"Saved evaluation annotations differ: {record['image']}")
        for ground_truth, annotation in zip(
            item["ground_truth"], record["annotations"]
        ):
            if ground_truth["class_id"] != annotation["class_id"] or any(
                abs(a - b) > 1e-4
                for a, b in zip(ground_truth["bbox"], annotation["bbox_xyxy"])
            ):
                raise ValueError(f"Saved evaluation boxes differ: {record['image']}")
        group = (
            "background"
            if not record["annotations"]
            else (
                "ambiguous_positive"
                if record["boundary_ambiguity"]
                else "unflagged_positive"
            )
        )
        groups[group].append(item)
    return {
        "interpretation": "Post-hoc diagnostic of existing predictions, partitioned by the original pre-inference ambiguity flags; not a new accuracy estimate or corrected evaluation.",
        "confidence_threshold": evaluation["fixed_operating_point"][
            "confidence_threshold"
        ],
        "matching_iou": evaluation["fixed_operating_point"]["matching_iou"],
        "groups": {
            group: {
                "images": len(items),
                "tp": sum(len(item["matches"]) for item in items),
                "fp": sum(len(item["false_positive_indices"]) for item in items),
                "fn": sum(len(item["missed_ground_truth_indices"]) for item in items),
            }
            for group, items in groups.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "reports/experiments/external_pilot_annotations.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data/external/public_video_pilot/dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/external_label_review_20261117",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        help="Optional full saved evaluation for a post-hoc ambiguity diagnostic; no inference is run",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Review output already exists; choose a fresh directory.")
    try:
        records = validate_records(args.manifest, args.dataset)
        diagnostic = None
        if args.evaluation:
            diagnostic = ambiguity_diagnostic(
                records, json.loads(args.evaluation.read_text(encoding="utf-8"))
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    grouped = defaultdict(list)
    for record in records:
        grouped[record["source_id"]].append(record)
    args.output.mkdir(parents=True)
    sheets = []
    for source, group in grouped.items():
        if not source or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in source
        ):
            raise ValueError(f"Invalid source identifier: {source}")
        for start in range(0, len(group), 6):
            for labels in (False, True):
                name = f"{source}_{start // 6 + 1}_{'labels' if labels else 'raw'}.jpg"
                render_sheet(
                    group[start : start + 6],
                    args.dataset,
                    args.output / name,
                    labels=labels,
                )
                sheets.append(name)
    summary = {
        "manifest_sha256": file_hash(args.manifest),
        "frames_checked": len(records),
        "images_and_labels_match_manifest": True,
        "dimensions_and_boxes_valid": True,
        "label_rows_match_manifest": True,
        "raw_and_label_only_sheets": sheets,
        "interpretation": "Automated integrity checks only. Rendering does not constitute visual or human approval of labels.",
    }
    if diagnostic is not None:
        summary["saved_evaluation_sha256"] = file_hash(args.evaluation)
        summary["ambiguity_diagnostic"] = diagnostic
    (args.output / "integrity.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
