"""Build diagnostic contact sheets from a saved evaluate_detector.py report.

These are selected examples for inspection, not a representative accuracy sample.
No inference runs here: every box comes from the report's recorded predictions or labels.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "missed": (50, 50, 235),
    "false_positive": (0, 175, 255),
    "correct": (70, 200, 70),
}
TILE_WIDTH, TILE_HEIGHT = 480, 370


def error_counts(record):
    return {
        "missed": len(record["missed_ground_truth_indices"]),
        "false_positive": len(record["false_positive_indices"]),
        "correct": len(record["matches"]),
    }


def select_examples(records, limit=12):
    """Rank each category by its count, then other errors, then filename."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    selected = {}
    for category in COLORS:
        candidates = [
            record for record in records if error_counts(record)[category] > 0
        ]
        selected[category] = sorted(
            candidates,
            key=lambda record: (
                -error_counts(record)[category],
                -(
                    error_counts(record)["missed"]
                    + error_counts(record)["false_positive"]
                ),
                record["image"],
            ),
        )[:limit]
    return selected


def render_tile(record, names):
    source = Path(record["image_path"])
    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Could not decode source image: {source}")
    height, width = frame.shape[:2]
    scale = min(TILE_WIDTH / width, (TILE_HEIGHT - 58) / height)
    resized = cv2.resize(
        frame, (max(1, round(width * scale)), max(1, round(height * scale)))
    )
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 28, dtype=np.uint8)
    left = (TILE_WIDTH - resized.shape[1]) // 2
    tile[58 : 58 + resized.shape[0], left : left + resized.shape[1]] = resized
    counts = error_counts(record)
    cv2.putText(
        tile,
        record["image"][-65:],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (240, 240, 240),
        1,
    )
    text = f"Missed: {counts['missed']} | FP: {counts['false_positive']} | Matched: {counts['correct']}"
    cv2.putText(tile, text, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    marked = []
    for index in record["missed_ground_truth_indices"]:
        marked.append(("missed", record["ground_truth"][index]))
    for index in record["false_positive_indices"]:
        marked.append(("false_positive", record["predictions"][index]))
    for prediction_index, _ in record["matches"]:
        marked.append(("correct", record["predictions"][prediction_index]))
    for category, item in marked:
        x1, y1, x2, y2 = item["bbox"]
        p1 = (left + round(x1 * scale), 58 + round(y1 * scale))
        p2 = (left + round(x2 * scale), 58 + round(y2 * scale))
        label = f"{category.replace('_', ' ')} {names[item['class_id']]}"
        if "confidence" in item:
            label += f" {item['confidence']:.2f}"
        color = COLORS[category]
        cv2.rectangle(tile, p1, p2, color, 3 if category == "missed" else 1)
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        text_x = max(0, min(p1[0], TILE_WIDTH - label_width - 4))
        label_offset = {"missed": 0, "false_positive": 15, "correct": 30}[category]
        text_y = min(TILE_HEIGHT - 4, max(72, p1[1] - 4) + label_offset)
        cv2.rectangle(
            tile,
            (text_x, text_y - 13),
            (text_x + label_width + 3, text_y + 3),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            tile, label, (text_x + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
    return tile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", type=Path, help="JSON produced by evaluate_detector.py"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "model_review"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Maximum images per category (default: 12)",
    )
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        fixed = report["fixed_operating_point"]
        records = fixed["images"]
        names = {row["class_id"]: row["class_name"] for row in fixed["per_class"]}
        selected = select_examples(records, args.limit)
        for examples in selected.values():
            for record in examples:
                if (
                    not record.get("image_path")
                    or not Path(record["image_path"]).is_file()
                ):
                    raise FileNotFoundError(
                        f"Source image missing for {record['image']}: {record.get('image_path')}"
                    )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for category, examples in selected.items():
            eligible = sum(error_counts(record)[category] > 0 for record in records)
            columns = min(3, max(1, len(examples)))
            sheet_rows = max(1, (len(examples) + columns - 1) // columns)
            sheet = np.full(
                (80 + sheet_rows * TILE_HEIGHT, columns * TILE_WIDTH, 3),
                28,
                dtype=np.uint8,
            )
            title = f"{category.replace('_', ' ').title()}: {len(examples)} selected / {eligible} eligible images"
            cv2.putText(
                sheet,
                title,
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
            )
            cv2.putText(
                sheet,
                "Diagnostic selection, not an accuracy estimate.",
                (12, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (245, 245, 245),
                1,
            )
            for text, color, x in (
                ("Red: missed GT", COLORS["missed"], 12),
                ("Orange: FP", COLORS["false_positive"], 165),
                ("Green: matched", COLORS["correct"], 300),
            ):
                cv2.putText(
                    sheet, text, (x, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
            for index, record in enumerate(examples):
                x, y = (
                    (index % columns) * TILE_WIDTH,
                    80 + (index // columns) * TILE_HEIGHT,
                )
                sheet[y : y + TILE_HEIGHT, x : x + TILE_WIDTH] = render_tile(
                    record, names
                )
                rows.append(
                    {
                        "category": category,
                        "rank": index + 1,
                        "image": record["image"],
                        "source": record["image_path"],
                        **error_counts(record),
                        "sheet": f"{category}.jpg",
                        "report": str(args.report.resolve()),
                    }
                )
            output = args.output_dir / f"{category}.jpg"
            if not cv2.imwrite(str(output), sheet):
                raise OSError(f"Could not save contact sheet: {output}")
            print(
                f"Saved {output}: {len(examples)} selected from {eligible} eligible images"
            )
        with (args.output_dir / "index.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "category",
                    "rank",
                    "image",
                    "source",
                    "missed",
                    "false_positive",
                    "correct",
                    "sheet",
                    "report",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Source report: {args.report.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
