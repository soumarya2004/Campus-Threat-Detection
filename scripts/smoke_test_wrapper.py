"""Check that the real checkpoint runs twice and returns valid detection records.

This is an execution check, not an accuracy evaluation. No files are written.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.object_detector import YOLOObjectDetector

GROUPED_DATA = REPO_ROOT / "data" / "processed" / "dangerous_items_grouped_v1"
LEGACY_DATA = REPO_ROOT / "data" / "processed" / "dangerous_items_cleaned_v2"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_image(data_dir: Path) -> Path:
    image_dir = data_dir / "images" / "test"
    label_dir = data_dir / "labels" / "test"
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"No test image directory: {image_dir}. Supply an image path or --data."
        )
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise FileNotFoundError(f"No test images found in {image_dir}")
    for path in images:
        label = label_dir / (path.stem + ".txt")
        if label.is_file() and label.read_text().strip():
            return path
    return images[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Input image (default: a labelled test image)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=GROUPED_DATA if GROUPED_DATA.is_dir() else LEGACY_DATA,
        help="Dataset directory used to choose the default image",
    )
    parser.add_argument("--model", type=Path, help="Override the configured checkpoint")
    parser.add_argument(
        "--confidence", type=float, help="Override the configured confidence threshold"
    )
    parser.add_argument("--device", help="Inference device, for example cpu or 0")
    args = parser.parse_args()
    detector = None
    try:
        image_path = args.image if args.image is not None else pick_image(args.data)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = frame.shape[:2]
        print(f"Image: {image_path}")
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        detector = YOLOObjectDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        print(json.dumps(detector.get_model_info(), indent=2))
        first = detector.predict(frame, frame_id=0, timestamp=0.0)
        second = detector.predict(frame, frame_id=1, timestamp=0.033)
        print("First call includes startup; do not use its latency as a benchmark.")
        print(json.dumps(first, indent=2))
        print(
            f"Second call: {len(second['detections'])} detections, "
            f"{second['inference_time_ms']:.1f} ms, {second['model_status']}"
        )

        expected = {"bbox", "class_id", "class_name", "confidence"}
        problems = []
        for call_name, result in (("first", first), ("second", second)):
            if result["model_status"] != "ok":
                problems.append(f"{call_name} call failed: {result['error']}")
            for index, detection in enumerate(result["detections"]):
                if set(detection) != expected:
                    problems.append(f"{call_name} detection {index}: unexpected keys")
                x1, y1, x2, y2 = detection["bbox"]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    problems.append(
                        f"{call_name} detection {index}: invalid bounding box"
                    )
                if not 0 <= detection["confidence"] <= 1:
                    problems.append(
                        f"{call_name} detection {index}: invalid confidence"
                    )
            json.dumps(result, allow_nan=False)
        if first["detections"] != second["detections"]:
            problems.append("Repeated calls returned different detections.")
        print("PASS: execution and schema checks" if not problems else "FAIL")
        for problem in problems:
            print(f" - {problem}")
        return 1 if problems else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
