"""Smoke test: run the real checkpoint through YOLOObjectDetector.

Usage: python scripts/smoke_test_wrapper.py [image_path]
If no image is given, the first test image with a label file listing a
detection is used. Nothing is written to disk.
"""
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.object_detector import YOLOObjectDetector  # noqa: E402

TEST_DIR = REPO_ROOT / "data" / "processed" / "dangerous_items_cleaned_v2"
IMG_DIR = TEST_DIR / "images" / "test"
LBL_DIR = TEST_DIR / "labels" / "test"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_image() -> Path:
    images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        sys.exit(f"No test images found in {IMG_DIR}")
    for p in images:
        lbl = LBL_DIR / (p.stem + ".txt")
        if lbl.is_file() and lbl.read_text().strip():
            return p
    return images[0]


def main() -> int:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else pick_image()
    frame = cv2.imread(str(img_path))
    if frame is None:
        sys.exit(f"cv2.imread could not read {img_path}")
    h, w = frame.shape[:2]
    print(f"image: {img_path}")
    print(f"shape (h, w, c): {frame.shape}, dtype: {frame.dtype}")

    lbl = LBL_DIR / (img_path.stem + ".txt")
    if lbl.is_file():
        print(f"ground-truth label lines: {len(lbl.read_text().strip().splitlines())}")
        print(f"ground-truth class ids: "
              f"{sorted({int(l.split()[0]) for l in lbl.read_text().strip().splitlines() if l.strip()})}")

    detector = YOLOObjectDetector()
    print("\n--- get_model_info() ---")
    print(json.dumps(detector.get_model_info(), indent=2))

    first = detector.predict(frame, frame_id=0, timestamp=0.0)
    print("\n--- first call (includes warm-up; do not quote as latency) ---")
    print(json.dumps(first, indent=2))

    second = detector.predict(frame, frame_id=1, timestamp=0.033)
    print("\n--- second call ---")
    print(f"detections: {len(second['detections'])}, "
          f"inference_time_ms: {second['inference_time_ms']:.1f}, "
          f"model_status: {second['model_status']}")

    expected = {"bbox", "class_id", "class_name", "confidence"}
    problems = []
    if first["model_status"] != "ok":
        problems.append(f"model_status={first['model_status']} error={first['error']}")
    for i, d in enumerate(first["detections"]):
        if set(d) != expected:
            problems.append(f"det {i}: keys {sorted(d)}")
        x1, y1, x2, y2 = d["bbox"]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            problems.append(f"det {i}: bbox {d['bbox']} outside {w}x{h}")
        if not 0.0 <= d["confidence"] <= 1.0:
            problems.append(f"det {i}: confidence {d['confidence']}")
    if first["detections"] != second["detections"]:
        problems.append("first and second call returned different detections")
    json.dumps(first)  # raises if anything is not JSON-serialisable

    print("\n--- checks ---")
    print(f"detection count: {len(first['detections'])}")
    print("PASS" if not problems else "FAIL")
    for p in problems:
        print(" -", p)

    detector.close()
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
