"""Print detection counts for a clip without writing an annotated video."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.object_detector import YOLOObjectDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("--model", type=Path, help="Override the configured checkpoint")
    parser.add_argument(
        "--confidence", type=float, help="Override the configured confidence threshold"
    )
    parser.add_argument("--device", help="Inference device, for example cpu or 0")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")

    capture = cv2.VideoCapture(str(args.video))
    detector = None
    counts = Counter()
    frames_by_class = Counter()
    total_frames = 0
    frames_with_detections = 0
    errors = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        detector = YOLOObjectDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = detector.predict(frame, frame_id=total_frames)
            total_frames += 1
            if result["model_status"] != "ok":
                errors += 1
                if errors <= 5:
                    print(f"Frame {total_frames}: {result['error']}", file=sys.stderr)
                continue
            classes = [item["class_name"] for item in result["detections"]]
            counts.update(classes)
            frames_by_class.update(set(classes))
            frames_with_detections += bool(classes)

        if total_frames == 0:
            raise RuntimeError("No frames were decoded.")
        print(f"Frames processed: {total_frames}")
        print(f"Frames with detections: {frames_with_detections}")
        print(f"Frames with inference errors: {errors}")
        for name, count in counts.most_common():
            print(f"{name}: {count} detections across {frames_by_class[name]} frames")
        print("Detection counts are predictions, not verified true positives.")
        return 1 if errors else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.release()
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
