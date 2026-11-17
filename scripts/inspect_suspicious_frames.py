"""Inspect selected frames using the configured detector and save annotated images."""

import argparse
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_video_inference import draw_detections
from src.models.object_detector import YOLOObjectDetector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=Path, default=REPO_ROOT / "outputs" / "rifle.mp4"
    )
    parser.add_argument("--model", type=Path, help="Override the configured checkpoint")
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[40, 80, 120],
        help="1-based frame numbers to inspect",
    )
    parser.add_argument(
        "--confidence", type=float, help="Override the configured confidence threshold"
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--device", help="Inference device, for example cpu or 0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        print(f"ERROR: Input video does not exist: {args.video}", file=sys.stderr)
        return 2
    if any(number < 1 for number in args.frames):
        print("ERROR: Frame numbers must be at least 1.", file=sys.stderr)
        return 2

    target_frames = set(args.frames)
    found_frames = set()
    frame_number = 0
    errors = 0
    capture = cv2.VideoCapture(str(args.video))
    detector = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        detector = YOLOObjectDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        while frame_number < max(target_frames):
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number not in target_frames:
                continue
            found_frames.add(frame_number)
            result = detector.predict(frame, frame_id=frame_number - 1)
            print(f"Frame {frame_number}:")
            if result["model_status"] != "ok":
                print(f"ERROR: {result['error']}", file=sys.stderr)
                errors += 1
                continue
            if not result["detections"]:
                print("No detections")
            for detection in result["detections"]:
                print(detection)
            output = args.output_dir / f"{args.video.stem}_frame_{frame_number}.jpg"
            if not cv2.imwrite(
                str(output), draw_detections(frame, result["detections"])
            ):
                raise RuntimeError(f"Could not save {output}")
            print(f"Saved: {output}")
        missing = sorted(target_frames - found_frames)
        if missing:
            print(
                f"WARNING: Requested frames were unavailable: {missing}",
                file=sys.stderr,
            )
        return 1 if errors or missing else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.release()
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
