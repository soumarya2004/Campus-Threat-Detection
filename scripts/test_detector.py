"""Run the detector on one image and save its annotated prediction."""

import argparse
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_video_inference import draw_detections
from src.models.object_detector import YOLOObjectDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image")
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "outputs" / "detector_test.jpg"
    )
    parser.add_argument("--model", type=Path, help="Override the configured checkpoint")
    parser.add_argument(
        "--confidence", type=float, help="Override the configured confidence threshold"
    )
    parser.add_argument("--device", help="Inference device, for example cpu or 0")
    args = parser.parse_args()
    args.image = args.image.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.image == args.output or (
        args.image.exists()
        and args.output.exists()
        and args.image.samefile(args.output)
    ):
        parser.error("Output path must differ from the input image.")
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f"Could not read image: {args.image}")

    detector = None
    try:
        detector = YOLOObjectDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        result = detector.predict(frame)
        if result["model_status"] != "ok":
            raise RuntimeError(result["error"])
        print(f"Image: {args.image}")
        print(f"Device: {detector.get_model_info()['device_resolved']}")
        print(f"Detections: {len(result['detections'])}")
        for detection in result["detections"]:
            print(detection)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(frame, result["detections"])
        if not cv2.imwrite(str(args.output), annotated):
            raise RuntimeError(f"Could not save output: {args.output}")
        print(f"Annotated image saved: {args.output}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
