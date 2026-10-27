import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

DEFAULT_MODEL = "runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="outputs/rifle.mp4", help="Input video path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO checkpoint path")
    parser.add_argument(
        "--frames", nargs="+", type=int, default=[40, 80, 120],
        help="1-based frame numbers to inspect (default: 40 80 120)",
    )
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output-dir", default="outputs", help="Directory for annotated frame images")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    if not video_path.is_file():
        raise SystemExit(f"Input video does not exist: {video_path}")
    if not model_path.is_file():
        raise SystemExit(f"Model checkpoint does not exist: {model_path}")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if any(n < 1 for n in args.frames):
        raise SystemExit("Frame numbers must be 1 or greater")

    target_frames = sorted(set(args.frames))
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open input video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_number = 0
    found_frames = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            if frame_number not in target_frames:
                continue

            found_frames.add(frame_number)
            result = model.predict(
                frame, conf=args.confidence, device=args.device, verbose=False
            )[0]
            print(f"\nFrame {frame_number}:")
            if len(result.boxes) == 0:
                print("No detections")
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                coords = [round(float(x), 1) for x in box.xyxy[0].tolist()]
                print(
                    f"{model.names[class_id]} | confidence={confidence:.3f} | box={coords}"
                )

            output_path = output_dir / f"inspect_frame_{frame_number}.jpg"
            if not cv2.imwrite(str(output_path), result.plot()):
                print(f"Warning: could not save {output_path}")
            else:
                print(f"Saved: {output_path}")
    finally:
        cap.release()

    missing = sorted(set(target_frames) - found_frames)
    if missing:
        print(f"\nWarning: requested frame(s) exceed video length or were unavailable: {missing}")
    if not found_frames:
        print("None of the requested frames were available in the video.")


if __name__ == "__main__":
    main()
