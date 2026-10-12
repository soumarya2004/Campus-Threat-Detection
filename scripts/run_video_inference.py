
"""Run weapon detection on a video using the existing YOLOObjectDetector.

Examples:
    python scripts/run_video_inference.py input.mp4
    python scripts/run_video_inference.py input.mp4 --output outputs/annotated.mp4
    python scripts/run_video_inference.py input.mp4 --max-frames 300
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.object_detector import YOLOObjectDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO weapon detection on a video."
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output video (default: outputs/<input>_annotated.mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process; useful for short tests",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Dummy-frame warm-up runs before video processing (default: 2)",
    )
    return parser.parse_args()


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw detection boxes and labels on a copy of the input frame."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(
            annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def percentile(values: list[float], q: float) -> float | None:
    """Calculate a latency percentile, or None when no values exist."""
    if not values:
        return None

    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    args = parse_args()

    # Validate arguments.
    if args.max_frames is not None and args.max_frames < 1:
        print("ERROR: --max-frames must be at least 1.", file=sys.stderr)
        return 2

    if not 0.0 <= args.confidence <= 1.0:
        print(
            "ERROR: --confidence must be between 0 and 1.",
            file=sys.stderr,
        )
        return 2

    if args.warmup_runs < 0:
        print(
            "ERROR: --warmup-runs cannot be negative.",
            file=sys.stderr,
        )
        return 2

    input_path = args.video.expanduser().resolve()

    if not input_path.is_file():
        print(
            f"ERROR: input video does not exist: {input_path}",
            file=sys.stderr,
        )
        return 2

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else REPO_ROOT / "outputs" / f"{input_path.stem}_annotated.mp4"
    )

    if input_path == output_path:
        print(
            "ERROR: output path must differ from input video.",
            file=sys.stderr,
        )
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        capture.release()
        print(
            f"ERROR: OpenCV could not open video: {input_path}",
            file=sys.stderr,
        )
        return 2

    fps_source = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width < 1 or height < 1:
        capture.release()
        print(
            "ERROR: video reports invalid frame dimensions.",
            file=sys.stderr,
        )
        return 2

    output_fps = (
        fps_source
        if np.isfinite(fps_source) and fps_source > 0
        else 30.0
    )

    writer = None
    detector = None

    latencies_ms: list[float] = []
    processed = 0
    detections_total = 0
    inference_errors = 0

    end_to_end_start = time.perf_counter()
    startup_and_warmup_s = 0.0
    processing_wall_time_s = 0.0
    exit_code = 0

    try:
        # Model initialization and warm-up are measured separately.
        startup_start = time.perf_counter()

        detector = YOLOObjectDetector(
            confidence_threshold=args.confidence
        )

        if args.warmup_runs:
            detector.warmup(
                runs=args.warmup_runs,
                size=(height, width),
            )

        startup_and_warmup_s = (
            time.perf_counter() - startup_start
        )

        # Use MP4V for broad local OpenCV compatibility.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"OpenCV could not create output video: {output_path}"
            )

        model_info = detector.get_model_info()

        print(f"Input:       {input_path}")
        print(f"Output:      {output_path}")
        print(f"Dimensions:  {width}x{height}")
        print(f"Source FPS:  {fps_source:.3f}")
        print(f"Source frames reported: {source_frames}")
        print(f"Device:      {model_info['device_resolved']}")
        print(f"Confidence:  {args.confidence}")
        print(f"Warm-up runs: {args.warmup_runs}")
        print(
            f"Startup + warm-up: {startup_and_warmup_s:.3f} seconds"
        )
        print("Processing video...")

        # Processing throughput excludes model loading and warm-up.
        # It includes frame reading, inference, annotation and video writing.
        processing_start = time.perf_counter()

        while args.max_frames is None or processed < args.max_frames:
            ok, frame = capture.read()

            if not ok:
                break

            frame_id = processed

            # Estimate timestamp from the video's frame rate.
            timestamp = frame_id / output_fps

            result = detector.predict(
                frame,
                frame_id=frame_id,
                timestamp=timestamp,
            )

            latencies_ms.append(
                float(result["inference_time_ms"])
            )

            if result["model_status"] != "ok":
                inference_errors += 1
            else:
                detections_total += len(result["detections"])

            annotated = draw_detections(
                frame,
                result["detections"],
            )

            if result["model_status"] != "ok":
                cv2.putText(
                    annotated,
                    "INFERENCE ERROR",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(annotated)
            processed += 1

            if processed % 30 == 0:
                print(
                    f"\rFrames: {processed} | "
                    f"latest latency: {latencies_ms[-1]:.1f} ms",
                    end="",
                    flush=True,
                )

        processing_wall_time_s = (
            time.perf_counter() - processing_start
        )
        end_to_end_wall_time_s = (
            time.perf_counter() - end_to_end_start
        )

        print()

        if processed == 0:
            raise RuntimeError(
                "No frames were decoded from the input video."
            )

        report = {
            "input_video": str(input_path),
            "output_video": str(output_path),
            "model": model_info,
            "source": {
                "width": width,
                "height": height,
                "fps": fps_source,
                "reported_frames": source_frames,
            },
            "results": {
                "processed_frames": processed,
                "detections_total": detections_total,
                "frames_with_inference_errors": inference_errors,

                # Timing metrics.
                "startup_and_warmup_seconds": startup_and_warmup_s,
                "processing_wall_time_seconds": processing_wall_time_s,
                "processing_fps": (
                    processed / processing_wall_time_s
                    if processing_wall_time_s > 0
                    else None
                ),
                "end_to_end_wall_time_seconds": end_to_end_wall_time_s,

                # Detector-reported inference latency per frame.
                "inference_latency_ms_mean": (
                    float(np.mean(latencies_ms))
                    if latencies_ms
                    else None
                ),
                "inference_latency_ms_p50": percentile(
                    latencies_ms, 50
                ),
                "inference_latency_ms_p95": percentile(
                    latencies_ms, 95
                ),
            },
        }

        report_path = output_path.with_suffix(".json")
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\n--- Video inference summary ---")
        print(json.dumps(report["results"], indent=2))
        print(f"Annotated video saved: {output_path}")
        print(f"Report saved: {report_path}")

        if inference_errors:
            print(
                "WARNING: some frames had inference errors; "
                "inspect the JSON report.",
                file=sys.stderr,
            )
            exit_code = 1

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        exit_code = 1

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        if detector is not None:
            detector.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
