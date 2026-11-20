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
        "--model", type=Path, help="Checkpoint path (default: project configuration)"
    )
    parser.add_argument(
        "--device", help="Inference device, for example cpu or 0 (default: auto)"
    )
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
        default=None,
        help="Detection confidence threshold (default: project configuration)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Dummy-frame warm-up runs before video processing (default: 2)",
    )
    return parser.parse_args()


def draw_detections(
    frame: np.ndarray, detections: list[dict], color: tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw detection boxes and labels on a copy of the input frame."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))

        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        if detection.get("track_id") is not None:
            associated_items = ", ".join(detection.get("associated_items", []))
            label = f"Candidate #{detection['track_id']}"
            if associated_items:
                label += f" near {associated_items}"
            elif detection.get("candidate_reason"):
                label += f" · {detection['candidate_reason']}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
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

    if args.confidence is not None and not 0.0 <= args.confidence <= 1.0:
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

    report_path = output_path.with_suffix(".json")
    if output_path.suffix.lower() != ".mp4":
        print("ERROR: --output must use the .mp4 extension.", file=sys.stderr)
        return 2

    if any(
        candidate == input_path
        or (candidate.exists() and candidate.samefile(input_path))
        for candidate in (output_path, report_path)
    ):
        print(
            "ERROR: output video and report must differ from the input video.",
            file=sys.stderr,
        )
        return 2

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: Could not create output directory: {exc}", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        capture.release()
        print(
            f"ERROR: OpenCV could not open video: {input_path}",
            file=sys.stderr,
        )
        return 2

    try:
        fps_source = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_frames = (
            int(reported_frames)
            if np.isfinite(reported_frames) and reported_frames >= 0
            else 0
        )
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width < 1 or height < 1:
            raise ValueError("video reports invalid frame dimensions")
    except (ValueError, TypeError, OverflowError, cv2.error) as exc:
        capture.release()
        print(f"ERROR: Could not read video metadata: {exc}", file=sys.stderr)
        return 2

    output_fps = fps_source if np.isfinite(fps_source) and fps_source > 0 else 30.0

    writer = None
    detector = None
    model_info = None

    latencies_ms: list[float] = []
    processed = 0
    detections_total = 0
    inference_errors = 0
    error_samples = []
    dropped_detections = 0
    missing_timestamps = 0
    previous_timestamp = None
    stop_reason = "end_of_video"
    completion_warning = None

    end_to_end_start = time.perf_counter()
    startup_and_warmup_s = 0.0
    processing_wall_time_s = 0.0
    exit_code = 0

    try:
        # Model initialization and warm-up are measured separately.
        startup_start = time.perf_counter()

        detector = YOLOObjectDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
        )

        if args.warmup_runs:
            detector.warmup(
                runs=args.warmup_runs,
                size=(height, width),
            )

        startup_and_warmup_s = time.perf_counter() - startup_start

        # Use MP4V for broad local OpenCV compatibility.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not create output video: {output_path}")

        model_info = detector.get_model_info()

        print(f"Input:       {input_path}")
        print(f"Output:      {output_path}")
        print(f"Dimensions:  {width}x{height}")
        print(f"Source FPS:  {fps_source:.3f}")
        print(f"Source frames reported: {source_frames}")
        print(f"Device:      {model_info['device_resolved']}")
        print(f"Confidence:  {model_info['confidence_threshold']}")
        print(f"Warm-up runs: {args.warmup_runs}")
        print(f"Startup + warm-up: {startup_and_warmup_s:.3f} seconds")
        print("Processing video...")

        # Processing throughput excludes model loading and warm-up.
        # It includes frame reading, inference, annotation and video writing.
        processing_start = time.perf_counter()

        while args.max_frames is None or processed < args.max_frames:
            ok, frame = capture.read()

            if not ok:
                # Container frame counts can be approximate. Flag a large shortfall
                # for review instead of claiming that decoding certainly failed.
                tolerance = max(2, round(source_frames * 0.01))
                if source_frames > 0 and processed < source_frames - tolerance:
                    stop_reason = "early_decode_stop"
                    completion_warning = (
                        f"Decoded {processed} of approximately {source_frames} frames. "
                        "The video may be incomplete or its frame-count metadata may be wrong."
                    )
                break

            frame_id = processed

            # Media time can differ from frame_index / fps for variable-rate video.
            # A missing or stalled decoder timestamp is unknown, not an estimate.
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if (
                not np.isfinite(timestamp)
                or timestamp < 0
                or (previous_timestamp is not None and timestamp <= previous_timestamp)
            ):
                timestamp = None
                missing_timestamps += 1
            else:
                previous_timestamp = timestamp

            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Frame {frame_id} dimensions differ from the video's metadata; "
                    "OpenCV cannot write it at the configured output size."
                )

            result = detector.predict(
                frame,
                frame_id=frame_id,
                timestamp=timestamp,
            )

            latencies_ms.append(float(result["inference_time_ms"]))

            if result["model_status"] != "ok":
                inference_errors += 1
                if len(error_samples) < 10:
                    error_samples.append(
                        {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "error": result.get("error"),
                        }
                    )
            else:
                detections_total += len(result["detections"])
            dropped_detections += result.get("dropped_detections", 0)

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
        else:
            stop_reason = "max_frames"

        print()

        if processed == 0:
            raise RuntimeError("No frames were decoded from the input video.")

        # Finalize the container before publishing a successful report.
        writer.release()
        writer = None
        processing_wall_time_s = time.perf_counter() - processing_start
        end_to_end_wall_time_s = time.perf_counter() - end_to_end_start

        report = {
            "status": "completed_with_errors" if inference_errors else "completed",
            "input_video": str(input_path),
            "output_video": str(output_path),
            "model": model_info,
            "source": {
                "width": width,
                "height": height,
                "fps": fps_source
                if np.isfinite(fps_source) and fps_source > 0
                else None,
                "reported_frames": source_frames,
                "output_fps": output_fps,
                "output_fps_is_fallback": not (
                    np.isfinite(fps_source) and fps_source > 0
                ),
                "timestamp_source": "OpenCV CAP_PROP_POS_MSEC after decoding; unavailable timestamps remain null",
            },
            "results": {
                "processed_frames": processed,
                "detections_total": detections_total,
                "frames_with_inference_errors": inference_errors,
                "inference_error_samples": error_samples,
                "dropped_detections_total": dropped_detections,
                "frames_without_media_timestamp": missing_timestamps,
                "stop_reason": stop_reason,
                "completion_warning": completion_warning,
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
                    float(np.mean(latencies_ms)) if latencies_ms else None
                ),
                "inference_latency_ms_p50": percentile(latencies_ms, 50),
                "inference_latency_ms_p95": percentile(latencies_ms, 95),
            },
        }

        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        print("\n--- Video inference summary ---")
        print(json.dumps(report["results"], indent=2))
        print(f"Annotated video saved: {output_path}")
        print(f"Report saved: {report_path}")

        if completion_warning:
            print(f"WARNING: {completion_warning}", file=sys.stderr)

        if inference_errors:
            print(
                "WARNING: some frames had inference errors; inspect the JSON report.",
                file=sys.stderr,
            )
            exit_code = 1

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        # A rerun may already have replaced part of the video. Do not leave its
        # previous successful report beside a new, incomplete output.
        failure_report = {
            "status": "failed",
            "input_video": str(input_path),
            "output_video": str(output_path),
            "model": model_info,
            "error": error,
            "results": {
                "processed_frames": processed,
                "frames_with_inference_errors": inference_errors,
                "stop_reason": "processing_error",
            },
        }
        try:
            report_path.write_text(
                json.dumps(failure_report, indent=2, allow_nan=False), encoding="utf-8"
            )
        except OSError as report_error:
            print(
                f"ERROR: Could not save failure report: {report_error}", file=sys.stderr
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
