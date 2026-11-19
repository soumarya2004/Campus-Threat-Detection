"""Show the Person 1 detectors on a local webcam. Press Q or Escape to stop.

This is a standalone demonstration, not the team's camera or tracking service.
Person boxes and item boxes remain separate; no ownership or threat is inferred.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_video_inference import draw_detections
from src.camera_source import (
    DEFAULT_CAMERA_CONFIG,
    CameraSource,
    load_camera_config,
)
from src.evaluation import file_sha256, validate_report_paths
from src.model_config import DEFAULT_CONFIG, load_detector_config
from src.models.object_detector import YOLOObjectDetector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Local camera index")
    source.add_argument("--video", type=Path, help="Recorded video source")
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help="Shared capture settings file",
    )
    parser.add_argument(
        "--backend", choices=("auto", "any", "msmf", "dshow"),
        help="Override the capture backend from camera.yaml",
    )
    parser.add_argument("--width", type=int, help="Requested capture width")
    parser.add_argument("--height", type=int, help="Requested capture height")
    parser.add_argument("--fps", type=float, help="Requested camera frame rate")
    parser.add_argument(
        "--reconnect-attempts", type=int, help="Override camera reconnect attempts"
    )
    parser.add_argument(
        "--reconnect-delay", type=float, help="Seconds between reconnect attempts"
    )
    parser.add_argument("--device", help="Inference device, for example cpu or 0")
    parser.add_argument("--model", type=Path, help="Override the item checkpoint")
    parser.add_argument("--confidence", type=float, help="Override item confidence")
    parser.add_argument(
        "--with-people",
        action="store_true",
        help="Also run the separate person detector",
    )
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument(
        "--report",
        type=Path,
        help="Save session counts, timing and model hashes as JSON",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing session report"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Bounded runtime check without a window",
    )
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be a nonnegative local camera index")
    if args.width is not None and args.width < 1:
        parser.error("--width must be positive")
    if args.height is not None and args.height < 1:
        parser.error("--height must be positive")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.reconnect_attempts is not None and args.reconnect_attempts < 0:
        parser.error("--reconnect-attempts cannot be negative")
    if args.reconnect_delay is not None and args.reconnect_delay < 0:
        parser.error("--reconnect-delay cannot be negative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.no_display and args.max_frames is None:
        parser.error("--no-display requires --max-frames")
    if args.confidence is not None and not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.report is not None and args.report.suffix.lower() != ".json":
        parser.error("--report must use a .json extension")
    return args


def camera_config_from_args(args):
    """Apply CLI source overrides to the shared camera configuration."""
    config_path = getattr(args, "camera_config", DEFAULT_CAMERA_CONFIG)
    config = load_camera_config(config_path)
    overrides = {}
    if getattr(args, "camera", None) is not None:
        overrides["source"] = args.camera
    if getattr(args, "video", None) is not None:
        overrides["source"] = args.video.expanduser().resolve()
    for argument, setting in (
        ("backend", "backend"),
        ("width", "width"),
        ("height", "height"),
        ("fps", "fps"),
        ("reconnect_attempts", "reconnect_attempts"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            overrides[setting] = value
    delay = getattr(args, "reconnect_delay", None)
    if delay is not None:
        overrides["reconnect_delay_seconds"] = delay
    return replace(config, **overrides), Path(config_path).expanduser().resolve()


def main():
    args = parse_args()
    try:
        capture_config, camera_config_path = camera_config_from_args(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.report is not None:
        settings = load_detector_config()
        try:
            validate_report_paths(
                [args.report.expanduser().resolve()],
                [
                    DEFAULT_CONFIG,
                    camera_config_path,
                    Path(args.model or settings["model_path"]),
                    REPO_ROOT / "yolo11n.pt",
                ] + (
                    []
                    if capture_config.is_live_camera
                    else [Path(capture_config.source)]
                ),
                overwrite=args.overwrite,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    capture = None
    item_detector = None
    person_detector = None
    frames = 0
    inference_errors = 0
    exit_code = 0
    window_open = False
    window = "Campus detector | green: items | blue: people | Q: quit"
    wall_start = time.perf_counter()
    loop_start = None
    processing_seconds = 0.0
    setup_seconds = None
    latencies = []
    model_info = {}
    counts = {"items": 0, "people": 0}
    dimensions = None
    last_timestamp_source = None
    error = None
    stop_reason = "max_frames"
    backend = None

    def describe_model(detector):
        info = detector.get_model_info()
        return {**info, "checkpoint_sha256": file_sha256(Path(info["model_path"]))}

    try:
        capture = CameraSource(capture_config)
        capture.open()
        backend = capture.backend

        item_detector = YOLOObjectDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        item_detector.warmup()
        if args.report is not None:
            model_info["items"] = describe_model(item_detector)
        if args.with_people:
            from src.models.person_detector import YOLOPersonDetector

            person_detector = YOLOPersonDetector(device=args.device)
            person_detector.warmup()
            if args.report is not None:
                model_info["people"] = describe_model(person_detector)

        print(
            f"Running {capture_config.source_type.replace('_', ' ')} "
            f"{capture_config.source_label}. Press Q or Escape to stop."
        )
        setup_seconds = time.perf_counter() - wall_start
        loop_start = time.perf_counter()
        while args.max_frames is None or frames < args.max_frames:
            captured = capture.read()
            if captured is None:
                if capture_config.is_live_camera:
                    raise RuntimeError("Live camera ended without a recoverable frame")
                if frames == 0:
                    raise RuntimeError("Video file contains no decodable frames")
                stop_reason = "end_of_video"
                break
            frame = captured.frame
            metadata = captured.metadata
            last_timestamp_source = metadata["timestamp_source"]
            if frame is None:
                raise RuntimeError(
                    f"Capture returned an empty frame after {frames} frames"
                )
            frame_id = metadata["frame_id"]
            timestamp = metadata["timestamp_s"]
            dimensions = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
            prediction_start = time.perf_counter()
            items = item_detector.predict(frame, frame_id=frame_id, timestamp=timestamp)
            people = (
                person_detector.predict(frame, frame_id=frame_id, timestamp=timestamp)
                if person_detector is not None
                else None
            )
            latencies.append((time.perf_counter() - prediction_start) * 1000)
            for name, result in (("items", items), ("people", people)):
                if result is not None and result["model_status"] == "ok":
                    counts[name] += len(result["detections"])
            failures = [
                f"{name}: {result.get('error')}"
                for name, result in (("items", items), ("people", people))
                if result is not None and result["model_status"] != "ok"
            ]
            if failures:
                inference_errors += 1
                if inference_errors <= 3:
                    print(f"Frame {frame_id}: {'; '.join(failures)}", file=sys.stderr)
            frames += 1

            if not args.no_display:
                annotated = frame
                if people is not None and people["model_status"] == "ok":
                    annotated = draw_detections(
                        annotated, people["detections"], color=(255, 160, 40)
                    )
                annotated = draw_detections(
                    annotated,
                    items["detections"] if items["model_status"] == "ok" else [],
                )
                label = (
                    "INFERENCE ERROR"
                    if failures
                    else "Object detections - no threat decision"
                )
                cv2.putText(
                    annotated,
                    label,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255) if failures else (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, annotated)
                window_open = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    stop_reason = "user_quit"
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "window_closed"
                    break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\nCamera demo interrupted.")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_reason = "error"
        print(f"ERROR: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        if loop_start is not None:
            processing_seconds = time.perf_counter() - loop_start
        if capture is not None:
            capture.close()
        if person_detector is not None:
            person_detector.close()
        if item_detector is not None:
            item_detector.close()
        if window_open:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass  # The user may already have closed this window.

    print(f"Processed {frames} frames; {inference_errors} frames had inference errors.")
    if args.report is not None:
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Detector runtime on shared live-camera or recorded-video input; not an accuracy or threat-assessment test.",
            "passed": exit_code == 0 and inference_errors == 0 and frames > 0,
            "camera": {
                "index": capture_config.source
                if capture_config.is_live_camera
                else None,
                "source_type": capture_config.source_type,
                "source_label": capture_config.source_label,
                "backend": backend,
                "last_frame_dimensions": dimensions,
                "frames_captured": capture.frame_id if capture is not None else 0,
                "reconnect_count": capture.reconnect_count
                if capture is not None
                else 0,
                "read_failures": capture.failed_reads if capture is not None else 0,
                "capture_properties": capture.capture_properties
                if capture is not None
                else {},
            },
            "with_people": args.with_people,
            "display_enabled": not args.no_display,
            "models": model_info,
            "frames_processed": frames,
            "frames_with_inference_errors": inference_errors,
            "detections_total": counts,
            "stop_reason": stop_reason,
            "error": error,
            "timing": {
                "setup_and_warmup_seconds": setup_seconds,
                "capture_inference_display_seconds": processing_seconds,
                "capture_inference_display_fps": frames / processing_seconds
                if processing_seconds > 0
                else None,
                "combined_predict_p50_ms": float(np.median(latencies))
                if latencies
                else None,
                "combined_predict_p95_ms": float(np.percentile(latencies, 95))
                if latencies
                else None,
                "scope": "Combined predict calls exclude capture and display; loop throughput includes them. Capture is read immediately before inference. Setup and warm-up are excluded from loop throughput.",
            },
            "timestamp_source": last_timestamp_source,
            "timestamp_policy": "Recorded video uses advancing media timestamps when available; live cameras use host monotonic elapsed time after capture, not camera hardware time.",
            "media_saved": False,
        }
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(f"Session report: {args.report}")
        except OSError as exc:
            print(f"ERROR: Could not save session report: {exc}", file=sys.stderr)
            exit_code = 1
    return 1 if inference_errors else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
