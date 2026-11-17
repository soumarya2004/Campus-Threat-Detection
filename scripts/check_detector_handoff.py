"""Rehearse a downstream consumer using a few frames from a local recording.

This checks runtime and the public interface, not detection accuracy or another
team member's complete pipeline. No camera, network, or annotated video is used.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.object_detector import YOLOObjectDetector


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def consume_payload(payload, frame, frame_id, timestamp, class_names):
    """Check the information a consumer needs; return the number of valid boxes."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("Expected an OpenCV BGR uint8 frame with three channels.")
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("The decoded frame is empty.")
    if not isinstance(payload, dict):
        raise ValueError("Detector output must be a dictionary.")
    required = {
        "frame_id",
        "timestamp",
        "detections",
        "inference_time_ms",
        "model_status",
        "error",
        "dropped_detections",
    }
    if not required.issubset(payload):
        raise ValueError(f"Missing output fields: {sorted(required - payload.keys())}")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Output is not valid JSON: {exc}") from exc
    if type(payload["frame_id"]) is not int or payload["frame_id"] != frame_id:
        raise ValueError("The supplied frame ID was not preserved.")
    if not finite_number(payload["timestamp"]) or payload["timestamp"] != timestamp:
        raise ValueError("The supplied media timestamp was not preserved.")
    if payload["model_status"] != "ok":
        raise ValueError(
            f"Detector status {payload['model_status']}: {payload['error']}"
        )
    if payload["error"] is not None:
        raise ValueError("An ok result contains an error message.")
    if (
        not finite_number(payload["inference_time_ms"])
        or payload["inference_time_ms"] < 0
    ):
        raise ValueError("Inference time must be finite and nonnegative.")
    if (
        type(payload["dropped_detections"]) is not int
        or payload["dropped_detections"] < 0
    ):
        raise ValueError("Dropped-detection count must be a nonnegative integer.")
    if not isinstance(payload["detections"], list):
        raise ValueError("Detections must be a list.")
    for detection in payload["detections"]:
        if not isinstance(detection, dict) or not {
            "bbox",
            "class_id",
            "class_name",
            "confidence",
        }.issubset(detection):
            raise ValueError("A detection is missing required fields.")
        class_id = detection["class_id"]
        if (
            type(class_id) is not int
            or class_id not in class_names
            or class_names[class_id] != detection["class_name"]
        ):
            raise ValueError(
                "Detection class ID/name does not match the model's public class map."
            )
        confidence = detection["confidence"]
        if not finite_number(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Detection confidence must be finite and in [0, 1].")
        bbox = detection["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(finite_number(value) for value in bbox)
        ):
            raise ValueError("Bounding box must contain four finite pixel coordinates.")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Bounding box lies outside the original frame dimensions.")
    return len(payload["detections"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Local recording supplied by the video-pipeline teammate",
    )
    parser.add_argument(
        "--max-frames", type=int, default=5, help="Maximum frames to check (default: 5)"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "outputs" / "handoff_check.json"
    )
    parser.add_argument(
        "--device",
        help="Inference device, for example cpu or 0 (default: selected automatically)",
    )
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    video_path, output_path = (
        args.video.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    if video_path == output_path or (
        video_path.exists()
        and output_path.exists()
        and video_path.samefile(output_path)
    ):
        parser.error("Output must differ from the input video.")
    report = {
        "purpose": "Local downstream-consumer rehearsal: runtime/schema only, not accuracy or teammate end-to-end validation.",
        "input_video": str(video_path),
        "input_format": "OpenCV decoded BGR uint8; boxes in original-frame pixels",
        "timestamp_source": "OpenCV CAP_PROP_POS_MSEC after decoding, converted to seconds",
        "model": None,
        "checks": {
            "frames_checked": 0,
            "passed_frames": 0,
            "failed_frames": 0,
            "detections_checked": 0,
        },
        "sample_payloads": [],
        "sample_policy": "First five returned payloads at most; invalid JSON payloads are omitted and reported as failures.",
        "errors": [],
    }
    capture = None
    detector = None
    try:
        if not video_path.is_file():
            raise FileNotFoundError(f"Local video not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        detector = YOLOObjectDetector(device=args.device)
        report["model"] = detector.get_model_info()
        class_names = report["model"]["class_names"]
        previous_timestamp = None
        for frame_id in range(args.max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(
                    "Recording has no valid media timestamp for this frame."
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    "Recording timestamps did not advance; a meaningful timestamp handoff cannot be checked."
                )
            previous_timestamp = timestamp
            report["checks"]["frames_checked"] += 1
            try:
                payload = detector.predict(
                    frame, frame_id=frame_id, timestamp=timestamp
                )
                if len(report["sample_payloads"]) < 5:
                    try:
                        report["sample_payloads"].append(
                            json.loads(json.dumps(payload, allow_nan=False))
                        )
                    except (ValueError, TypeError):
                        pass  # consume_payload reports this as a failed check below.
                count = consume_payload(
                    payload, frame, frame_id, timestamp, class_names
                )
                report["checks"]["passed_frames"] += 1
                report["checks"]["detections_checked"] += count
            except (ValueError, TypeError, RuntimeError) as exc:
                report["checks"]["failed_frames"] += 1
                if len(report["errors"]) < 10:
                    report["errors"].append({"frame_id": frame_id, "error": str(exc)})
        if report["checks"]["frames_checked"] == 0:
            raise RuntimeError("No frames could be decoded.")
    except Exception as exc:
        report["errors"].append({"frame_id": None, "error": str(exc)})
    finally:
        if capture is not None:
            capture.release()
        if detector is not None:
            detector.close()
    report["passed"] = not report["errors"] and report["checks"]["failed_frames"] == 0
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: Could not save handoff report: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: runtime/schema rehearsal"
        if report["passed"]
        else "FAIL: inspect errors in the handoff report"
    )
    print(json.dumps(report["checks"], indent=2))
    print(f"Saved: {output_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
