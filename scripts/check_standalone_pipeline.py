"""Rehearse both Person 1 detector outputs using a local recording.

The consumer validates JSON, original-frame boxes and separate class namespaces.
This checks the standalone detection interface, not tracking or threat accuracy.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_detector_handoff import consume_payload
from src.evaluation import file_sha256, validate_report_paths
from src.model_config import DEFAULT_CONFIG, load_detector_config
from src.models.object_detector import EXPECTED_CLASSES, YOLOObjectDetector
from src.models.person_detector import YOLOPersonDetector

STREAM_CLASSES = {"items": EXPECTED_CLASSES, "people": {0: "person"}}


def consume_frame(packet, frame, frame_id, timestamp):
    """A small standalone consumer: class IDs only have meaning inside a stream."""
    if not isinstance(packet, dict) or set(packet) != set(STREAM_CLASSES):
        raise ValueError(
            "A frame packet must contain separate items and people streams"
        )
    return {
        stream: consume_payload(packet[stream], frame, frame_id, timestamp, names)
        for stream, names in STREAM_CLASSES.items()
    }


def check_output_paths(outputs, inputs, *, overwrite=False):
    validate_report_paths(outputs, inputs, overwrite=overwrite)


def rehearse(video_path, *, max_frames, device=None, payload_stream=None):
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Standalone Person 1 runtime and consumer-contract rehearsal; no P3/P4 code required. Not an accuracy evaluation.",
        "source": {"path": str(video_path), "sha256": file_sha256(video_path)},
        "timestamp_source": "OpenCV CAP_PROP_POS_MSEC after decoding, in seconds; missing or stalled timestamps fail this strict rehearsal.",
        "class_namespaces": STREAM_CLASSES,
        "models": {},
        "checks": {
            "frames_checked": 0,
            "passed_frames": 0,
            "failed_frames": 0,
            "item_boxes": 0,
            "person_boxes": 0,
            "dropped_detections": {"items": 0, "people": 0},
        },
        "errors": [],
        "sample_packets": [],
        "stop_reason": None,
    }
    capture = None
    detectors = {}
    latencies = []
    setup_start = time.perf_counter()
    loop_start = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open recording: {video_path}")
        for stream, factory in (
            ("items", YOLOObjectDetector),
            ("people", YOLOPersonDetector),
        ):
            detector = factory(device=device)
            detectors[stream] = detector
            info = detector.get_model_info()
            if info["class_names"] != STREAM_CLASSES[stream]:
                raise ValueError(f"Wrong class mapping in {stream} stream")
            report["models"][stream] = {
                **info,
                "checkpoint_sha256": file_sha256(Path(info["model_path"])),
            }
            detector.warmup(runs=2)
        report["startup_and_warmup_seconds"] = time.perf_counter() - setup_start
        loop_start = time.perf_counter()
        previous_timestamp = None
        for frame_id in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                report["stop_reason"] = "end_of_video"
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if (
                not math.isfinite(timestamp)
                or timestamp < 0
                or (previous_timestamp is not None and timestamp <= previous_timestamp)
            ):
                raise ValueError(
                    f"Frame {frame_id} has no valid advancing media timestamp"
                )
            previous_timestamp = timestamp
            report["checks"]["frames_checked"] += 1
            try:
                started = time.perf_counter()
                packet = {
                    stream: detector.predict(
                        frame, frame_id=frame_id, timestamp=timestamp
                    )
                    for stream, detector in detectors.items()
                }
                latencies.append((time.perf_counter() - started) * 1000)
                # Validate what a JSON consumer actually receives, not Python-only objects.
                line = json.dumps(packet, allow_nan=False)
                packet = json.loads(line)
                counts = consume_frame(packet, frame, frame_id, timestamp)
                report["checks"]["passed_frames"] += 1
                report["checks"]["item_boxes"] += counts["items"]
                report["checks"]["person_boxes"] += counts["people"]
                for stream in STREAM_CLASSES:
                    report["checks"]["dropped_detections"][stream] += packet[stream][
                        "dropped_detections"
                    ]
                if payload_stream is not None:
                    payload_stream.write(line + "\n")
                if len(report["sample_packets"]) < 3:
                    report["sample_packets"].append(packet)
            except (ValueError, TypeError, RuntimeError) as exc:
                report["checks"]["failed_frames"] += 1
                if len(report["errors"]) < 10:
                    report["errors"].append({"frame_id": frame_id, "error": str(exc)})
        else:
            report["stop_reason"] = "max_frames"
        if report["checks"]["frames_checked"] == 0:
            raise RuntimeError("No frames decoded from the recording")
    except Exception as exc:
        report["errors"].append({"frame_id": None, "error": str(exc)})
        report["stop_reason"] = "error"
    finally:
        loop_seconds = (
            time.perf_counter() - loop_start if loop_start is not None else None
        )
        if capture is not None:
            capture.release()
        for detector in detectors.values():
            detector.close()
    report["timing"] = {
        "processing_seconds": loop_seconds,
        "processing_fps": report["checks"]["frames_checked"] / loop_seconds
        if loop_seconds
        else None,
        "combined_prediction_mean_ms": float(np.mean(latencies)) if latencies else None,
        "combined_prediction_p50_ms": float(np.median(latencies))
        if latencies
        else None,
        "combined_prediction_p95_ms": float(np.percentile(latencies, 95))
        if latencies
        else None,
        "scope": "Sequential item and person predictions. Processing includes decoding, JSON serialization and consumer checks; excludes setup, drawing, display and final report writing.",
    }
    report["passed"] = not report["errors"] and report["checks"]["frames_checked"] > 0
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument(
        "--device", help="cpu or CUDA device index; defaults to automatic selection"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "outputs/standalone_handoff.json"
    )
    parser.add_argument(
        "--payloads",
        type=Path,
        help="Optional JSONL file of all validated frame packets",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    video = args.video.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payloads = args.payloads.expanduser().resolve() if args.payloads else None
    if not video.is_file():
        parser.error(f"Recording not found: {video}")
    try:
        settings = load_detector_config()
        check_output_paths(
            [output] + ([payloads] if payloads else []),
            [video, settings["model_path"], REPO_ROOT / "yolo11n.pt", DEFAULT_CONFIG],
            overwrite=args.overwrite,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if payloads:
            payloads.parent.mkdir(parents=True, exist_ok=True)
            with payloads.open("w", encoding="utf-8", newline="\n") as stream:
                report = rehearse(
                    video,
                    max_frames=args.max_frames,
                    device=args.device,
                    payload_stream=stream,
                )
            report["payloads"] = {
                "path": str(payloads),
                "sha256": file_sha256(payloads),
            }
        else:
            report = rehearse(video, max_frames=args.max_frames, device=args.device)
        output.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{'PASS' if report['passed'] else 'FAIL'}: standalone two-stream rehearsal")
    print(json.dumps(report["checks"], indent=2))
    print(f"Saved: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
