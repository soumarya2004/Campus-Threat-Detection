"""Run local campus camera detection, tracking, temporal analysis, and review alerts."""

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_video_inference import draw_detections  # noqa: E402
from src.camera_source import (  # noqa: E402
    DEFAULT_CAMERA_CONFIG,
    CameraSource,
    load_camera_config,
)
from src.evaluation import file_sha256, validate_report_paths  # noqa: E402
from src.model_config import DEFAULT_CONFIG, load_detector_config  # noqa: E402
from src.models.object_detector import YOLOObjectDetector  # noqa: E402
from src.tracking_analysis.item_association import (  # noqa: E402
    link_people_to_detected_items,
)
from src.tracking_analysis.analytics import (  # noqa: E402
    MovementAnalyzer,
    analyze_interactions,
)
from src.monitoring.threat_assessment import ThreatAssessmentEngine  # noqa: E402


class BehaviorInferenceWorker:
    """Run the slower temporal model off the camera's frame-processing loop."""

    def __init__(self, classifier):
        self.classifier = classifier
        self._condition = threading.Condition()
        self._requested = False
        self._closed = False
        self._result = None
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(self):
        with self._condition:
            self._requested = True
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._requested or self._closed)
                if self._closed and not self._requested:
                    return
                self._requested = False
            try:
                result = self.classifier.infer()
            except Exception as exc:
                with self._condition:
                    self._error = f"{type(exc).__name__}: {exc}"
            else:
                with self._condition:
                    self._result = result
                    self._error = None

    def snapshot(self):
        with self._condition:
            return self._result, self._error

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=5)


def attach_track_ids(detections, tracked_people):
    """Associate current person boxes to distinct ByteTrack IDs by IoU."""
    output = [dict(detection) for detection in detections]
    matches = []
    for detection_index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox"]
        for track_id, box in tracked_people:
            tx1, ty1, tx2, ty2 = box
            intersection = max(0.0, min(x2, tx2) - max(x1, tx1)) * max(
                0.0, min(y2, ty2) - max(y1, ty1)
            )
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            track_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
            union = area + track_area - intersection
            iou = intersection / union if union else 0.0
            if iou >= 0.3:
                matches.append((iou, detection_index, int(track_id)))

    assigned_detections = set()
    assigned_tracks = set()
    for _, detection_index, track_id in sorted(matches, reverse=True):
        if detection_index in assigned_detections or track_id in assigned_tracks:
            continue
        output[detection_index]["track_id"] = track_id
        assigned_detections.add(detection_index)
        assigned_tracks.add(track_id)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Local camera index")
    source.add_argument("--video", type=Path, help="Recorded video source")
    source.add_argument(
        "--stream-url",
        help="HTTP(S) live video stream URL, such as a DroidCam MJPEG endpoint",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help="Shared capture settings file",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "any", "msmf", "dshow"),
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
    parser.add_argument(
        "--image-size",
        type=int,
        default=416,
        help="YOLO inference image size; 416 is tuned for real-time live use",
    )
    parser.add_argument("--model", type=Path, help="Override the item checkpoint")
    parser.add_argument("--confidence", type=float, help="Override item confidence")
    parser.add_argument(
        "--with-people",
        action="store_true",
        help="Also run the separate person detector",
    )
    parser.add_argument(
        "--person-interval",
        type=int,
        default=2,
        help="Run person detection every N frames; reuse the latest boxes between runs",
    )
    parser.add_argument(
        "--with-tracking",
        action="store_true",
        help="Assign ByteTrack IDs to person detections",
    )
    parser.add_argument(
        "--with-behavior",
        action="store_true",
        help="Load the trained temporal behavior classifier",
    )
    parser.add_argument(
        "--behavior-interval",
        type=int,
        default=12,
        help="Run behavior inference every N frames in a background worker",
    )
    parser.add_argument(
        "--dashboard-image-interval",
        type=int,
        default=2,
        help="Write a new dashboard JPEG every N processed frames",
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
        "--dashboard-session",
        type=Path,
        help="Write a local monitoring feed into a new session folder",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening a preview window",
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
    if (
        args.no_display
        and args.max_frames is None
        and args.dashboard_session is None
    ):
        parser.error("--no-display requires --max-frames or --dashboard-session")
    if args.confidence is not None and not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    if args.person_interval < 1:
        parser.error("--person-interval must be at least 1")
    if args.behavior_interval < 1:
        parser.error("--behavior-interval must be at least 1")
    if args.dashboard_image_interval < 1:
        parser.error("--dashboard-image-interval must be at least 1")
    if args.with_tracking:
        args.with_people = True
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
    if getattr(args, "stream_url", None) is not None:
        overrides["source"] = args.stream_url
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
                ]
                + (
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
    cached_people = None
    cached_people_frame_id = None
    person_tracker = None
    behavior_classifier = None
    behavior_worker = None
    behavior_result = None
    behavior_error = None
    frames = 0
    inference_errors = 0
    exit_code = 0
    window_open = False
    window = "Campus Threat Detection | review candidates | Q: quit"
    wall_start = time.perf_counter()
    loop_start = None
    processing_seconds = 0.0
    setup_seconds = None
    latencies = []
    fps_samples = deque(maxlen=151)
    model_info = {}
    counts = {"items": 0, "people": 0}
    dimensions = None
    last_timestamp_source = None
    error = None
    stop_reason = "max_frames"
    backend = None
    dashboard_writer = None
    movement_analyzer = None
    previous_motion_rows = []
    assessment_engine = ThreatAssessmentEngine()
    analysis_event_id = 1
    last_capture_reconnect_count = 0

    def describe_model(detector):
        info = detector.get_model_info()
        return {**info, "checkpoint_sha256": file_sha256(Path(info["model_path"]))}

    try:
        if args.dashboard_session is not None:
            from src.monitoring.session import DashboardSessionWriter

            dashboard_writer = DashboardSessionWriter(
                args.dashboard_session,
                metadata={
                    "purpose": "camera-demo monitoring feed",
                    "source_type": capture_config.source_type,
                    "source_label": capture_config.source_label,
                },
            )
        capture = CameraSource(capture_config)
        capture.open()
        backend = capture.backend
        last_capture_reconnect_count = capture.reconnect_count

        item_detector = YOLOObjectDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
            image_size=args.image_size,
        )
        item_detector.warmup()
        if args.report is not None:
            model_info["items"] = describe_model(item_detector)
        if args.with_people:
            from src.models.person_detector import YOLOPersonDetector

            person_detector = YOLOPersonDetector(
                device=args.device, image_size=args.image_size
            )
            person_detector.warmup()
            if args.report is not None:
                model_info["people"] = describe_model(person_detector)
        if args.with_tracking:
            from src.tracking_analysis.person_tracker import PersonTracker

            person_tracker = PersonTracker(weights=None, device=args.device or "cpu")
            person_tracker.reset()
            movement_analyzer = MovementAnalyzer()
        if args.with_behavior:
            from src.behavior.models.behavior_classifier import BehaviorClassifier

            behavior_classifier = BehaviorClassifier(
                checkpoint_path=str(REPO_ROOT / "models/behavior/best_model.pt"),
                device=args.device,
            )
            if behavior_classifier.device.type == "cuda":
                import torch

                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
            behavior_worker = BehaviorInferenceWorker(behavior_classifier)
            print(
                f"Behavior model ready on {behavior_classifier.device}; "
                f"{behavior_classifier.required_buffer_size} frames needed before its first score."
            )

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
            reconnect_count = int(metadata.get("reconnect_count", 0))
            if reconnect_count > last_capture_reconnect_count:
                last_capture_reconnect_count = reconnect_count
                analysis_event_id += 1
                cached_people = None
                cached_people_frame_id = None
                previous_motion_rows = []
                if person_tracker is not None:
                    person_tracker.reset()
                if movement_analyzer is not None:
                    movement_analyzer = MovementAnalyzer()
                assessment_engine.clear_temporal_signals()
                print(
                    f"Camera stream reconnected ({reconnect_count}); "
                    "person tracking history was restarted."
                )
            dimensions = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
            prediction_start = time.perf_counter()
            items = item_detector.predict(frame, frame_id=frame_id, timestamp=timestamp)
            people_fresh = False
            if person_detector is not None and (
                cached_people is None or frames % args.person_interval == 0
            ):
                people = person_detector.predict(
                    frame, frame_id=frame_id, timestamp=timestamp
                )
                people_fresh = True
                if people["model_status"] == "ok":
                    cached_people = people["detections"]
                    cached_people_frame_id = frame_id
            elif person_detector is not None:
                people = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "detections": cached_people or [],
                    "inference_time_ms": 0.0,
                    "model_status": "ok",
                    "error": None,
                    "dropped_detections": 0,
                    "cached": True,
                }
            else:
                people = None
            if behavior_classifier is not None:
                behavior_classifier.ingest_frame(frame, timestamp=timestamp)
                if (
                    frames + 1 >= behavior_classifier.required_buffer_size
                    and (frames + 1) % args.behavior_interval == 0
                ):
                    behavior_worker.request()
                behavior_result, behavior_error = behavior_worker.snapshot()

            person_detections = (
                people["detections"]
                if people is not None and people["model_status"] == "ok"
                else []
            )
            item_detections = (
                items["detections"] if items["model_status"] == "ok" else []
            )
            linked_people = link_people_to_detected_items(
                person_detections, item_detections
            )
            tracked_people = []
            if person_tracker is not None:
                tracked_people = person_tracker.observe(
                    frame,
                    timestamp_s=timestamp,
                    replay_boxes=[detection["bbox"] for detection in person_detections],
                )
            tracked_detections = (
                attach_track_ids(person_detections, tracked_people)
                if person_tracker is not None
                else []
            )
            linked_candidates = (
                attach_track_ids(linked_people, tracked_people)
                if person_tracker is not None
                else linked_people
            )
            event_id = analysis_event_id
            motion_rows = []
            if movement_analyzer is not None:
                motion_rows = [
                    movement_analyzer.observe(
                        event_id,
                        int(frame_id),
                        float(timestamp),
                        int(detection["track_id"]),
                        detection["bbox"],
                    )
                    for detection in tracked_detections
                    if detection.get("track_id") is not None
                ]
            fps_for_analysis = metadata.get("fps")
            if not isinstance(fps_for_analysis, (int, float)) or fps_for_analysis <= 0:
                fps_for_analysis = 30.0
            frame_diagonal = math.hypot(frame.shape[1], frame.shape[0])
            pair_rows, _ = analyze_interactions(
                previous_motion_rows + motion_rows,
                float(fps_for_analysis),
                proximity_px=frame_diagonal * 0.15,
                distance_change_px=frame_diagonal * 0.01,
            )
            current_interactions = [
                row for row in pair_rows if row["frame_id"] == int(frame_id)
            ]
            previous_motion_rows = motion_rows
            assessment, alert_event = assessment_engine.update(
                frame_id=int(frame_id),
                timestamp_s=float(timestamp),
                item_detections=item_detections,
                behavior=behavior_result,
                interaction_rows=current_interactions,
                item_model_ok=items["model_status"] == "ok",
                person_model_ok=people is not None
                and people["model_status"] == "ok",
            )
            linked_track_ids = {
                detection.get("track_id")
                for detection in linked_candidates
                if detection.get("track_id") is not None
            }
            interaction_track_ids = set()
            if assessment["signals"]["repeated_close_approach"]:
                persistent_pairs = {
                    tuple(pair)
                    for pair in assessment["signals"]["repeated_close_approach_pairs"]
                }
                interaction_track_ids = {
                    track_id
                    for row in current_interactions
                    if row["is_close"] is True
                    and tuple(sorted((row["track_id_a"], row["track_id_b"])))
                    in persistent_pairs
                    for track_id in (row["track_id_a"], row["track_id_b"])
                }
            displayed_people = []
            for detection in tracked_detections:
                track_id = detection.get("track_id")
                reasons = []
                if track_id in linked_track_ids:
                    reasons.append("near detected item")
                if track_id in interaction_track_ids:
                    reasons.append("repeated close approach")
                if not reasons:
                    continue
                displayed_people.append(
                    {
                        **detection,
                        "associated_items": next(
                            (
                                candidate.get("associated_items", [])
                                for candidate in linked_candidates
                                if candidate.get("track_id") == track_id
                            ),
                            [],
                        ),
                        "candidate_reason": "; ".join(reasons),
                    }
                )
            if cached_people_frame_id is not None:
                displayed_people = [
                    {
                        **detection,
                        "inference_frame_id": cached_people_frame_id,
                        "age_frames": max(0, frame_id - cached_people_frame_id),
                    }
                    for detection in displayed_people
                ]
            latencies.append((time.perf_counter() - prediction_start) * 1000)
            for name, result in (("items", items), ("people", people)):
                if result is not None and result["model_status"] == "ok" and (
                    name != "people" or people_fresh
                ):
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

            annotated = None
            if not args.no_display or dashboard_writer is not None:
                annotated = frame.copy()
                if people is not None and people["model_status"] == "ok":
                    annotated = draw_detections(
                        annotated,
                        displayed_people,
                        color=(255, 160, 40),
                    )
                annotated = draw_detections(
                    annotated,
                    items["detections"] if items["model_status"] == "ok" else [],
                )
                label = f"Threat assessment: {assessment['level']}"
                cv2.putText(
                    annotated,
                    label,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255)
                    if assessment["level"] == "HIGH_RISK" or failures
                    else (0, 200, 255)
                    if assessment["level"] == "SUSPICIOUS"
                    else (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if dashboard_writer is not None:
                evidence = [
                    {**detection, "source": name}
                    for name, result in (
                        ("object_detection", items),
                        ("person_detection", people),
                    )
                    if result is not None and result["model_status"] == "ok"
                    for detection in result["detections"]
                ]
                if person_detector is not None and people is not None:
                    evidence = [
                        {**detection, "source": "person_detection"}
                        for detection in displayed_people
                    ] + [
                        {**detection, "source": "object_detection"}
                        for detection in items["detections"]
                        if items["model_status"] == "ok"
                    ]
                now = time.perf_counter()
                fps_samples.append(now)
                fps_elapsed = (
                    fps_samples[-1] - fps_samples[0] if len(fps_samples) > 1 else 0.0
                )
                processing_fps = (
                    (len(fps_samples) - 1) / fps_elapsed if fps_elapsed > 0 else 0.0
                )
                try:
                    dashboard_writer.publish_frame(
                        annotated
                        if frames % args.dashboard_image_interval == 0
                        else None,
                        capture={
                            "status": "connected"
                            if capture_config.is_live_camera
                            else "reading_file",
                            "source_type": capture_config.source_type,
                            "source_label": capture_config.source_label,
                            "backend": capture.backend,
                            "fps": metadata["fps"],
                            "reconnect_count": capture.reconnect_count,
                            "read_failures": capture.failed_reads,
                        },
                        models={
                            "object_detection": item_detector.model_status,
                            "person_detection": person_detector.model_status
                            if person_detector
                            else "not_connected",
                            "behavior_classification": "error"
                            if behavior_error
                            else "ready"
                            if behavior_classifier
                            else "not_connected",
                            "tracking": "ready"
                            if person_tracker
                            else "not_connected",
                        },
                        performance={
                            "processing_fps": processing_fps,
                            "fps_measurement_window_s": round(fps_elapsed, 2),
                            "capture_to_dashboard_ms": max(
                                0.0, (now - metadata["monotonic_time"]) * 1000.0
                            ),
                            "mean_model_latency_ms": sum(latencies) / len(latencies)
                            if latencies
                            else 0.0,
                            "p95_model_latency_ms": float(np.percentile(latencies, 95))
                            if latencies
                            else 0.0,
                            "dropped_frames": 0,
                            "minimum_fps_target": 30,
                        },
                        current_frame={
                            "frame_id": int(frame_id),
                            "timestamp_s": float(timestamp),
                            "timestamp_source": metadata["timestamp_source"],
                            "detections": evidence,
                            "behavior": behavior_result,
                            "movement": motion_rows,
                            "interactions": current_interactions,
                            "tracking": {
                                "active_tracks": len(
                                    {
                                        item.get("track_id")
                                        for item in tracked_detections
                                        if item.get("track_id") is not None
                                    }
                                ),
                                "highlighted_candidates": len(displayed_people),
                                "status": (
                                    "not_connected"
                                    if not person_tracker
                                    else "active"
                                    if displayed_people and item_detections
                                    else "interaction_review"
                                    if assessment["signals"]["repeated_close_approach"]
                                    else "item_detected_no_person_link"
                                    if item_detections
                                    else "waiting_for_item"
                                ),
                            },
                        },
                        assessment=assessment,
                    )
                    if alert_event:
                        dashboard_writer.record_event(alert_event)
                    if behavior_error:
                        print(
                            f"Behavior inference error: {behavior_error}",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    print(
                        f"Dashboard feed disabled: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    try:
                        dashboard_writer.close(status="error", error=str(exc))
                    finally:
                        dashboard_writer = None

            if not args.no_display and annotated is not None:
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
        if behavior_worker is not None:
            behavior_worker.close()
        if dashboard_writer is not None:
            try:
                dashboard_writer.close(
                    status="error" if error else "stopped",
                    error=error,
                )
            except OSError as exc:
                print(
                    f"ERROR: Could not finalize dashboard session: {exc}",
                    file=sys.stderr,
                )
                exit_code = 1
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
            "purpose": "Local camera demo with unvalidated multi-signal review assessment; not an accuracy study or confirmed incident record.",
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
