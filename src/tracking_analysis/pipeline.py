"""Run one video through continuous or event-gated tracking."""

import csv
import hashlib
import json
import math
import platform
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
from src.camera_source import CameraConfig, CameraSource

from .analytics import (
    MOVEMENT_FIELDS,
    PAIR_FIELDS,
    PROXIMITY_FIELDS,
    MovementAnalyzer,
    analyze_interactions,
)
from .gate import EventGate
from .person_tracker import (
    PERSON_CONFIDENCE,
    TRACK_ACTIVATION_THRESHOLD,
    TRACK_MIN_CONSECUTIVE_FRAMES,
    PersonTracker,
)


MODEL_DIR = Path(__file__).resolve().parents[2] / "assets" / "tracking" / "models"
ITEM_CONFIDENCE = 0.35
FRAME_FIELDS = (
    "frame_id",
    "timestamp_s",
    "event_id",
    "detector_status",
    "trigger",
    "person_count",
    "trigger_count",
    "detector_error",
)


@dataclass(frozen=True)
class ReplayFrame:
    people: tuple
    trigger: bool
    inference_error: bool


EMPTY_REPLAY_FRAME = ReplayFrame((), False, False)


def read_replay(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("frames"), dict):
        raise ValueError("Replay must contain a 'frames' object.")

    frames = {}
    for key, value in data["frames"].items():
        if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(f"Invalid replay frame index: {key!r}.")
        if not isinstance(value, dict):
            raise ValueError(f"Replay frame {key} must contain an object.")
        people = value.get("people", [])
        trigger = value.get("trigger", False)
        inference_error = value.get("inference_error", False)
        if not isinstance(people, list):
            raise ValueError(f"Replay frame {key} people must be an array.")
        if not isinstance(trigger, bool) or not isinstance(inference_error, bool):
            raise ValueError(f"Replay frame {key} trigger and inference_error must be booleans.")
        for bbox in people:
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(type(coord) in (int, float) and math.isfinite(coord) for coord in bbox)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise ValueError(f"Invalid xyxy person box in replay frame {key}: {bbox!r}.")
        frames[int(key)] = ReplayFrame(tuple(people), trigger, inference_error)
    return frames


@contextmanager
def open_capture(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        yield capture
    finally:
        capture.release()


@contextmanager
def open_video_source(path):
    """Read a recorded clip through the shared, timestamped capture interface."""
    source = CameraSource(
        CameraConfig(source=Path(path), backend="auto", reconnect_attempts=0)
    )
    try:
        source.open()
        yield source
    finally:
        source.close()


@contextmanager
def open_writer(path, fps, width, height):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        if not writer.isOpened():
            raise ValueError(f"Cannot open MP4 writer: {path}")
        yield writer
    finally:
        writer.release()


def write_table(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weight_metadata(path):
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    return {"file": path.name, "sha256": file_sha256(path)}


def draw_box(frame, bbox, label, color):
    x1, y1, x2, y2 = [round(coord) for coord in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
    )


def annotate(frame, frame_info, motion, candidate_objects):
    for detection in candidate_objects:
        draw_box(
            frame, detection["bbox"],
            f'{detection["class_name"]} candidate {detection["confidence"]:.2f}', (0, 165, 255)
        )
    event_id = frame_info["event_id"]
    for observation in motion:
        scoped_label = f'{event_id}:{observation["track_id"]}'
        bbox = tuple(observation[field] for field in ("x1", "y1", "x2", "y2"))
        draw_box(frame, bbox, scoped_label, (40, 224, 120))

    status_text = "inactive" if event_id is None else f"event {event_id}"
    if frame_info["detector_status"] == "inference_error":
        status_text += " | trigger unavailable"
    cv2.putText(
        frame, f'{frame_info["frame_id"]} | {status_text}', (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA
    )


def process_frames(capture, writer, fps, size, person_tracker, gate, detector, replay_frames, analyzer):
    frame_rows = []
    motion_rows = []
    previous_event = None
    person_stage_calls = 0
    item_stage_calls = 0
    person_stage_s = 0.0
    item_stage_s = 0.0

    while True:
        packet = capture.read()
        if packet is None:
            break
        frame = packet.frame
        frame_id = packet.metadata["frame_id"]
        timestamp_s = packet.metadata["timestamp_s"]
        if (frame.shape[1], frame.shape[0]) != size:
            raise ValueError(f"Frame {frame_id} differs from the input video dimensions.")

        frame_replay = None if replay_frames is None else replay_frames.get(frame_id, EMPTY_REPLAY_FRAME)
        candidate_objects = []
        detector_status = "not_run"
        detector_error = None
        trigger = False
        trigger_count = 0
        event_id = 1
        if gate is not None:
            stage_start = time.perf_counter()
            item_stage_calls += 1
            if frame_replay is None:
                result = detector.predict(frame, frame_id=frame_id, timestamp=timestamp_s)
                detector_status = result["model_status"]
                detector_error = result["error"]
                candidate_objects = result["detections"] if detector_status == "ok" else []
                trigger_count = len(candidate_objects)
            elif frame_replay.inference_error:
                detector_status = "inference_error"
                detector_error = "Replay fixture declared an inference error."
            else:
                detector_status = "ok"
                trigger_count = int(frame_replay.trigger)
            trigger = trigger_count > 0
            event_id = gate.update(frame_id, trigger, inference_ok=detector_status == "ok")
            item_stage_s += time.perf_counter() - stage_start

        motion = []
        if event_id is not None:
            stage_start = time.perf_counter()
            if event_id != previous_event:
                person_tracker.reset()
            replay_boxes = None if frame_replay is None else frame_replay.people
            tracks = person_tracker.observe(frame, timestamp_s, replay_boxes=replay_boxes)
            person_stage_calls += 1
            person_stage_s += time.perf_counter() - stage_start
            motion = [
                analyzer.observe(event_id, frame_id, timestamp_s, track_id, bbox)
                for track_id, bbox in tracks
            ]
            motion_rows.extend(motion)
        previous_event = event_id

        frame_info = {
            "frame_id": frame_id,
            "timestamp_s": timestamp_s,
            "event_id": event_id,
            "detector_status": detector_status,
            "trigger": trigger,
            "person_count": len(motion),
            "trigger_count": trigger_count,
            "detector_error": detector_error,
        }
        frame_rows.append(frame_info)
        annotate(frame, frame_info, motion, candidate_objects)
        writer.write(frame)

    counts = {
        "person_stage_calls": person_stage_calls,
        "item_stage_calls": item_stage_calls,
        "person_stage_s": round(person_stage_s, 6),
        "item_stage_s": round(item_stage_s, 6),
        "person_model_calls": person_stage_calls if replay_frames is None else 0,
        "item_model_calls": item_stage_calls if replay_frames is None else 0,
    }
    return frame_rows, motion_rows, counts


def verify_video(path, expected_count, expected_fps=None, expected_size=None):
    count = 0
    with open_capture(path) as capture:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if expected_fps is not None and not math.isclose(fps, expected_fps, rel_tol=0.001):
            raise ValueError(f"Output MP4 FPS is {fps}, expected {expected_fps}.")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            size = frame.shape[1], frame.shape[0]
            if expected_size is not None and size != expected_size:
                raise ValueError(f"Output MP4 dimensions are {size}, expected {expected_size}.")
            count += 1
    if count != expected_count:
        raise ValueError(f"Output MP4 decoded {count} frames, expected {expected_count}.")


def run(
    video, output, mode, *, person_weights=None, item_weights=None, replay=None,
    device="cpu", grace=0.8, proximity_px=150, distance_change_px=5, max_speed_px_s=600
):
    if mode not in ("continuous", "gated"):
        raise ValueError(f"Unknown tracking mode: {mode!r}.")
    for name, value in (
        ("grace", grace),
        ("proximity-px", proximity_px),
        ("distance-change-px", distance_change_px),
        ("max-speed-px-s", max_speed_px_s),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if max_speed_px_s == 0:
        raise ValueError("max-speed-px-s must be positive.")

    video = Path(video)
    output = Path(output)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists. Choose a new run directory: {output}")

    replay_frames = None if replay is None else read_replay(replay)
    person_path = None if replay is not None else Path(person_weights or MODEL_DIR / "yolov8n.pt")
    item_path = (
        Path(item_weights or MODEL_DIR / "best.pt")
        if replay is None and mode == "gated"
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".tracking-run-") as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        output_video = stage / "annotated.mp4"

        with open_video_source(video) as capture:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Input video has invalid FPS: {fps}.")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError(f"Input video has invalid dimensions: {width} × {height}.")

            with open_writer(output_video, fps, width, height) as writer:
                startup_start = time.perf_counter()
                person_info = weight_metadata(person_path)
                item_info = weight_metadata(item_path)
                person_tracker = PersonTracker(person_path, device)
                detector = None
                if item_path is not None:
                    from src.models.object_detector import YOLOObjectDetector

                    detector = YOLOObjectDetector(item_path, confidence_threshold=ITEM_CONFIDENCE, device=device)
                gate = EventGate(fps, grace) if mode == "gated" else None
                analyzer = MovementAnalyzer(max_speed_px_s=max_speed_px_s)
                startup_s = time.perf_counter() - startup_start
                start = time.perf_counter()
                frame_rows, motion_rows, counts = process_frames(
                    capture, writer, fps, (width, height), person_tracker, gate, detector,
                    replay_frames, analyzer
                )

        frame_count = len(frame_rows)
        if not frame_count:
            raise ValueError(f"Input video has no decodable frames: {video}")
        if replay_frames and (last_replay_frame := max(replay_frames)) >= frame_count:
            raise ValueError(f"Replay frame {last_replay_frame} lies outside the {frame_count}-frame video.")
        pair_rows, proximity_rows = analyze_interactions(
            motion_rows, fps, proximity_px=proximity_px, distance_change_px=distance_change_px
        )

        write_table(stage / "frames.csv", FRAME_FIELDS, frame_rows)
        write_table(stage / "movement.csv", MOVEMENT_FIELDS, motion_rows)
        write_table(stage / "pairs.csv", PAIR_FIELDS, pair_rows)
        write_table(stage / "proximity.csv", PROXIMITY_FIELDS, proximity_rows)
        elapsed_s = time.perf_counter() - start
        verify_video(output_video, frame_count, expected_fps=fps, expected_size=(width, height))

        events = {row["event_id"] for row in frame_rows if row["event_id"] is not None}
        summary = {
            "mode": mode,
            "video": video.name,
            "video_sha256": file_sha256(video),
            "replay": None if replay is None else Path(replay).name,
            "replay_sha256": None if replay is None else file_sha256(Path(replay)),
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "active_frames": counts["person_stage_calls"],
            "event_count": len(events),
            "trigger_frames": sum(row["trigger"] for row in frame_rows),
            "inference_errors": sum(row["detector_status"] == "inference_error" for row in frame_rows),
            "track_count": len({(row["event_id"], row["track_id"]) for row in motion_rows}),
            "movement_rows": len(motion_rows),
            "pair_rows": len(pair_rows),
            "proximity_episodes": len(proximity_rows),
            "startup_s": round(startup_s, 6),
            "elapsed_s": round(elapsed_s, 6),
            "processing_fps": round(frame_count / elapsed_s, 3),
            **counts,
            "environment": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "trackers": version("trackers"),
                "supervision": version("supervision"),
                "ultralytics": version("ultralytics") if replay_frames is None else None,
            },
            "config": {
                "device": device,
                "grace_s": grace,
                "proximity_px": proximity_px,
                "distance_change_px": distance_change_px,
                "max_speed_px_s": max_speed_px_s,
                "item_confidence": ITEM_CONFIDENCE,
                "person_confidence": PERSON_CONFIDENCE,
                "track_activation_threshold": TRACK_ACTIVATION_THRESHOLD,
                "track_min_consecutive_frames": TRACK_MIN_CONSECUTIVE_FRAMES,
                "person_weights": person_info,
                "item_weights": item_info,
            },
        }

        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return summary
