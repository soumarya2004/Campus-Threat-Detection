"""Offline video inference used by the dashboard's recorded-video demo."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def _attach_track_ids(
    detections: list[dict], tracks: list[tuple[int, tuple]]
) -> list[dict]:
    """Associate current person boxes with the tracker output by IoU."""
    output = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        best_id, best_iou = None, 0.0
        for track_id, box in tracks:
            tx1, ty1, tx2, ty2 = box
            intersection = max(0.0, min(x2, tx2) - max(x1, tx1)) * max(
                0.0, min(y2, ty2) - max(y1, ty1)
            )
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            track_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
            union = area + track_area - intersection
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_id, best_iou = track_id, iou
        item = dict(detection)
        if best_iou >= 0.3:
            item["track_id"] = best_id
        output.append(item)
    return output


def _draw_detections(cv2, frame, detections: list[dict], color, *, show_tracks=False):
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        if show_tracks and detection.get("track_id") is not None:
            label = (
                f"Person #{detection['track_id']} "
                f"{detection['confidence']:.2f}"
            )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def process_uploaded_video(
    video_bytes: bytes,
    filename: str,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run object, person, tracking, and behavior models on one uploaded clip."""
    import cv2
    import numpy as np
    import torch

    from src.behavior.models.behavior_classifier import BehaviorClassifier
    from src.models.object_detector import YOLOObjectDetector
    from src.models.person_detector import YOLOPersonDetector
    from src.tracking_analysis.person_tracker import PersonTracker

    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError("Choose an MP4, MOV, AVI, MKV, or WebM video.")
    if not video_bytes:
        raise ValueError("The uploaded video is empty.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    input_capture = None
    writer = None
    object_detector = None
    person_detector = None
    with tempfile.TemporaryDirectory(prefix="campus-threat-video-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / f"input{suffix}"
        intermediate_path = temp_path / "annotated-mp4v.mp4"
        output_path = temp_path / "annotated.mp4"
        input_path.write_bytes(video_bytes)

        try:
            input_capture = cv2.VideoCapture(str(input_path))
            if not input_capture.isOpened():
                raise ValueError(
                    "The uploaded file could not be opened as a video. Try MP4 or MOV."
                )

            width = int(input_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_fps = float(input_capture.get(cv2.CAP_PROP_FPS))
            reported_frame_count = float(
                input_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            reported_frames = (
                int(reported_frame_count)
                if np.isfinite(reported_frame_count) and reported_frame_count > 0
                else 0
            )
            if width < 1 or height < 1:
                raise ValueError("The uploaded video has invalid frame dimensions.")
            if not np.isfinite(source_fps) or source_fps <= 0:
                source_fps = 30.0
            if reported_frames < 0:
                reported_frames = 0

            object_detector = YOLOObjectDetector(device=device, image_size=416)
            person_detector = YOLOPersonDetector(device=device, image_size=416)
            behavior_classifier = BehaviorClassifier(
                checkpoint_path=str(REPO_ROOT / "models/behavior/best_model.pt"),
                device=device,
            )
            tracker = PersonTracker(weights=None, device=device)
            tracker.reset()

            writer = cv2.VideoWriter(
                str(intermediate_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create the annotated MP4 output.")

            frame_index = 0
            object_detections_total = 0
            person_detections_total = 0
            inference_errors = 0
            active_tracks = 0
            latest_people: list[dict] = []
            behavior_result = None
            peak_behavior_score = None
            started = time.perf_counter()

            while True:
                ok, frame = input_capture.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                timestamp = frame_index / source_fps

                objects = object_detector.predict(
                    frame, frame_id=frame_index, timestamp=timestamp
                )
                if frame_index % 2 == 0 or not latest_people:
                    people = person_detector.predict(
                        frame, frame_id=frame_index, timestamp=timestamp
                    )
                    if people["model_status"] == "ok":
                        latest_people = people["detections"]
                        person_detections_total += len(latest_people)
                else:
                    people = {"model_status": "ok", "detections": latest_people}

                tracked_people = tracker.observe(
                    frame,
                    timestamp_s=timestamp,
                    replay_boxes=[
                        item["bbox"]
                        for item in people["detections"]
                        if people["model_status"] == "ok"
                    ],
                )
                displayed_people = _attach_track_ids(
                    people["detections"] if people["model_status"] == "ok" else [],
                    tracked_people,
                )
                active_tracks = len(tracked_people)

                behavior_classifier.ingest_frame(frame, timestamp=timestamp)
                if (
                    frame_index + 1 >= behavior_classifier.required_buffer_size
                    and (frame_index + 1) % 12 == 0
                ):
                    behavior_result = behavior_classifier.infer()
                    if behavior_result.get("model_status") == "valid":
                        score = behavior_result.get("violence_probability")
                        if isinstance(score, (int, float)):
                            peak_behavior_score = max(peak_behavior_score or 0.0, score)

                inference_errors += int(objects["model_status"] != "ok")
                if objects["model_status"] == "ok":
                    object_detections_total += len(objects["detections"])

                annotated = frame.copy()
                _draw_detections(
                    cv2, annotated, displayed_people, (255, 160, 40), show_tracks=True
                )
                _draw_detections(cv2, annotated, objects["detections"], (0, 0, 255))
                cv2.putText(
                    annotated,
                    f"Person tracks: {len(tracked_people)}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if (
                    behavior_result
                    and behavior_result.get("model_status") == "valid"
                ):
                    score = behavior_result["violence_probability"]
                    behavior_text = f"Behavior score: {score:.1%} (review only)"
                else:
                    buffered = min(
                        frame_index + 1, behavior_classifier.required_buffer_size
                    )
                    behavior_text = (
                        f"Behavior model warming up: {buffered}/"
                        f"{behavior_classifier.required_buffer_size} frames"
                    )
                cv2.putText(
                    annotated,
                    behavior_text,
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                frame_index += 1

                if progress_callback and (
                    frame_index % 15 == 0 or frame_index == reported_frames
                ):
                    progress_callback(frame_index, reported_frames)

            writer.release()
            writer = None
            input_capture.release()
            input_capture = None
            if frame_index == 0:
                raise ValueError("No frames could be decoded from the uploaded video.")

            import imageio_ffmpeg

            conversion = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-nostdin",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(intermediate_path),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if conversion.returncode != 0:
                detail = conversion.stderr.strip().splitlines()
                reason = detail[-1] if detail else "H.264 encoding failed."
                raise RuntimeError(f"Could not encode the browser-ready video: {reason}")

            elapsed = time.perf_counter() - started
            output_bytes = output_path.read_bytes()
            if progress_callback:
                progress_callback(frame_index, reported_frames or frame_index)
            return {
                "video_bytes": output_bytes,
                "frames": frame_index,
                "duration_seconds": frame_index / source_fps,
                "processing_seconds": elapsed,
                "processing_fps": frame_index / elapsed if elapsed else 0.0,
                "object_detections": object_detections_total,
                "person_detections": person_detections_total,
                "active_tracks": active_tracks,
                "peak_behavior_score": peak_behavior_score,
                "behavior_status": (
                    behavior_result.get("model_status", "buffering")
                    if behavior_result
                    else "buffering"
                ),
                "device": device,
                "inference_errors": inference_errors,
                "audio_retained": False,
            }
        finally:
            if writer is not None:
                writer.release()
            if input_capture is not None:
                input_capture.release()
            if object_detector is not None:
                object_detector.close()
            if person_detector is not None:
                person_detector.close()
