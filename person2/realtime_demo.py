"""Run rolling-window inference over a video, webcam, or synthetic clip."""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from data.synthetic_generator import generate_synthetic_video
from interfaces.alert_dispatcher import AlertDispatcher
from models.behavior_classifier import BehaviorClassifier

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RealtimeDemo")


def draw_hud_overlay(
    frame: np.ndarray,
    result: dict,
    threshold: float = 0.50,
    fps: float = 0.0,
) -> np.ndarray:
    """
    Renders an informative real-time telemetry HUD onto the video frame.
    """
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Probability and status
    prob = result.get("violence_probability", 0.0)
    status = result.get("model_status", "valid")
    w_start = result.get("window_start", 0.0)
    w_end = result.get("window_end", 0.0)
    latency = result.get("inference_latency_ms", 0.0)
    buffered = result.get("buffered_frames", 0)
    required = result.get("required_frames", 16)

    # Top overlay bar
    bar_height = 80
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Determine status color
    if status == "buffering":
        state_text = f"BUFFERING [{buffered}/{required}]"
        state_color = (180, 180, 180)
    elif prob >= 0.75:
        state_text = "HIGH VIOLENCE SCORE - REVIEW"
        state_color = (0, 0, 240)  # Bright Red
    elif prob >= threshold:
        state_text = "ELEVATED VIOLENCE SCORE"
        state_color = (0, 165, 255)  # Orange
    else:
        state_text = "BELOW VIOLENCE THRESHOLD"
        state_color = (60, 220, 60)  # Green

    # Status Header
    cv2.putText(
        canvas,
        state_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        state_color,
        2,
        cv2.LINE_AA,
    )

    # Probability Meter
    meter_x, meter_y, meter_w, meter_h = 15, 45, int(w * 0.45), 20
    cv2.rectangle(
        canvas,
        (meter_x, meter_y),
        (meter_x + meter_w, meter_y + meter_h),
        (60, 60, 60),
        -1,
    )
    fill_w = int(meter_w * np.clip(prob, 0.0, 1.0))

    # Fill color gradient from green to red
    fill_color = (int(60 + 195 * prob), int(220 * (1.0 - prob)), 60)
    cv2.rectangle(
        canvas,
        (meter_x, meter_y),
        (meter_x + fill_w, meter_y + meter_h),
        fill_color,
        -1,
    )
    cv2.rectangle(
        canvas,
        (meter_x, meter_y),
        (meter_x + meter_w, meter_y + meter_h),
        (200, 200, 200),
        1,
    )

    # Threshold indicator notch
    th_x = int(meter_x + meter_w * threshold)
    cv2.line(
        canvas, (th_x, meter_y - 2), (th_x, meter_y + meter_h + 2), (255, 255, 255), 2
    )

    # Text telemetry
    cv2.putText(
        canvas,
        f"Violence Prob: {prob * 100:5.1f}%",
        (meter_x + meter_w + 12, meter_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    telemetry = f"Window: [{w_start:4.1f}s -> {w_end:4.1f}s] | Latency: {latency:4.1f}ms | FPS: {fps:4.1f}"
    cv2.putText(
        canvas,
        telemetry,
        (15, bar_height - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Highlight border if alert active
    if prob >= threshold and status == "valid":
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), state_color, 4)

    return canvas


def run_demo(
    source_type: str = "synthetic",
    source_path: Optional[str] = None,
    checkpoint_path: str = "checkpoints/best_model.pt",
    config_path: str = "configs/training_config.yaml",
    save_output: Optional[str] = "logs/demo_output.mp4",
    max_frames: int = 150,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model first; "
            "the demo will not present random weights as violence predictions."
        )

    if source_type == "synthetic":
        syn_path = "./data/synthetic/demo_simulated_stream.mp4"
        if not os.path.exists(syn_path):
            logger.info("Generating a synthetic clip for the pipeline demonstration.")
            generate_synthetic_video(
                syn_path, duration_seconds=5.0, fps=25, behavior_type="violence"
            )
        source_path = syn_path
    elif source_path is None:
        raise ValueError(
            "Provide a camera index or video path for a non-synthetic source."
        )
    elif not source_path.isdigit() and not Path(source_path).is_file():
        raise FileNotFoundError(f"Video source not found: {source_path}")

    cap = cv2.VideoCapture(int(source_path) if source_path.isdigit() else source_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video source: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # Initialize Classifier
    classifier = BehaviorClassifier(
        checkpoint_path=checkpoint_path,
        model_config=cfg,
        clip_length=cfg["dataset"]["clip_length"],
        sampling_stride=cfg["dataset"]["sampling_stride"],
        spatial_resolution=tuple(cfg["dataset"]["spatial_resolution"]),
        threshold=cfg["evaluation"]["decision_threshold"],
        temporal_smoothing_window=cfg["evaluation"].get("temporal_smoothing_window", 3),
    )

    # Prepare the alert integration interface.
    dispatcher = AlertDispatcher(
        threshold_monitor=cfg["evaluation"]["decision_threshold"],
        threshold_alarm=cfg["evaluation"].get("high_confidence_threshold", 0.75),
    )

    # Video writer for saving demo output
    writer = None
    if save_output:
        os.makedirs(os.path.dirname(save_output), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, fps, (w, h))

    logger.info(
        "Starting real-time streaming demonstration (Max frames: %d)...", max_frames
    )
    frame_count = 0
    t_start = time.time()

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp = frame_count / fps
            frame_count += 1

            # Ingest frame and run rolling inference
            result = classifier.process_frame(frame, timestamp=timestamp)

            # Convert the current score into an alert state.
            event = dispatcher.process_prediction(result)

            # Compute running FPS
            current_fps = frame_count / (time.time() - t_start)

            # Render HUD
            annotated_frame = draw_hud_overlay(
                frame, result, threshold=classifier.threshold, fps=current_fps
            )

            # Terminal log every 10 frames
            if (
                frame_count % 10 == 0
                or result["violence_probability"] >= classifier.threshold
            ):
                print(
                    f"Frame {frame_count:04d} | "
                    f"Time: {result['timestamp']:5.2f}s | "
                    f"Window: [{result['window_start']:5.2f}s - {result['window_end']:5.2f}s] | "
                    f"Prob: {result['violence_probability']:.4f} | "
                    f"Status: {result['model_status']} | "
                    f"Alert: {event.alert_level}"
                )

            if writer is not None:
                writer.write(annotated_frame)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info("Saved annotated video recording to: %s", save_output)

    logger.info("Demonstration finished. Processed %d frames.", frame_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Behavior Classifier Demo")
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help="'synthetic', '0' for webcam, or video path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to config",
    )
    parser.add_argument(
        "--save_output",
        type=str,
        default="logs/demo_output.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--max_frames", type=int, default=100, help="Frames to simulate"
    )
    args = parser.parse_args()

    run_demo(
        source_type="file" if args.source != "synthetic" else "synthetic",
        source_path=args.source if args.source != "synthetic" else None,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_output=args.save_output,
        max_frames=args.max_frames,
    )
