"""Rolling-window inference for timestamped video frames."""

import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.behavior.data.preprocessing import KINETICS_MEAN, KINETICS_STD

from .temporal_model import TemporalViolenceModel, build_temporal_model

logger = logging.getLogger("ViolenceClassifier.Inference")


@dataclass
class ClassificationResult:
    """Prediction and timing information for one sampled video window."""

    timestamp: float
    window_start: float
    window_end: float
    violence_probability: float
    model_status: str
    raw_probability: float = 0.0
    is_violence: bool = False
    buffered_frames: int = 0
    required_frames: int = 0
    inference_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BehaviorClassifier:
    """Consume timestamped frames and classify rolling video windows."""

    def __init__(
        self,
        model: Optional[TemporalViolenceModel] = None,
        checkpoint_path: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        clip_length: Optional[int] = None,
        sampling_stride: Optional[int] = None,
        spatial_resolution: Optional[Tuple[int, int]] = None,
        threshold: float = 0.50,
        temporal_smoothing_window: int = 3,
        device: Optional[Union[str, torch.device]] = None,
    ):
        checkpoint = None
        checkpoint_config = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if isinstance(checkpoint, dict):
                checkpoint_config = checkpoint.get("config")
        elif checkpoint_path:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        effective_config = checkpoint_config or model_config or {}
        dataset_config = effective_config.get("dataset", {})
        evaluation_config = (model_config or effective_config).get("evaluation", {})
        self.clip_length = int(
            clip_length
            if clip_length is not None
            else dataset_config.get("clip_length", 16)
        )
        self.sampling_stride = int(
            sampling_stride
            if sampling_stride is not None
            else dataset_config.get("sampling_stride", 2)
        )
        configured_resolution = spatial_resolution or dataset_config.get(
            "spatial_resolution", (224, 224)
        )
        self.target_w, self.target_h = map(int, configured_resolution)
        self.threshold = float(evaluation_config.get("decision_threshold", threshold))
        self.smoothing_window_size = max(
            1,
            int(
                evaluation_config.get(
                    "temporal_smoothing_window", temporal_smoothing_window
                )
            ),
        )
        normalization = dataset_config.get("normalization", {})
        self.mean = np.asarray(
            normalization.get("mean", KINETICS_MEAN), dtype=np.float32
        ).reshape(1, 1, 1, 3)
        self.std = np.asarray(
            normalization.get("std", KINETICS_STD), dtype=np.float32
        ).reshape(1, 1, 1, 3)

        # Buffer capacity needed to form one clip
        self.required_buffer_size = (self.clip_length - 1) * self.sampling_stride + 1
        self.frame_buffer = deque(maxlen=self.required_buffer_size)
        self.recent_predictions = deque(maxlen=self.smoothing_window_size)
        self.lock = Lock()

        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Model instantiation
        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint is not None:
            saved_cfg = (
                checkpoint_config
                or effective_config
                or {
                    "model": {
                        "architecture": "r2plus1d_18",
                        "num_classes": 1,
                        "pretrained": False,
                    }
                }
            )
            saved_cfg = {
                **saved_cfg,
                "model": {**saved_cfg.get("model", {}), "pretrained": False},
            }
            self.model = build_temporal_model(saved_cfg).to(self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
        else:
            cfg = effective_config or {
                "model": {
                    "architecture": "r2plus1d_18",
                    "num_classes": 1,
                    "pretrained": False,
                }
            }
            self.model = build_temporal_model(cfg).to(self.device)

        self.model.eval()
        logger.info(
            "BehaviorClassifier online on %s | Window: %d frames (stride %d) | Target: %dx%d",
            self.device,
            self.clip_length,
            self.sampling_stride,
            self.target_w,
            self.target_h,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads trained model weights from checkpoint.
        """
        logger.info("Loading checkpoint from: %s", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def set_threshold(self, threshold: float) -> None:
        """
        Dynamically adjusts decision threshold for false alarm tuning.
        """
        self.threshold = float(threshold)
        logger.info("Decision threshold adjusted to: %.3f", self.threshold)

    def reset(self) -> None:
        """
        Flushes frame buffer and prediction history.
        """
        with self.lock:
            self.frame_buffer.clear()
            self.recent_predictions.clear()
        logger.info("BehaviorClassifier rolling buffer reset.")

    def ingest_frame(
        self, frame: np.ndarray, timestamp: Optional[float] = None
    ) -> None:
        """
        Thread-safe frame ingestion from a camera or video capture adapter.

        Args:
            frame: OpenCV BGR frame (H, W, 3) or RGB.
            timestamp: Floating point wall-clock or video timestamp in seconds.
        """
        if timestamp is None:
            timestamp = time.time()

        # Convert to RGB if needed
        if frame.ndim == 3 and frame.shape[2] == 3:
            # Assume OpenCV BGR input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        with self.lock:
            self.frame_buffer.append((frame_rgb, float(timestamp)))

    def _prepare_clip_tensor(self) -> Tuple[torch.Tensor, float, float]:
        """
        Extracts and preprocesses fixed-length clip from rolling buffer.
        """
        buffer_list = list(self.frame_buffer)
        total_buffered = len(buffer_list)

        # Sample frames with specified stride backwards from most recent frame
        selected_pairs = []
        for i in range(self.clip_length):
            # Compute index from end
            offset = (self.clip_length - 1 - i) * self.sampling_stride
            idx = total_buffered - 1 - offset
            selected_pairs.append(buffer_list[idx])

        window_start = selected_pairs[0][1]
        window_end = selected_pairs[-1][1]

        # Resize and normalize frames
        resized_frames = []
        for frame, _ in selected_pairs:
            if (frame.shape[1], frame.shape[0]) != (self.target_w, self.target_h):
                f_resized = cv2.resize(
                    frame,
                    (self.target_w, self.target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                f_resized = frame
            resized_frames.append(f_resized)

        clip_np = np.stack(resized_frames, axis=0).astype(np.float32) / 255.0
        # ImageNet normalization
        clip_normalized = (clip_np - self.mean) / self.std

        # Shape: (1, C, T, H, W)
        tensor = (
            torch.from_numpy(clip_normalized).permute(3, 0, 1, 2).unsqueeze(0).float()
        )
        return tensor.to(self.device), window_start, window_end

    def infer(self) -> Dict[str, Any]:
        """
        Executes inference over current rolling window if sufficient frames are available.

        Returns exact dictionary schema:
        {
            "timestamp": 12.4,
            "window_start": 9.4,
            "window_end": 12.4,
            "violence_probability": 0.78,
            "model_status": "valid"
        }
        """
        with self.lock:
            current_buffer_len = len(self.frame_buffer)
            if current_buffer_len < self.required_buffer_size:
                latest_ts = self.frame_buffer[-1][1] if current_buffer_len > 0 else 0.0
                first_ts = (
                    self.frame_buffer[0][1] if current_buffer_len > 0 else latest_ts
                )
                result = ClassificationResult(
                    timestamp=round(latest_ts, 3),
                    window_start=round(first_ts, 3),
                    window_end=round(latest_ts, 3),
                    violence_probability=0.0,
                    model_status="buffering",
                    raw_probability=0.0,
                    is_violence=False,
                    buffered_frames=current_buffer_len,
                    required_frames=self.required_buffer_size,
                    inference_latency_ms=0.0,
                )
                return result.to_dict()

            clip_tensor, win_start, win_end = self._prepare_clip_tensor()
            current_timestamp = win_end

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(clip_tensor)
            raw_prob = float(torch.sigmoid(logits).squeeze().cpu().item())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Apply temporal moving average smoothing over recent predictions
        with self.lock:
            self.recent_predictions.append(raw_prob)
            smoothed_prob = float(np.mean(self.recent_predictions))

        result = ClassificationResult(
            timestamp=round(current_timestamp, 3),
            window_start=round(win_start, 3),
            window_end=round(win_end, 3),
            violence_probability=round(smoothed_prob, 4),
            model_status="valid",
            raw_probability=round(raw_prob, 4),
            is_violence=bool(smoothed_prob >= self.threshold),
            buffered_frames=self.required_buffer_size,
            required_frames=self.required_buffer_size,
            inference_latency_ms=round(latency_ms, 2),
        )

        return result.to_dict()

    def process_frame(
        self, frame: np.ndarray, timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Convenience method: Ingests one frame and immediately queries rolling inference.
        """
        self.ingest_frame(frame, timestamp=timestamp)
        return self.infer()
