"""Frame-level detection API for the Campus Threat Detection pipeline.

Wraps the existing WeaponDetector (unchanged) and returns the project's
frame-level schema. This module detects objects only. It does not track,
classify behaviour, or decide whether anyone is dangerous.
"""
from __future__ import annotations

import gc
import math
import numbers
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.detector import WeaponDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "runs" / "detect" / "person1_clean"
    / "yolo11n_clean_v2" / "weights" / "best.pt"
)


class FrameValidationError(ValueError):
    """predict() received something that is not a valid BGR frame."""


class ModelLoadError(RuntimeError):
    """The model could not be loaded."""


def _is_int(value: Any) -> bool:
    """True for real integers (Python or NumPy). Rejects bool and floats."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _validate_frame(frame: Any) -> None:
    if not isinstance(frame, np.ndarray):
        raise FrameValidationError(
            f"frame must be a numpy.ndarray, got {type(frame).__name__}"
        )
    if frame.dtype != np.uint8:
        raise FrameValidationError(f"frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise FrameValidationError(
            f"frame must have shape (H, W, 3) in BGR order, got {frame.shape}"
        )
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise FrameValidationError(f"frame has an empty dimension: {frame.shape}")


def _build_class_map(raw: Any) -> dict[int, str]:
    """Convert the model's class names (dict or list) to {int: str}."""
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    mapping = {int(k): str(v) for k, v in items}
    if not mapping or any(k < 0 for k in mapping):
        raise ValueError("model reports an empty or invalid class mapping")
    return mapping


def _normalise(
    det: Any, width: int, height: int, class_map: dict[int, str]
) -> Optional[dict]:
    """Return a clean detection dict, or None if the detection is malformed.

    class_id must be an integer present in class_map, and class_name must equal
    the model's name for that id. Mismatches are rejected, not rewritten.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in det["xyxy"])
        confidence = float(det["confidence"])
        raw_id = det["class_id"]
        class_name = det["class_name"]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence)):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    if not _is_int(raw_id):
        return None
    class_id = int(raw_id)
    if class_map.get(class_id) != class_name:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = (min(max(v, 0.0), float(width)) for v in (x1, x2))
    y1, y2 = (min(max(v, 0.0), float(height)) for v in (y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "bbox": [x1, y1, x2, y2],
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
    }


class YOLOObjectDetector:
    """Loads the model once; call predict() once per frame."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.model_status = "not_loaded"
        self._detector: Optional[WeaponDetector] = None
        self._class_map: dict[int, str] = {}
        self.load_model()

    def load_model(self) -> None:
        """Load the model. No-op if already loaded; reloads after close()."""
        if self._detector is not None:
            return
        if not self.model_path.is_file():
            self.model_status = "load_failed"
            raise ModelLoadError(f"Model file not found: {self.model_path}")
        try:
            detector = WeaponDetector(
                model_path=str(self.model_path),
                confidence=self.confidence_threshold,
                device=self.device,
            )
            class_map = _build_class_map(detector.class_names)
        except Exception as exc:
            self.model_status = "load_failed"
            raise ModelLoadError(
                f"Failed to load {self.model_path}: {type(exc).__name__}: {exc}"
            ) from exc
        self._detector = detector
        self._class_map = class_map
        self.model_status = "ready"

    def warmup(self, runs: int = 1, size: tuple[int, int] = (640, 640)) -> None:
        """Run dummy frames so later latency measurements exclude warm-up.

        runs: integer >= 1. size: (height, width), positive integers.
        Errors raised by the model during warm-up are not caught.
        """
        if self._detector is None:
            raise ModelLoadError("Model is not loaded; call load_model() first.")
        if not _is_int(runs) or runs < 1:
            raise ValueError(f"runs must be an integer >= 1, got {runs!r}")
        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or not all(_is_int(v) and v >= 1 for v in size)
        ):
            raise ValueError(
                f"size must be (height, width) of positive integers, got {size!r}"
            )
        dummy = np.zeros((int(size[0]), int(size[1]), 3), dtype=np.uint8)
        for _ in range(runs):
            self._detector.detect(dummy)

    def predict(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> dict:
        """Run detection on one BGR frame.

        Invalid frames raise FrameValidationError. Inference failures do not
        raise: they return empty detections with model_status="inference_error"
        so a live pipeline keeps running. A timestamp that was not supplied
        stays None; inference time is never substituted for capture time.
        Detections with an invalid class_id, a class_name that does not match
        the model's mapping, or invalid geometry are dropped and counted in
        dropped_detections.
        """
        if self._detector is None:
            raise ModelLoadError("Model is not loaded; call load_model() first.")
        _validate_frame(frame)
        height, width = frame.shape[:2]

        status, error = "ok", None
        detections: list[dict] = []
        dropped = 0
        start = time.perf_counter()
        try:
            for raw in self._detector.detect(frame):
                clean = _normalise(raw, width, height, self._class_map)
                if clean is None:
                    dropped += 1
                else:
                    detections.append(clean)
        except Exception as exc:
            status, error = "inference_error", f"{type(exc).__name__}: {exc}"
            detections, dropped = [], 0
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "detections": detections,
            "inference_time_ms": elapsed_ms,
            "model_status": status,
            "error": error,
            "dropped_detections": dropped,
        }

    def get_model_info(self) -> dict:
        resolved_device = self._detector.device if self._detector else None
        return {
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "device_requested": self.device,
            "device_resolved": resolved_device,
            "class_names": dict(self._class_map),
            "model_status": self.model_status,
        }

    def close(self) -> None:
        """Release the model. Safe to call repeatedly; load_model() reloads."""
        self._detector = None
        self._class_map = {}
        self.model_status = "closed"
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
