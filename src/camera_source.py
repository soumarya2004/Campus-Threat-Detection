"""Shared OpenCV source for live cameras and recorded video."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Integral, Real
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit

import cv2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "configs" / "camera.yaml"

_BACKENDS = {
    "auto": None,
    "any": cv2.CAP_ANY,
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
}


class CaptureError(RuntimeError):
    """Base error for opening or reading a configured video source."""


class CaptureOpenError(CaptureError):
    """The configured camera or video could not be opened."""


class CaptureReadError(CaptureError):
    """A live camera did not recover after the configured retries."""


@dataclass(frozen=True)
class CameraConfig:
    """Validated settings for one capture source."""

    source: int | Path | str = 0
    backend: str = "auto"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    buffer_size: int | None = 1
    max_read_failures: int = 5
    reconnect_attempts: int = 2
    reconnect_delay_seconds: float = 0.5

    def __post_init__(self) -> None:
        source = self.source
        if isinstance(source, bool):
            raise ValueError(
                "source must be a nonnegative camera index, video path, or HTTP(S) stream URL"
            )
        if isinstance(source, Integral):
            if source < 0:
                raise ValueError("camera index must be nonnegative")
            object.__setattr__(self, "source", int(source))
        elif isinstance(source, (str, Path)):
            value = str(source).strip()
            if not value:
                raise ValueError("video path must not be empty")
            if value.isdecimal():
                object.__setattr__(self, "source", int(value))
            elif "://" in value:
                parsed = urlsplit(value)
                if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                    raise ValueError("network source must be an HTTP(S) URL with a hostname")
                if parsed.username is not None or parsed.password is not None:
                    raise ValueError("network source URLs must not contain embedded credentials")
                try:
                    parsed.port
                except ValueError as exc:
                    raise ValueError("network source URL has an invalid port") from exc
                object.__setattr__(self, "source", value)
            else:
                object.__setattr__(self, "source", Path(value).expanduser())
        else:
            raise ValueError(
                "source must be a nonnegative camera index, video path, or HTTP(S) stream URL"
            )

        if not isinstance(self.backend, str):
            raise ValueError("backend must be a string")
        backend = self.backend.strip().lower()
        if backend not in _BACKENDS:
            raise ValueError(
                f"backend must be one of {', '.join(_BACKENDS)}, got {self.backend!r}"
            )
        object.__setattr__(self, "backend", backend)

        for name in ("width", "height", "buffer_size", "max_read_failures"):
            value = getattr(self, name)
            if value is None and name in ("width", "height", "buffer_size"):
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name == "buffer_size" else 1):
                raise ValueError(f"{name} is outside its valid range")
            object.__setattr__(self, name, int(value))

        if isinstance(self.reconnect_attempts, bool) or not isinstance(
            self.reconnect_attempts, Integral
        ):
            raise ValueError("reconnect_attempts must be a nonnegative integer")
        if self.reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be a nonnegative integer")
        object.__setattr__(self, "reconnect_attempts", int(self.reconnect_attempts))

        for name in ("fps", "reconnect_delay_seconds"):
            value = getattr(self, name)
            if value is None and name == "fps":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            if value < (0 if name == "reconnect_delay_seconds" else 0.001):
                raise ValueError(f"{name} is outside its valid range")
            object.__setattr__(self, name, float(value))

    @property
    def is_live_camera(self) -> bool:
        return isinstance(self.source, int) or isinstance(self.source, str)

    @property
    def source_type(self) -> str:
        if isinstance(self.source, int):
            return "live_camera"
        if isinstance(self.source, str):
            return "network_stream"
        return "video_file"

    @property
    def source_label(self) -> str:
        if isinstance(self.source, int):
            return f"camera:{self.source}"
        if isinstance(self.source, str):
            parsed = urlsplit(self.source)
            host = parsed.hostname or "network-stream"
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            port = f":{parsed.port}" if parsed.port is not None else ""
            return f"{host}{port}{parsed.path or '/'}"
        return Path(self.source).name


@dataclass(frozen=True)
class CapturedFrame:
    """An OpenCV BGR frame and the timing/capture metadata for that frame."""

    frame: Any
    metadata: dict[str, Any]


def _read_source(value: Any) -> int | Path | str:
    if isinstance(value, bool):
        raise ValueError("source must be a camera index, video path, or HTTP(S) stream URL")
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source must be a camera index, video path, or HTTP(S) stream URL")
    value = value.strip()
    if value.isdecimal():
        return int(value)
    if "://" in value:
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_camera_config(config_path: str | Path | None = None) -> CameraConfig:
    """Load the shared camera settings; relative video paths use the repo root."""

    path = Path(config_path).expanduser() if config_path else DEFAULT_CAMERA_CONFIG
    try:
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read camera config {path}: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError(f"Expected a mapping in camera config {path}")
    if any(not isinstance(key, str) for key in settings):
        raise ValueError("Camera config keys must be strings")

    fields = {
        "source",
        "backend",
        "width",
        "height",
        "fps",
        "buffer_size",
        "max_read_failures",
        "reconnect_attempts",
        "reconnect_delay_seconds",
    }
    unknown = settings.keys() - fields
    if unknown:
        raise ValueError(f"Unknown camera settings: {', '.join(sorted(unknown))}")

    if "source" in settings:
        settings["source"] = _read_source(settings["source"])
    return CameraConfig(**settings)


class CameraSource:
    """Capture frames with stable IDs, source timing and bounded reconnects.

    `read()` returns a `CapturedFrame` for each decoded frame and `None` at the
    end of a recorded file. A live camera retries after repeated failed reads,
    then raises `CaptureReadError` if it cannot resume. Returned frames are not
    copied; callers should treat them as read-only while inference is running.
    """

    def __init__(
        self,
        config: CameraConfig | None = None,
        *,
        capture_factory=None,
        clock=None,
        sleep=None,
    ) -> None:
        self.config = config or load_camera_config()
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._clock = clock or time.perf_counter
        self._sleep = sleep or time.sleep
        self._capture = None
        self._closed = False
        self._session_started: float | None = None
        self._last_capture_time: float | None = None
        self._last_timestamp_s: float | None = None
        self._last_media_timestamp_s: float | None = None
        self._fps = 0.0
        self.frame_id = 0
        self.failed_reads = 0
        self.reconnect_count = 0
        self.backend: str | None = None
        self.capture_properties: dict[str, bool] = {}

    @property
    def source_type(self) -> str:
        return self.config.source_type

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._is_opened(self._capture)

    def _backend_id(self):
        return _BACKENDS[self.config.backend]

    @staticmethod
    def _is_opened(capture) -> bool:
        try:
            return bool(capture.isOpened())
        except (AttributeError, cv2.error):
            return False

    def _create_capture(self):
        source = self.config.source
        value = source if isinstance(source, int) else str(source)
        # URL streams should use OpenCV's automatic backend selection (FFmpeg
        # on Windows), even when a camera index backend is configured.
        backend = None if isinstance(source, str) else self._backend_id()
        try:
            if backend is None:
                return self._capture_factory(value)
            return self._capture_factory(value, backend)
        except cv2.error:
            return None

    def _apply_capture_settings(self) -> None:
        capture = self._capture
        if capture is None:
            return
        requested = (
            ("width", cv2.CAP_PROP_FRAME_WIDTH, self.config.width),
            ("height", cv2.CAP_PROP_FRAME_HEIGHT, self.config.height),
            ("fps", cv2.CAP_PROP_FPS, self.config.fps),
            ("buffer_size", cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size),
        )
        for name, prop, value in requested:
            if value is None or (name == "buffer_size" and not self.config.is_live_camera):
                continue
            try:
                self.capture_properties[name] = bool(capture.set(prop, value))
            except (AttributeError, cv2.error):
                self.capture_properties[name] = False

    def _release_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except cv2.error:
                pass

    def open(self) -> None:
        if self._closed:
            raise CaptureOpenError("This source has already been closed")
        if self.is_open:
            return
        self._release_capture()
        capture = self._create_capture()
        if not self._is_opened(capture):
            if capture is not None:
                capture.release()
            raise CaptureOpenError(
                f"Could not open {self.config.source_type.replace('_', ' ')} "
                f"{self.config.source_label} using backend {self.config.backend}"
            )
        self._capture = capture
        self._apply_capture_settings()
        try:
            self.backend = capture.getBackendName()
        except (AttributeError, cv2.error):
            self.backend = self.config.backend.upper()
        self._session_started = self._clock()

    def _reopen_once(self) -> bool:
        if self._closed:
            return False
        self._release_capture()
        if self.config.reconnect_delay_seconds:
            self._sleep(self.config.reconnect_delay_seconds)
        capture = self._create_capture()
        if not self._is_opened(capture):
            if capture is not None:
                capture.release()
            return False
        self._capture = capture
        self._apply_capture_settings()
        try:
            self.backend = capture.getBackendName()
        except (AttributeError, cv2.error):
            self.backend = self.config.backend.upper()
        self._last_capture_time = None
        self.reconnect_count += 1
        return True

    def _make_packet(self, frame) -> CapturedFrame:
        now = self._clock()
        if self._session_started is None:
            self._session_started = now
        elapsed_s = max(0.0, now - self._session_started)

        previous_capture_time = self._last_capture_time
        if self._last_capture_time is not None:
            interval = now - self._last_capture_time
            if interval > 0:
                instant_fps = 1.0 / interval
                self._fps = (
                    instant_fps if self._fps == 0 else 0.9 * self._fps + 0.1 * instant_fps
                )
        self._last_capture_time = now

        timestamp_s = elapsed_s
        timestamp_source = "host_monotonic"
        media_timestamp_s = None
        if not self.config.is_live_camera and self._capture is not None:
            try:
                raw_timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            except (AttributeError, TypeError, ValueError, cv2.error):
                raw_timestamp = math.nan
            if math.isfinite(raw_timestamp) and raw_timestamp >= 0:
                media_timestamp_s = raw_timestamp
                if (
                    self._last_media_timestamp_s is None
                    or raw_timestamp > self._last_media_timestamp_s
                ):
                    timestamp_s = raw_timestamp
                    timestamp_source = "video_media"
                    self._last_media_timestamp_s = raw_timestamp

        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            monotonic_step = max(
                0.000001,
                now - previous_capture_time if previous_capture_time is not None else 0.0,
            )
            timestamp_s = self._last_timestamp_s + monotonic_step
            timestamp_source = "host_monotonic_fallback"
        self._last_timestamp_s = timestamp_s

        self.frame_id += 1
        height, width = frame.shape[:2]
        metadata = {
            "frame_id": self.frame_id - 1,
            "timestamp_s": float(timestamp_s),
            "timestamp_source": timestamp_source,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "monotonic_time": float(now),
            "fps": float(self._fps),
            "width": int(width),
            "height": int(height),
            "source": self.config.source_type,
            "backend": self.backend,
            "reconnect_count": self.reconnect_count,
            "read_failures": self.failed_reads,
        }
        if media_timestamp_s is not None:
            metadata["media_timestamp_s"] = media_timestamp_s
        return CapturedFrame(frame=frame, metadata=metadata)

    def read(self) -> CapturedFrame | None:
        """Read one frame; return None only when a recorded video reaches EOF."""

        if self._closed:
            raise CaptureReadError("Cannot read from a closed source")
        if not self.is_open:
            raise CaptureReadError("Source is not open; call open() first")

        if not self.config.is_live_camera:
            try:
                ok, frame = self._capture.read()
            except cv2.error:
                ok, frame = False, None
            if not ok or frame is None:
                return None
            return self._make_packet(frame)

        failures_in_row = 0
        reconnects_without_frame = 0
        while True:
            if self._closed:
                raise CaptureReadError("Camera source was closed while reading")
            if not self.is_open:
                if reconnects_without_frame >= self.config.reconnect_attempts:
                    raise CaptureReadError(
                        f"Camera {self.config.source} disconnected after "
                        f"{self.reconnect_count} successful reconnects"
                    )
                reconnects_without_frame += 1
                if not self._reopen_once():
                    continue

            try:
                ok, frame = self._capture.read()
            except cv2.error:
                ok, frame = False, None
            if ok and frame is not None:
                packet = self._make_packet(frame)
                return packet

            self.failed_reads += 1
            failures_in_row += 1
            if failures_in_row < self.config.max_read_failures:
                self._sleep(min(0.05, self.config.reconnect_delay_seconds))
                continue

            if reconnects_without_frame >= self.config.reconnect_attempts:
                raise CaptureReadError(
                    f"Camera {self.config.source} did not return a frame after "
                    f"{self.reconnect_count} successful reconnects"
                )
            reconnects_without_frame += 1
            if not self._reopen_once():
                failures_in_row = 0
                continue
            failures_in_row = 0

    def get(self, property_id: int) -> float:
        """Read an OpenCV capture property while the source is open."""

        if not self.is_open:
            return 0.0
        try:
            return float(self._capture.get(property_id))
        except (AttributeError, TypeError, ValueError, cv2.error):
            return 0.0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release_capture()

    def __enter__(self) -> CameraSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
