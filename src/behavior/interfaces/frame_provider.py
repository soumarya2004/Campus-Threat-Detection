"""Bounded asynchronous adapter for the shared camera source."""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
from typing import Any

from src.camera_source import CameraConfig, CameraSource, CapturedFrame


logger = logging.getLogger("CampusThreatDetection.FrameProvider")


class StreamFrameProvider:
    """Read in the background and expose timestamped frames to inference.

    Live cameras use a one-frame, latest-only queue so a slow model does not
    accumulate stale frames. Recorded videos use a bounded FIFO queue and apply
    backpressure instead, preserving every decoded frame.
    """

    def __init__(
        self,
        source: int | str | Path = 0,
        target_fps: float = 30.0,
        max_queue_size: int = 30,
        *,
        latest_only: bool | None = None,
        backend: str = "auto",
        width: int | None = None,
        height: int | None = None,
        capture_fps: float | None = None,
        buffer_size: int | None = 1,
        reconnect_attempts: int = 2,
        reconnect_delay_seconds: float = 0.5,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if target_fps < 0:
            raise ValueError("target_fps must be nonnegative")
        if latest_only is not None and not isinstance(latest_only, bool):
            raise ValueError("latest_only must be a boolean or None")

        normalized_source: int | Path = source
        if isinstance(source, str):
            normalized_source = int(source) if source.strip().isdecimal() else Path(source)

        source_is_live = isinstance(normalized_source, int)
        self.latest_only = source_is_live if latest_only is None else latest_only
        self.target_fps = float(target_fps)
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        self.queue: queue.Queue[CapturedFrame] = queue.Queue(
            maxsize=1 if self.latest_only else max_queue_size
        )
        self.source_config = CameraConfig(
            source=normalized_source,
            backend=backend,
            width=width,
            height=height,
            fps=capture_fps,
            buffer_size=buffer_size,
            max_read_failures=5,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay_seconds=reconnect_delay_seconds,
        )
        self.source: CameraSource | None = None
        self.running = False
        self.thread: threading.Thread | None = None
        self.end_of_stream = False
        self.error: Exception | None = None
        self.dropped_frames = 0
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self.running:
            return
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("Previous capture worker is still shutting down")
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.source = CameraSource(self.source_config)
        self.source.open()
        self._stop_event.clear()
        self.end_of_stream = False
        self.error = None
        self.dropped_frames = 0
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="campus-video-capture",
            daemon=True,
        )
        self.thread.start()
        logger.info("Capture started from %s", self.source_config.source_label)

    def _put_latest(self, packet: CapturedFrame) -> None:
        if self.latest_only and self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            # A consumer can race the queue check; keep the newest frame.
            try:
                self.queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            self.queue.put_nowait(packet)

    def _put_in_order(self, packet: CapturedFrame) -> None:
        while self.running and not self._stop_event.is_set():
            try:
                self.queue.put(packet, timeout=0.1)
                return
            except queue.Full:
                continue

    def _capture_loop(self) -> None:
        assert self.source is not None
        try:
            while self.running and not self._stop_event.is_set():
                packet = self.source.read()
                if packet is None:
                    self.end_of_stream = True
                    break

                if self.latest_only:
                    self._put_latest(packet)
                    if self.frame_interval:
                        self._stop_event.wait(self.frame_interval)
                else:
                    self._put_in_order(packet)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.error = exc
                logger.exception("Capture stopped after an input error")
        finally:
            self.running = False
            if self.source is not None:
                self.source.close()

    def get_packet(self, timeout: float = 1.0) -> CapturedFrame | None:
        """Return the next captured frame, or None if no frame arrives in time."""

        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_frame(self, timeout: float = 1.0) -> tuple[Any, float] | None:
        """Compatibility helper returning a frame and its source-relative time."""

        packet = self.get_packet(timeout)
        if packet is None:
            return None
        return packet.frame, float(packet.metadata["timestamp_s"])

    def stop(self) -> None:
        self._stop_event.set()
        self.running = False
        thread = self.thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive() and self.source is not None:
                # Releasing the backend can unblock a driver stuck inside read().
                self.source.close()
                thread.join(timeout=0.5)
        elif self.source is not None:
            self.source.close()
        logger.info(
            "Capture stopped after %d frames; discarded %d stale frames",
            self.source.frame_id if self.source is not None else 0,
            self.dropped_frames,
        )

    def __enter__(self) -> StreamFrameProvider:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
