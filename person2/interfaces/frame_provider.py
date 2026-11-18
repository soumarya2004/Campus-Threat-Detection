"""Read timestamped frames from cameras and video files."""

import logging
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("ViolenceClassifier.FrameProvider")


class StreamFrameProvider:
    """
    Asynchronous capture worker for live camera feeds (RTSP/USB) or recorded video files.
    Decouples frame reading from model inference to prevent IO lag.
    """

    def __init__(
        self,
        source: str = "0",
        target_fps: float = 30.0,
        max_queue_size: int = 60,
    ):
        self.source = int(source) if source.isdigit() else source
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.033
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video stream source: {self.source}")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Frame provider started for source: %s", self.source)

    def _capture_loop(self) -> None:
        t_start = time.time()
        while self.running and self.cap.isOpened():
            t_loop = time.time()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.info("Stream source reached end or disconnected.")
                break

            timestamp = time.time() - t_start
            # Push to queue (drop oldest frame if queue full to prevent lag)
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            self.queue.put((frame, timestamp))

            # Maintain target FPS
            elapsed = time.time() - t_loop
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.running = False

    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, float]]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        logger.info("Frame provider stopped.")
