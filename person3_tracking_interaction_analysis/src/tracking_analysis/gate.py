from fractions import Fraction
from numbers import Integral


class EventGate:
    """Keep one event active through the inclusive frame-based grace period."""

    def __init__(self, fps: float, grace_s: float):
        fps = Fraction(str(fps))
        grace_s = Fraction(str(grace_s))
        if fps <= 0 or grace_s < 0:
            raise ValueError("FPS must be positive and grace must be nonnegative.")

        self._grace_frames = int(fps * grace_s)
        self._last_frame: int | None = None
        self._last_trigger_frame: int | None = None
        self._event_id = 0

    def update(
        self, frame_id: int, trigger: bool, inference_ok: bool = True
    ) -> int | None:
        if not isinstance(frame_id, Integral) or isinstance(frame_id, bool) or frame_id < 0:
            raise ValueError("Frame ID must be a nonnegative integer.")
        if self._last_frame is not None and frame_id <= self._last_frame:
            raise ValueError("Gate frames must be observed in increasing order.")
        self._last_frame = frame_id

        if (
            self._last_trigger_frame is not None
            and frame_id - self._last_trigger_frame > self._grace_frames
        ):
            self._last_trigger_frame = None

        if inference_ok and trigger:
            if self._last_trigger_frame is None:
                self._event_id += 1
            self._last_trigger_frame = frame_id

        return self._event_id if self._last_trigger_frame is not None else None
