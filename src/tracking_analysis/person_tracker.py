"""Person-box detection and ByteTrack association."""

import numpy as np
from supervision import Detections
from trackers import ByteTrackTracker


PERSON_CONFIDENCE = 0.25
TRACK_ACTIVATION_THRESHOLD = 0.25
TRACK_MIN_CONSECUTIVE_FRAMES = 1


class PersonTracker:
    def __init__(self, weights=None, device="cpu"):
        self.detector = None
        if weights is not None:
            from ultralytics import YOLO

            self.detector = YOLO(str(weights))
        self.device = device
        self.tracker = None

    def reset(self):
        self.tracker = ByteTrackTracker(
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            minimum_consecutive_frames=TRACK_MIN_CONSECUTIVE_FRAMES,
        )

    def observe(self, frame, timestamp_s, replay_boxes=None):
        if replay_boxes is None:
            result = self.detector.predict(
                frame,
                classes=[0],
                conf=PERSON_CONFIDENCE,
                device=self.device,
                verbose=False,
            )[0]
            detections = Detections.from_ultralytics(result)
        else:
            if replay_boxes:
                detections = Detections(
                    xyxy=np.asarray(replay_boxes, dtype=float),
                    confidence=np.full(len(replay_boxes), 0.99, dtype=float),
                    class_id=np.zeros(len(replay_boxes), dtype=int),
                )
            else:
                detections = Detections.empty()

        tracked = self.tracker.update(detections, timestamp=timestamp_s)
        if not len(tracked) or tracked.tracker_id is None:
            return []

        best_by_id = {}
        for track_id, bbox, confidence in zip(
            tracked.tracker_id, tracked.xyxy, tracked.confidence, strict=True
        ):
            if track_id < 0:
                continue
            track_id = int(track_id)
            match = best_by_id.get(track_id)
            if match is None or confidence > match[0]:
                best_by_id[track_id] = (confidence, tuple(map(float, bbox)))
        return [
            (track_id, best_by_id[track_id][1])
            for track_id in sorted(best_by_id)
        ]
