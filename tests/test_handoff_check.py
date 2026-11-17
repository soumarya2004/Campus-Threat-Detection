"""Check the downstream consumer contract without loading a model."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.check_detector_handoff import consume_payload
from scripts import check_detector_handoff as handoff

FRAME = np.zeros((120, 200, 3), dtype=np.uint8)
NAMES = {1: "knife"}
PAYLOAD = {
    "frame_id": 3,
    "timestamp": 0.12,
    "detections": [
        {
            "bbox": [10.0, 20.0, 100.0, 110.0],
            "class_id": 1,
            "class_name": "knife",
            "confidence": 0.8,
        }
    ],
    "inference_time_ms": 12.0,
    "model_status": "ok",
    "error": None,
    "dropped_detections": 0,
}


def test_consumer_accepts_original_pixel_coordinates_and_preserved_metadata():
    assert consume_payload(PAYLOAD, FRAME, 3, 0.12, NAMES) == 1
    empty = {**PAYLOAD, "detections": []}
    assert consume_payload(empty, FRAME, 3, 0.12, NAMES) == 0


@pytest.mark.parametrize(
    "change, expected",
    [
        ("wrong_id", "frame ID"),
        ("wrong_timestamp", "timestamp"),
        ("resized_coordinates", "original frame"),
        ("wrong_class_name", "class ID/name"),
        ("nonfinite_confidence", "valid JSON"),
        ("error_status", "Detector status"),
    ],
)
def test_consumer_rejects_unusable_or_failed_payloads(change, expected):
    payload = deepcopy(PAYLOAD)
    if change == "wrong_id":
        payload["frame_id"] = 4
    elif change == "wrong_timestamp":
        payload["timestamp"] = 0.2
    elif change == "resized_coordinates":
        payload["detections"][0]["bbox"] = [10, 20, 640, 640]
    elif change == "wrong_class_name":
        payload["detections"][0]["class_name"] = "gun"
    elif change == "nonfinite_confidence":
        payload["detections"][0]["confidence"] = float("nan")
    else:
        payload.update(
            model_status="inference_error", detections=[], error="NMS failed"
        )
    with pytest.raises(ValueError, match=expected):
        consume_payload(payload, FRAME, 3, 0.12, NAMES)


def test_handoff_failure_returns_nonzero_and_saves_actual_payload(
    monkeypatch, tmp_path
):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"decoded by the fake capture")
    output = tmp_path / "handoff.json"
    cleanup = []
    capture = SimpleNamespace(
        isOpened=lambda: True,
        read=lambda: (True, FRAME),
        get=lambda prop: 0.0,
        release=lambda: cleanup.append("capture"),
    )
    failure = {
        **PAYLOAD,
        "frame_id": 0,
        "timestamp": 0.0,
        "model_status": "inference_error",
        "detections": [],
        "error": "NMS failed",
    }

    def predict(frame, frame_id, timestamp):
        assert (
            frame is FRAME
        )  # No resize or color conversion before the public wrapper.
        assert frame_id == 0 and timestamp == 0.0
        return failure

    detector = SimpleNamespace(
        get_model_info=lambda: {"class_names": NAMES},
        predict=predict,
        close=lambda: cleanup.append("detector"),
    )
    monkeypatch.setattr(handoff.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(handoff, "YOLOObjectDetector", lambda **kwargs: detector)
    monkeypatch.setattr(
        handoff.sys,
        "argv",
        [
            "handoff",
            "--video",
            str(source),
            "--max-frames",
            "1",
            "--output",
            str(output),
        ],
    )
    assert handoff.main() == 1
    report = json.loads(output.read_text())
    assert not report["passed"] and report["checks"]["failed_frames"] == 1
    assert report["sample_payloads"] == [failure]
    assert "NMS failed" in report["errors"][0]["error"]
    assert cleanup == ["capture", "detector"]
