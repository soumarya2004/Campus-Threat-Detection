"""The person stream uses its own class namespace and the common frame contract."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import src.detector as adapter
from src.models.object_detector import FrameValidationError, ModelLoadError
from src.models.person_detector import YOLOPersonDetector


FRAME = np.zeros((120, 200, 3), dtype=np.uint8)


@pytest.fixture
def person_model(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        names = {0: "person", 1: "bicycle", 2: "car"}

        def __init__(self, path):
            calls.append({"checkpoint": path})

        def predict(self, **kwargs):
            calls.append(kwargs)
            box = SimpleNamespace(
                cls=np.array(0),
                conf=np.array(0.8),
                xyxy=np.array([[10.0, 20.0, 90.0, 110.0]]),
            )
            return [SimpleNamespace(boxes=[box])]

    monkeypatch.setattr(adapter, "YOLO", FakeModel)
    weights = tmp_path / "general.pt"
    weights.write_bytes(b"fake")
    return SimpleNamespace(weights=weights, calls=calls, model=FakeModel)


def test_person_stream_filters_coco_and_keeps_frame_contract(person_model):
    with YOLOPersonDetector(model_path=person_model.weights, device="cpu") as detector:
        result = detector.predict(FRAME, frame_id=8, timestamp=0.4)
        assert result["model_status"] == "ok"
        assert result["frame_id"] == 8
        assert result["timestamp"] == 0.4
        assert result["detections"] == [
            {
                "bbox": [10.0, 20.0, 90.0, 110.0],
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.8,
            }
        ]
        assert detector.get_model_info()["class_names"] == {0: "person"}
        assert person_model.calls[-1]["classes"] == [0]
        assert person_model.calls[-1]["source"] is FRAME
        json.dumps(result, allow_nan=False)
    assert detector.get_model_info()["model_status"] == "closed"


def test_weapon_checkpoint_cannot_be_used_for_person_stream(person_model):
    person_model.model.names = {0: "machete", 1: "knife"}
    with pytest.raises(ModelLoadError, match="COCO class 0"):
        YOLOPersonDetector(model_path=person_model.weights, device="cpu")


def test_person_stream_requires_local_weights_before_loading(monkeypatch, tmp_path):
    def unexpected_load(*args, **kwargs):
        pytest.fail("A missing local checkpoint must not reach Ultralytics")

    monkeypatch.setattr(adapter, "YOLO", unexpected_load)
    with pytest.raises(ModelLoadError, match="not found"):
        YOLOPersonDetector(model_path=tmp_path / "missing.pt", device="cpu")


def test_person_defaults_do_not_depend_on_weapon_configuration(
    person_model, monkeypatch
):
    def unexpected_config():
        pytest.fail("The separate person stream must not read weapon defaults")

    import src.models.object_detector as objects

    monkeypatch.setattr(adapter, "load_detector_config", unexpected_config)
    monkeypatch.setattr(objects, "load_detector_config", unexpected_config)
    with YOLOPersonDetector(model_path=person_model.weights, device="cpu") as detector:
        assert detector.get_model_info()["confidence_threshold"] == 0.35
        assert detector.get_model_info()["image_size"] == 640


def test_person_stream_rejects_invalid_frames_before_prediction(person_model):
    with YOLOPersonDetector(model_path=person_model.weights, device="cpu") as detector:
        with pytest.raises(FrameValidationError):
            detector.predict(np.zeros((120, 200), dtype=np.uint8))
        assert len(person_model.calls) == 1  # checkpoint construction only


def test_person_stream_reports_inference_failures(person_model, monkeypatch):
    def fail_prediction(self, **kwargs):
        raise RuntimeError("inference unavailable")

    monkeypatch.setattr(person_model.model, "predict", fail_prediction)
    with YOLOPersonDetector(model_path=person_model.weights, device="cpu") as detector:
        result = detector.predict(FRAME, frame_id=0)
        assert result["model_status"] == "inference_error"
        assert result["detections"] == []
        assert "inference unavailable" in result["error"]


@pytest.mark.integration
def test_local_person_checkpoint_predicts_on_cpu():
    with YOLOPersonDetector(device="cpu") as detector:
        result = detector.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result["model_status"] == "ok", result["error"]
        assert detector.get_model_info()["class_names"] == {0: "person"}
        assert all(item["class_name"] == "person" for item in result["detections"])
