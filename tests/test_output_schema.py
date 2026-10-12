"""Interface tests for YOLOObjectDetector using a fake WeaponDetector.

No checkpoint is loaded and no GPU is used.
"""
import json

import numpy as np
import pytest

import src.models.object_detector as od


class FakeWeaponDetector:
    """Stands in for WeaponDetector. Behaviour is set through class attributes."""

    output: list = []
    error = None        # exception raised by detect()
    init_error = None   # exception raised by __init__
    calls: list = []    # frames passed to detect()

    def __init__(self, **kwargs):
        if FakeWeaponDetector.init_error:
            raise FakeWeaponDetector.init_error
        self.kwargs = kwargs
        self.device = "cpu"
        self.class_names = {0: "machete", 1: "knife"}

    def detect(self, frame):
        FakeWeaponDetector.calls.append(frame)
        if FakeWeaponDetector.error:
            raise FakeWeaponDetector.error
        return FakeWeaponDetector.output


@pytest.fixture
def detector(monkeypatch, tmp_path):
    FakeWeaponDetector.output = []
    FakeWeaponDetector.error = None
    FakeWeaponDetector.init_error = None
    FakeWeaponDetector.calls = []
    monkeypatch.setattr(od, "WeaponDetector", FakeWeaponDetector)
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"not a real checkpoint")
    return od.YOLOObjectDetector(model_path=weights)


FRAME = np.zeros((100, 200, 3), dtype=np.uint8)  # height 100, width 200

SCHEMA_KEYS = {
    "frame_id", "timestamp", "detections", "inference_time_ms",
    "model_status", "error", "dropped_detections",
}


def make_det(xyxy, confidence=0.8, class_id=1, class_name="knife"):
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "xyxy": xyxy,
    }


# ---------- valid and empty detections ----------

def test_valid_detection(detector):
    FakeWeaponDetector.output = [make_det([10.0, 20.0, 60.0, 80.0], 0.639)]
    r = detector.predict(FRAME, frame_id=3, timestamp=1.25)
    assert set(r) == SCHEMA_KEYS
    assert r["model_status"] == "ok"
    assert r["error"] is None
    assert r["dropped_detections"] == 0
    assert r["inference_time_ms"] >= 0.0
    assert r["detections"] == [{
        "bbox": [10.0, 20.0, 60.0, 80.0],
        "class_id": 1,
        "class_name": "knife",
        "confidence": 0.639,
    }]
    assert "xyxy" not in r["detections"][0]


def test_result_is_json_serialisable(detector):
    FakeWeaponDetector.output = [make_det([1, 2, 30, 40])]
    r = detector.predict(FRAME, frame_id=0, timestamp=0.0)
    assert json.loads(json.dumps(r))["detections"][0]["class_name"] == "knife"


def test_multiple_detections_keep_order(detector):
    FakeWeaponDetector.output = [
        make_det([1, 1, 20, 20], 0.9, 0, "machete"),
        make_det([30, 30, 60, 60], 0.5, 1, "knife"),
    ]
    r = detector.predict(FRAME)
    assert [d["class_name"] for d in r["detections"]] == ["machete", "knife"]


def test_empty_detections_return_full_schema(detector):
    r = detector.predict(FRAME, frame_id=1, timestamp=1.5)
    assert set(r) == SCHEMA_KEYS
    assert r["detections"] == []
    assert r["model_status"] == "ok"
    assert r["error"] is None
    assert r["dropped_detections"] == 0


# ---------- invalid frames ----------

@pytest.mark.parametrize(
    "bad_frame",
    [
        None,
        [[0, 0, 0]],
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 1), dtype=np.uint8),
        np.zeros((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 0, 3), dtype=np.uint8),
    ],
    ids=[
        "none", "list", "grayscale-2d", "float32", "four-channel",
        "one-channel", "zero-height", "zero-width",
    ],
)
def test_invalid_frames_raise(detector, bad_frame):
    with pytest.raises(od.FrameValidationError):
        detector.predict(bad_frame)
    assert FakeWeaponDetector.calls == []  # the model was never called


# ---------- bounding-box clipping and ordering ----------

def test_bbox_is_clipped_and_ordered(detector):
    # Reversed x and y, partly outside the 200x100 frame.
    FakeWeaponDetector.output = [make_det([-5.0, 150.0, 300.0, 20.0])]
    r = detector.predict(FRAME)
    assert r["detections"][0]["bbox"] == [0.0, 20.0, 200.0, 100.0]


def test_bbox_fully_outside_frame_is_dropped(detector):
    FakeWeaponDetector.output = [make_det([250.0, 10.0, 300.0, 50.0])]
    r = detector.predict(FRAME)
    assert r["detections"] == []
    assert r["dropped_detections"] == 1


# ---------- malformed detections ----------

BAD_DETECTIONS = [
    make_det([10, 10, 10, 50]),                          # zero width
    make_det([10, 10, 50, 10]),                          # zero height
    make_det([1, 1, 5, 5], confidence=1.5),              # confidence above 1
    make_det([1, 1, 5, 5], confidence=-0.1),             # confidence below 0
    make_det([float("nan"), 1, 5, 5]),                   # NaN coordinate
    {"class_id": 1, "class_name": "knife", "xyxy": [1, 1, 5, 5]},   # no confidence
    make_det(["a", "b", "c", "d"]),                      # non-numeric coordinates
    make_det([1, 2, 3]),                                 # wrong coordinate count
]
BAD_IDS = [
    "zero-width", "zero-height", "conf-above-1", "conf-below-0",
    "nan-coordinate", "missing-confidence", "non-numeric", "three-coordinates",
]


@pytest.mark.parametrize("bad", BAD_DETECTIONS, ids=BAD_IDS)
def test_malformed_detection_is_dropped(detector, bad):
    FakeWeaponDetector.output = [bad]
    r = detector.predict(FRAME)
    assert r["detections"] == []
    assert r["dropped_detections"] == 1
    assert r["model_status"] == "ok"


def test_good_detection_survives_malformed_neighbours(detector):
    FakeWeaponDetector.output = [
        make_det([10, 10, 10, 50]),
        make_det([5, 5, 50, 50], 0.7),
        make_det([1, 1, 5, 5], confidence=2.0),
    ]
    r = detector.predict(FRAME)
    assert len(r["detections"]) == 1
    assert r["detections"][0]["confidence"] == 0.7
    assert r["dropped_detections"] == 2


# ---------- frame ID and timestamp ----------

def test_frame_id_and_timestamp_preserved(detector):
    r = detector.predict(FRAME, frame_id=42, timestamp=123.456)
    assert r["frame_id"] == 42 and r["timestamp"] == 123.456
    r = detector.predict(FRAME, frame_id=0, timestamp=0.0)
    assert r["frame_id"] == 0 and r["timestamp"] == 0.0   # falsy values kept
    r = detector.predict(FRAME)
    assert r["frame_id"] is None and r["timestamp"] is None  # never invented


# ---------- inference errors ----------

def test_inference_error_is_contained(detector):
    FakeWeaponDetector.error = RuntimeError("boom")
    r = detector.predict(FRAME, frame_id=7, timestamp=9.0)
    assert set(r) == SCHEMA_KEYS
    assert r["model_status"] == "inference_error"
    assert r["detections"] == []
    assert "RuntimeError" in r["error"] and "boom" in r["error"]
    assert r["frame_id"] == 7 and r["timestamp"] == 9.0


def test_recovers_after_inference_error(detector):
    FakeWeaponDetector.error = RuntimeError("boom")
    assert detector.predict(FRAME)["model_status"] == "inference_error"
    FakeWeaponDetector.error = None
    FakeWeaponDetector.output = [make_det([1, 1, 20, 20])]
    r = detector.predict(FRAME)
    assert r["model_status"] == "ok" and len(r["detections"]) == 1


# ---------- model loading ----------

def test_missing_model_file_raises(tmp_path):
    with pytest.raises(od.ModelLoadError, match="not found"):
        od.YOLOObjectDetector(model_path=tmp_path / "missing.pt")


def test_loader_exception_is_wrapped(monkeypatch, tmp_path):
    FakeWeaponDetector.init_error = RuntimeError("corrupt checkpoint")
    monkeypatch.setattr(od, "WeaponDetector", FakeWeaponDetector)
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    with pytest.raises(od.ModelLoadError, match="corrupt checkpoint"):
        od.YOLOObjectDetector(model_path=weights)
    FakeWeaponDetector.init_error = None


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1])
def test_invalid_confidence_threshold_rejected(bad_threshold):
    with pytest.raises(ValueError):
        od.YOLOObjectDetector(confidence_threshold=bad_threshold)


# ---------- repeated calls, info, cleanup, warm-up ----------

def test_repeated_calls(detector):
    for i in range(5):
        r = detector.predict(FRAME, frame_id=i, timestamp=float(i))
        assert r["frame_id"] == i and r["timestamp"] == float(i)
    assert len(FakeWeaponDetector.calls) == 5


def test_get_model_info(detector):
    info = detector.get_model_info()
    assert info["model_status"] == "ready"
    assert info["class_names"] == {0: "machete", 1: "knife"}
    assert info["device_resolved"] == "cpu"
    assert info["confidence_threshold"] == 0.35


def test_close_then_predict_raises(detector):
    detector.close()
    assert detector.get_model_info()["model_status"] == "closed"
    with pytest.raises(od.ModelLoadError):
        detector.predict(FRAME)


def test_warmup_runs_detector(detector):
    detector.warmup(runs=3)
    assert len(FakeWeaponDetector.calls) == 3
    assert FakeWeaponDetector.calls[0].shape == (640, 640, 3)


# ---------- class-id and class-name validation ----------

@pytest.mark.parametrize(
    "bad_id",
    [-1, 2, 99, 1.0, True, "1", None],
    ids=["negative", "unknown-2", "unknown-99", "float", "bool", "string", "none"],
)
def test_invalid_class_id_is_dropped(detector, bad_id):
    FakeWeaponDetector.output = [make_det([5, 5, 50, 50], class_id=bad_id)]
    r = detector.predict(FRAME)
    assert r["detections"] == []
    assert r["dropped_detections"] == 1
    assert r["model_status"] == "ok"


def test_class_name_mismatch_is_dropped_not_rewritten(detector):
    FakeWeaponDetector.output = [
        make_det([5, 5, 50, 50], class_id=1, class_name="machete")
    ]
    r = detector.predict(FRAME)
    assert r["detections"] == []
    assert r["dropped_detections"] == 1


def test_numpy_integer_class_id_is_accepted_as_plain_int(detector):
    FakeWeaponDetector.output = [make_det([5, 5, 50, 50], class_id=np.int64(1))]
    r = detector.predict(FRAME)
    assert len(r["detections"]) == 1
    assert type(r["detections"][0]["class_id"]) is int
    json.dumps(r)


def test_list_class_names_are_converted(monkeypatch, tmp_path):
    class ListNames(FakeWeaponDetector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.class_names = ["machete", "knife"]

    FakeWeaponDetector.init_error = None
    monkeypatch.setattr(od, "WeaponDetector", ListNames)
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    d = od.YOLOObjectDetector(model_path=weights)
    assert d.get_model_info()["class_names"] == {0: "machete", 1: "knife"}


def test_empty_class_mapping_fails_load(monkeypatch, tmp_path):
    class NoClasses(FakeWeaponDetector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.class_names = {}

    FakeWeaponDetector.init_error = None
    monkeypatch.setattr(od, "WeaponDetector", NoClasses)
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    with pytest.raises(od.ModelLoadError, match="class mapping"):
        od.YOLOObjectDetector(model_path=weights)


# ---------- warmup validation ----------

@pytest.mark.parametrize(
    "runs",
    [0, -1, 1.5, True, "2", None],
    ids=["zero", "negative", "float", "bool", "string", "none"],
)
def test_warmup_rejects_bad_runs(detector, runs):
    with pytest.raises(ValueError, match="runs"):
        detector.warmup(runs=runs)
    assert FakeWeaponDetector.calls == []


@pytest.mark.parametrize(
    "size",
    [(0, 640), (640, 0), (-1, 640), (640,), (640, 640, 3), "640", 640,
     (640.0, 640), (True, 640), None],
    ids=["zero-h", "zero-w", "negative", "one-value", "three-values", "string",
         "int", "float", "bool", "none"],
)
def test_warmup_rejects_bad_size(detector, size):
    with pytest.raises(ValueError, match="size"):
        detector.warmup(size=size)
    assert FakeWeaponDetector.calls == []


def test_warmup_size_is_height_then_width(detector):
    detector.warmup(runs=2, size=(480, 640))
    assert len(FakeWeaponDetector.calls) == 2
    assert FakeWeaponDetector.calls[0].shape == (480, 640, 3)


def test_warmup_after_close_raises(detector):
    detector.close()
    with pytest.raises(od.ModelLoadError):
        detector.warmup()


# ---------- close() and load_model() behaviour ----------

def test_close_is_idempotent(detector):
    detector.close()
    detector.close()
    info = detector.get_model_info()
    assert info["model_status"] == "closed"
    assert info["class_names"] == {}


def test_load_model_is_a_noop_when_already_loaded(detector, monkeypatch):
    created = []

    class Counting(FakeWeaponDetector):
        def __init__(self, **kwargs):
            created.append(1)
            super().__init__(**kwargs)

    monkeypatch.setattr(od, "WeaponDetector", Counting)
    detector.load_model()
    assert created == []
    assert detector.get_model_info()["model_status"] == "ready"


def test_load_model_after_close_reloads(detector, monkeypatch):
    created = []

    class Counting(FakeWeaponDetector):
        def __init__(self, **kwargs):
            created.append(1)
            super().__init__(**kwargs)

    monkeypatch.setattr(od, "WeaponDetector", Counting)
    detector.close()
    detector.load_model()
    assert len(created) == 1
    info = detector.get_model_info()
    assert info["model_status"] == "ready"
    assert info["class_names"] == {0: "machete", 1: "knife"}
    assert detector.predict(FRAME)["model_status"] == "ok"


def test_reload_after_close_fails_cleanly_if_file_deleted(detector):
    detector.close()
    detector.model_path.unlink()
    with pytest.raises(od.ModelLoadError, match="not found"):
        detector.load_model()
    assert detector.get_model_info()["model_status"] == "load_failed"
    with pytest.raises(od.ModelLoadError):
        detector.predict(FRAME)
