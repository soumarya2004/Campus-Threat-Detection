"""Check local camera orchestration without accessing a physical camera."""

from argparse import Namespace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_camera_demo as demo


@pytest.fixture
def camera_case(monkeypatch, tmp_path):
    args = Namespace(
        camera=0,
        device="cpu",
        model=None,
        confidence=None,
        with_people=False,
        max_frames=2,
        no_display=True,
        report=None,
        overwrite=False,
    )
    state = SimpleNamespace(released=False, closed=[], calls=[], reads=0)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)

    def read():
        state.reads += 1
        return True, frame

    capture = SimpleNamespace(
        isOpened=lambda: True,
        read=read,
        release=lambda: setattr(state, "released", True),
    )

    def make_detector(name):
        checkpoint = tmp_path / f"{name}.pt"
        checkpoint.write_bytes(b"fake model")

        def predict(image, frame_id, timestamp):
            assert image is frame
            state.calls.append((name, frame_id, timestamp))
            return {"model_status": "ok", "detections": [], "error": None}

        return SimpleNamespace(
            warmup=lambda: None,
            predict=predict,
            close=lambda: state.closed.append(name),
            get_model_info=lambda: {
                "model_path": str(checkpoint),
                "device_resolved": "cpu",
            },
        )

    items, people = make_detector("items"), make_detector("people")
    monkeypatch.setattr(demo, "parse_args", lambda: args)
    monkeypatch.setattr(demo.cv2, "VideoCapture", lambda index: capture)
    monkeypatch.setattr(demo, "YOLOObjectDetector", lambda **kwargs: items)
    monkeypatch.setitem(
        sys.modules,
        "src.models.person_detector",
        SimpleNamespace(YOLOPersonDetector=lambda **kwargs: people),
    )
    return SimpleNamespace(
        args=args,
        state=state,
        capture=capture,
        items=items,
        people=people,
    )


def test_bounded_camera_run_keeps_detector_streams_separate(camera_case):
    camera_case.args.with_people = True
    assert demo.main() == 0
    state = camera_case.state
    assert state.reads == 2 and state.released
    assert state.closed == ["people", "items"]
    assert [(name, frame_id) for name, frame_id, _ in state.calls] == [
        ("items", 0),
        ("people", 0),
        ("items", 1),
        ("people", 1),
    ]
    assert state.calls[0][2] == state.calls[1][2]
    assert state.calls[2][2] == state.calls[3][2]


@pytest.mark.parametrize("failure", ["capture", "prediction", "interrupt"])
def test_camera_failure_and_interrupt_release_resources(camera_case, failure):
    def fail():
        raise KeyboardInterrupt

    if failure == "capture":
        camera_case.capture.read = lambda: (False, None)
    elif failure == "prediction":

        def bad_predict(*args, **kwargs):
            raise RuntimeError("decoder supplied an invalid frame")

        camera_case.items.predict = bad_predict
    else:
        camera_case.capture.read = fail
    assert demo.main() == (0 if failure == "interrupt" else 1)
    assert camera_case.state.released
    assert camera_case.state.closed == ["items"]


def test_inference_error_is_not_reported_as_success(camera_case):
    camera_case.items.predict = lambda *args, **kwargs: {
        "model_status": "inference_error",
        "error": "NMS failed",
        "detections": [],
    }
    assert demo.main() == 1
    assert camera_case.state.reads == 2
    assert camera_case.state.released


def test_quit_key_closes_only_the_demo_window(camera_case, monkeypatch):
    windows = []
    camera_case.args.no_display = False
    monkeypatch.setattr(demo.cv2, "imshow", lambda name, frame: windows.append(name))
    monkeypatch.setattr(demo.cv2, "waitKey", lambda delay: ord("q"))
    destroyed = []
    monkeypatch.setattr(demo.cv2, "destroyWindow", destroyed.append)
    assert demo.main() == 0
    assert camera_case.state.reads == 1
    assert camera_case.state.released
    assert destroyed == windows


def test_headless_camera_requires_explicit_bound(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo", "--no-display"])
    with pytest.raises(SystemExit) as error:
        demo.parse_args()
    assert error.value.code == 2


def test_camera_report_records_both_streams_and_finite_timing(camera_case, tmp_path):
    camera_case.args.with_people = True
    camera_case.args.report = tmp_path / "session.json"
    assert demo.main() == 0
    report = json.loads(camera_case.args.report.read_text())
    assert report["passed"]
    assert report["frames_processed"] == 2
    assert set(report["models"]) == {"items", "people"}
    assert report["timing"]["combined_predict_p95_ms"] >= 0
    assert report["camera"]["last_frame_dimensions"] == {"width": 60, "height": 40}
    assert report["media_saved"] is False


def test_failed_camera_session_still_saves_failure_report(camera_case, tmp_path):
    camera_case.args.report = tmp_path / "failed.json"
    camera_case.capture.read = lambda: (False, None)
    assert demo.main() == 1
    report = json.loads(camera_case.args.report.read_text())
    assert report["passed"] is False
    assert report["stop_reason"] == "error"
    assert report["frames_processed"] == 0
    assert report["error"]


def test_existing_report_is_preserved_before_camera_opens(camera_case, tmp_path):
    camera_case.args.report = tmp_path / "old.json"
    camera_case.args.report.write_text("previous evidence")
    assert demo.main() == 2
    assert camera_case.args.report.read_text() == "previous evidence"
    assert camera_case.state.reads == 0
