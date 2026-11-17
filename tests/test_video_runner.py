"""Exercise video failures with fakes; no model or GPU is used."""

from argparse import Namespace
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from scripts import run_video_inference as runner
from scripts import smoke_test_wrapper as smoke
from src.utils import check_environment


class FakeCapture:
    def __init__(self, count=3, reported=3):
        self.frames = [np.zeros((40, 60, 3), dtype=np.uint8) for _ in range(count)]
        self.reported = reported
        self.opened = True
        self.released = False
        self.decoded = 0
        self.metadata = {}
        self.timestamps_ms = None

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop in self.metadata:
            return self.metadata[prop]
        if prop == runner.cv2.CAP_PROP_POS_MSEC:
            if self.timestamps_ms is not None:
                return self.timestamps_ms[self.decoded - 1]
            return (self.decoded - 1) * (1000 / 30)
        return {
            runner.cv2.CAP_PROP_FPS: 30,
            runner.cv2.CAP_PROP_FRAME_COUNT: self.reported,
            runner.cv2.CAP_PROP_FRAME_WIDTH: 60,
            runner.cv2.CAP_PROP_FRAME_HEIGHT: 40,
        }[prop]

    def read(self):
        if self.frames:
            self.decoded += 1
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.opened = True
        self.released = False
        self.error = None

    def isOpened(self):
        return self.opened

    def write(self, frame):
        if self.error:
            raise self.error
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self):
        self.closed = False
        self.fail_frames = set()
        self.warmup_error = None
        self.timestamps = []

    def warmup(self, **kwargs):
        if self.warmup_error:
            raise self.warmup_error

    def get_model_info(self):
        return {"device_resolved": "cpu", "confidence_threshold": 0.42}

    def predict(self, frame, frame_id=0, timestamp=None):
        self.timestamps.append(timestamp)
        failed = frame_id in self.fail_frames
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "model_status": "inference_error" if failed else "ok",
            "error": "simulated NMS failure" if failed else None,
            "detections": [],
            "dropped_detections": 0,
            "inference_time_ms": 1.0,
        }

    def close(self):
        self.closed = True


@pytest.fixture
def video_case(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake video, decoded by FakeCapture")
    args = Namespace(
        video=input_path,
        output=tmp_path / "output.mp4",
        model=None,
        device=None,
        confidence=None,
        max_frames=None,
        warmup_runs=0,
    )
    capture = FakeCapture()
    writer = FakeWriter()
    detector = FakeDetector()
    model_arguments = []

    def create_detector(**kwargs):
        model_arguments.append(kwargs)
        return detector

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(runner.cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(runner, "YOLOObjectDetector", create_detector)
    return SimpleNamespace(
        args=args,
        capture=capture,
        writer=writer,
        detector=detector,
        model_arguments=model_arguments,
    )


def read_report(case):
    return json.loads(case.args.output.with_suffix(".json").read_text())


def assert_released(case):
    assert case.capture.released
    assert case.writer.released
    assert case.detector.closed


def test_video_completion_preserves_report_and_config_defaults(video_case):
    assert runner.main() == 0
    report = read_report(video_case)
    assert report["results"]["processed_frames"] == 3
    assert report["results"]["stop_reason"] == "end_of_video"
    assert report["results"]["completion_warning"] is None
    assert report["model"]["confidence_threshold"] == 0.42
    assert video_case.model_arguments == [
        {
            "model_path": None,
            "confidence_threshold": None,
            "device": None,
        }
    ]
    assert len(video_case.writer.frames) == 3
    assert_released(video_case)


def test_cli_overrides_reach_detector(video_case):
    video_case.args.model = "custom.pt"
    video_case.args.confidence = 0.6
    video_case.args.device = "cpu"
    assert runner.main() == 0
    assert video_case.model_arguments[0] == {
        "model_path": "custom.pt",
        "confidence_threshold": 0.6,
        "device": "cpu",
    }


def test_early_decode_warns_without_assuming_metadata_is_exact(video_case, capsys):
    video_case.capture.reported = 145
    assert runner.main() == 0
    result = read_report(video_case)["results"]
    assert result["stop_reason"] == "early_decode_stop"
    assert "3 of approximately 145" in result["completion_warning"]
    assert "WARNING" in capsys.readouterr().err
    assert_released(video_case)


def test_small_metadata_difference_is_tolerated(video_case):
    video_case.capture.reported = 4
    assert runner.main() == 0
    assert read_report(video_case)["results"]["completion_warning"] is None


def test_frame_limit_is_intentional_not_decode_warning(video_case):
    video_case.args.max_frames = 1
    assert runner.main() == 0
    result = read_report(video_case)["results"]
    assert result["processed_frames"] == 1
    assert result["stop_reason"] == "max_frames"
    assert result["completion_warning"] is None


def test_inference_errors_keep_frame_and_message(video_case):
    video_case.detector.fail_frames = {1}
    assert runner.main() == 1
    result = read_report(video_case)["results"]
    assert result["frames_with_inference_errors"] == 1
    assert result["inference_error_samples"] == [
        {
            "frame_id": 1,
            "timestamp": 1 / 30,
            "error": "simulated NMS failure",
        }
    ]
    assert result["processed_frames"] == 3
    assert_released(video_case)


def test_error_samples_are_bounded(video_case):
    video_case.capture.frames *= 5
    video_case.capture.reported = 15
    video_case.detector.fail_frames = set(range(15))
    assert runner.main() == 1
    result = read_report(video_case)["results"]
    assert result["frames_with_inference_errors"] == 15
    assert len(result["inference_error_samples"]) == 10


@pytest.mark.parametrize(
    "failure", ["no_frames", "writer_open", "writer_write", "warmup"]
)
def test_failures_return_nonzero_and_release_resources(video_case, failure):
    if failure == "no_frames":
        video_case.capture.frames = []
    elif failure == "writer_open":
        video_case.writer.opened = False
    elif failure == "writer_write":
        video_case.writer.error = OSError("disk write failed")
    else:
        video_case.args.warmup_runs = 1
        video_case.detector.warmup_error = RuntimeError("warmup failed")
    assert runner.main() == 1
    assert video_case.capture.released
    assert video_case.detector.closed
    if failure != "warmup":
        assert video_case.writer.released
    report = read_report(video_case)
    assert report["status"] == "failed"
    assert report["results"]["stop_reason"] == "processing_error"


def test_variable_rate_video_preserves_decoder_timestamps(video_case):
    video_case.capture.timestamps_ms = [0.0, 40.0, 95.0]
    assert runner.main() == 0
    assert video_case.detector.timestamps == [0.0, 0.04, 0.095]


def test_unknown_fps_and_stalled_timestamps_are_not_invented(video_case):
    video_case.capture.metadata = {
        runner.cv2.CAP_PROP_FPS: float("nan"),
        runner.cv2.CAP_PROP_FRAME_COUNT: float("nan"),
    }
    video_case.capture.timestamps_ms = [0.0, 0.0, float("nan")]
    assert runner.main() == 0
    assert video_case.detector.timestamps == [0.0, None, None]
    report = read_report(video_case)
    assert report["source"]["fps"] is None
    assert report["source"]["output_fps"] == 30.0
    assert report["source"]["output_fps_is_fallback"]
    assert report["results"]["frames_without_media_timestamp"] == 2
    assert "NaN" not in video_case.args.output.with_suffix(".json").read_text()


def test_invalid_metadata_releases_capture_without_loading_model(video_case):
    video_case.capture.metadata[runner.cv2.CAP_PROP_FRAME_WIDTH] = float("nan")
    assert runner.main() == 2
    assert video_case.capture.released
    assert video_case.model_arguments == []


def test_wrong_frame_dimensions_fail_instead_of_silently_dropping_frame(video_case):
    video_case.capture.frames[1] = np.zeros((80, 120, 3), dtype=np.uint8)
    assert runner.main() == 1
    assert len(video_case.writer.frames) == 1
    assert "dimensions" in read_report(video_case)["error"]
    assert_released(video_case)


def test_report_cannot_overwrite_input_with_unusual_extension(video_case):
    original = b"video whose file extension is misleading"
    video_case.args.video = video_case.args.output.with_suffix(".json")
    video_case.args.video.write_bytes(original)
    assert runner.main() == 2
    assert video_case.args.video.read_bytes() == original
    assert video_case.model_arguments == []


def test_video_and_report_cannot_share_one_path(video_case):
    video_case.args.output = video_case.args.output.with_suffix(".json")
    assert runner.main() == 2
    assert video_case.model_arguments == []


def test_hardlinked_output_cannot_overwrite_input(video_case):
    video_case.args.output.hardlink_to(video_case.args.video)
    original = video_case.args.video.read_bytes()
    assert runner.main() == 2
    assert video_case.args.video.read_bytes() == original
    assert video_case.model_arguments == []


def test_failure_replaces_stale_success_report(video_case):
    report_path = video_case.args.output.with_suffix(".json")
    report_path.write_text('{"status": "completed"}')
    video_case.writer.error = OSError("disk full")
    assert runner.main() == 1
    assert read_report(video_case)["status"] == "failed"


def test_unopenable_input_does_not_load_model(video_case):
    video_case.capture.opened = False
    assert runner.main() == 2
    assert video_case.capture.released
    assert video_case.model_arguments == []


def test_model_load_failure_releases_capture(video_case, monkeypatch):
    def fail_load(**kwargs):
        raise RuntimeError("missing model")

    monkeypatch.setattr(runner, "YOLOObjectDetector", fail_load)
    assert runner.main() == 1
    assert video_case.capture.released


def test_second_smoke_call_failure_cannot_pass(monkeypatch, tmp_path, capsys):
    detector = FakeDetector()
    detector.fail_frames = {1}
    monkeypatch.setattr(
        sys, "argv", ["smoke_test_wrapper.py", str(tmp_path / "image.jpg")]
    )
    monkeypatch.setattr(
        smoke.cv2, "imread", lambda path: np.zeros((40, 60, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(smoke, "YOLOObjectDetector", lambda **kwargs: detector)
    assert smoke.main() == 1
    assert "second call failed" in capsys.readouterr().out
    assert detector.closed


@pytest.mark.parametrize("require_cuda, expected", [(False, 0), (True, 1)])
def test_cpu_environment_is_valid_unless_cuda_required(
    monkeypatch, require_cuda, expected
):
    fake_torch = SimpleNamespace(
        tensor=lambda value: value,
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "torchvision.ops", SimpleNamespace(nms=lambda *args: [0])
    )
    monkeypatch.setattr(check_environment.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(check_environment, "version", lambda name: "test")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_environment.py"] + (["--require-cuda"] if require_cuda else []),
    )
    assert check_environment.main() == expected
