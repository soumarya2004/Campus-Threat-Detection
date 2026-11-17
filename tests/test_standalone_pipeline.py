"""The standalone consumer keeps item and person class IDs in separate streams."""

from copy import deepcopy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import check_standalone_pipeline as pipeline

FRAME = np.zeros((120, 200, 3), dtype=np.uint8)


def payload(name, frame_id=0, timestamp=0.0):
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "detections": [
            {
                "bbox": [10.0, 20.0, 100.0, 110.0],
                "class_id": 0,
                "class_name": name,
                "confidence": 0.8,
            }
        ],
        "inference_time_ms": 10.0,
        "model_status": "ok",
        "error": None,
        "dropped_detections": 0,
    }


def test_class_zero_is_valid_for_both_namespaces_without_collision():
    packet = {"items": payload("machete"), "people": payload("person")}
    assert pipeline.consume_frame(packet, FRAME, 0, 0.0) == {"items": 1, "people": 1}
    packet["items"], packet["people"] = packet["people"], packet["items"]
    with pytest.raises(ValueError, match="class ID/name"):
        pipeline.consume_frame(packet, FRAME, 0, 0.0)


@pytest.mark.parametrize("field,value", [("frame_id", 1), ("timestamp", 0.5)])
def test_consumer_rejects_unsynchronized_streams(field, value):
    packet = {"items": payload("machete"), "people": payload("person")}
    packet["people"][field] = value
    with pytest.raises(ValueError, match="preserved"):
        pipeline.consume_frame(packet, FRAME, 0, 0.0)


def test_consumer_requires_both_streams():
    with pytest.raises(ValueError, match="separate"):
        pipeline.consume_frame({"items": payload("machete")}, FRAME, 0, 0.0)


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"test recording")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test model")
    calls = []
    state = {"frames": 0, "fail_person": False, "stalled_timestamp": False}

    def read():
        state["frames"] += 1
        return True, FRAME

    capture = SimpleNamespace(
        isOpened=lambda: True,
        read=read,
        get=lambda prop: (
            0.0 if state["stalled_timestamp"] else (state["frames"] - 1) * 40.0
        ),
        release=lambda: calls.append("capture closed"),
    )

    def factory(stream):
        def predict(frame, frame_id, timestamp):
            assert frame is FRAME
            calls.append((stream, frame_id, timestamp))
            result = payload(
                "machete" if stream == "items" else "person", frame_id, timestamp
            )
            if stream == "people" and state["fail_person"]:
                result.update(
                    detections=[], model_status="inference_error", error="model failed"
                )
            return result

        return SimpleNamespace(
            get_model_info=lambda: {
                "class_names": deepcopy(pipeline.STREAM_CLASSES[stream]),
                "model_path": str(checkpoint),
            },
            warmup=lambda runs: calls.append((stream, "warmup", runs)),
            predict=predict,
            close=lambda: calls.append(f"{stream} closed"),
        )

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(
        pipeline, "YOLOObjectDetector", lambda **kwargs: factory("items")
    )
    monkeypatch.setattr(
        pipeline, "YOLOPersonDetector", lambda **kwargs: factory("people")
    )
    return source, calls, state


def test_rehearsal_round_trips_both_streams_and_closes_resources(fake_runtime):
    source, calls, state = fake_runtime
    stream = io.StringIO()
    report = pipeline.rehearse(
        source, max_frames=2, device="cpu", payload_stream=stream
    )
    assert report["passed"]
    assert report["checks"]["item_boxes"] == report["checks"]["person_boxes"] == 2
    assert report["checks"]["passed_frames"] == 2
    packets = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [packet["items"]["frame_id"] for packet in packets] == [0, 1]
    assert packets[1]["people"]["timestamp"] == packets[1]["items"]["timestamp"] == 0.04
    assert calls[-3:] == ["capture closed", "items closed", "people closed"]


def test_one_failed_stream_fails_the_rehearsal_and_does_not_emit_success_packet(
    fake_runtime,
):
    source, calls, state = fake_runtime
    state["fail_person"] = True
    stream = io.StringIO()
    report = pipeline.rehearse(source, max_frames=1, payload_stream=stream)
    assert not report["passed"]
    assert report["checks"]["failed_frames"] == 1
    assert "model failed" in report["errors"][0]["error"]
    assert stream.getvalue() == ""
    assert calls[-3:] == ["capture closed", "items closed", "people closed"]


def test_stalled_timestamps_cannot_claim_synchronized_output(fake_runtime):
    source, calls, state = fake_runtime
    state["stalled_timestamp"] = True
    report = pipeline.rehearse(source, max_frames=2)
    assert not report["passed"]
    assert "timestamp" in report["errors"][0]["error"]
    assert report["checks"]["passed_frames"] == 1


def test_person_load_failure_releases_already_loaded_item_detector(
    fake_runtime, monkeypatch
):
    source, calls, state = fake_runtime

    def broken_person(**kwargs):
        raise RuntimeError("person weights missing")

    monkeypatch.setattr(pipeline, "YOLOPersonDetector", broken_person)
    report = pipeline.rehearse(source, max_frames=1)
    assert not report["passed"]
    assert "person weights missing" in report["errors"][0]["error"]
    assert calls[-2:] == ["capture closed", "items closed"]


def test_report_paths_preserve_inputs_and_existing_results(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"recording")
    output = tmp_path / "report.json"
    output.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        pipeline.check_output_paths([output], [source])
    with pytest.raises(ValueError, match="input"):
        pipeline.check_output_paths([source], [source], overwrite=True)
    with pytest.raises(ValueError, match="different"):
        pipeline.check_output_paths([output, output], [source], overwrite=True)


def test_hardlinked_input_cannot_be_overwritten(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"recording")
    output = tmp_path / "alias.json"
    output.hardlink_to(source)
    with pytest.raises(ValueError, match="aliases|experiment input"):
        pipeline.check_output_paths([output], [source], overwrite=True)
    assert source.read_bytes() == b"recording"
