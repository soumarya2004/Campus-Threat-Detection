import csv
import json

import cv2
import numpy as np
import pytest

from src.tracking_analysis import pipeline
from src.tracking_analysis.__main__ import main


@pytest.fixture
def video_and_replay(tmp_path):
    video = tmp_path / "short input.mp4"
    replay = tmp_path / "frame fixture.json"
    fps = 10
    frames = {
        str(index): {"people": [[12 + index, 24, 30 + index, 58], [60, 25, 78, 59]]}
        for index in range(12)
    }
    frames["0"]["trigger"] = True
    frames["1"]["inference_error"] = True
    frames["7"]["trigger"] = True
    replay.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    with pipeline.open_writer(video, fps, 96, 72) as writer:
        for _ in frames:
            writer.write(np.zeros((72, 96, 3), dtype=np.uint8))
    return video, replay


def read_table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_replay_gated_restarts_tracks_and_reconciles_counts(video_and_replay, tmp_path):
    video, replay = video_and_replay
    output = tmp_path / "gated run"
    summary = pipeline.run(video, output, "gated", replay=replay, grace=0.2)
    with (output / "summary.json").open(encoding="utf-8") as handle:
        assert json.load(handle) == summary
    assert summary["mode"] == "gated"
    assert summary["fps"] == 10
    assert summary["frame_count"] == 12
    assert summary["active_frames"] == summary["person_stage_calls"] == 6
    assert summary["item_stage_calls"] == 12
    assert summary["person_model_calls"] == summary["item_model_calls"] == 0
    assert summary["trigger_frames"] == summary["event_count"] == 2
    assert summary["inference_errors"] == 1
    assert len(summary["video_sha256"]) == len(summary["replay_sha256"]) == 64
    assert set(path.name for path in output.iterdir()) == {
        "frames.csv", "movement.csv", "pairs.csv", "proximity.csv", "annotated.mp4", "summary.json"
    }
    _, frames = read_table(output / "frames.csv")
    assert [row["event_id"] for row in frames] == [
        "1", "1", "1", "", "", "", "", "2", "2", "2", "", ""
    ]
    assert frames[1]["detector_status"] == "inference_error"
    _, motion = read_table(output / "movement.csv")
    ids_by_event = {
        event: {row["track_id"] for row in motion if row["event_id"] == event}
        for event in ("1", "2")
    }
    assert ids_by_event["1"] == ids_by_event["2"] == {"0", "1"}
    assert summary["track_count"] == 4
    assert summary["movement_rows"] == sum(int(row["person_count"]) for row in frames)
    _, pairs = read_table(output / "pairs.csv")
    _, proximities = read_table(output / "proximity.csv")
    assert len(pairs) == summary["pair_rows"] > 0
    assert len(proximities) == summary["proximity_episodes"] == 2
    pipeline.verify_video(output / "annotated.mp4", 12)


def test_continuous_replay_omits_item_stage(video_and_replay, tmp_path):
    video, replay = video_and_replay
    output = tmp_path / "continuous"
    assert main(
        ["run", "--video", str(video), "--output", str(output), "--mode", "continuous", "--replay", str(replay)]
    ) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["active_frames"] == summary["person_stage_calls"] == 12
    assert summary["event_count"] == 1
    assert summary["inference_errors"] == summary["item_stage_calls"] == 0
    _, frames = read_table(output / "frames.csv")
    assert all(row["detector_status"] == "not_run" and row["event_id"] == "1" for row in frames)


def test_no_trigger_still_produces_video_and_header_only_csvs(video_and_replay, tmp_path):
    video, replay = video_and_replay
    replay.write_text('{"frames": {}}', encoding="utf-8")
    output = tmp_path / "empty analytics"
    summary = pipeline.run(video, output, "gated", replay=replay)
    assert summary["event_count"] == summary["active_frames"] == 0
    assert summary["frame_count"] == 12
    for filename, expected_fields in (
        ("movement.csv", pipeline.MOVEMENT_FIELDS),
        ("pairs.csv", pipeline.PAIR_FIELDS),
        ("proximity.csv", pipeline.PROXIMITY_FIELDS),
    ):
        columns, rows = read_table(output / filename)
        assert columns == list(expected_fields)
        assert rows == []


def test_continuous_empty_people_keeps_writing_frames(video_and_replay, tmp_path):
    video, replay = video_and_replay
    replay.write_text('{"frames": {}}', encoding="utf-8")
    output = tmp_path / "active empty"
    summary = pipeline.run(video, output, "continuous", replay=replay)
    assert summary["frame_count"] == summary["active_frames"] == 12
    assert summary["movement_rows"] == 0


def test_missing_input_fails_cli_without_output(tmp_path, capsys):
    output = tmp_path / "missing input run"
    assert main(
        ["run", "--video", str(tmp_path / "missing.mp4"), "--output", str(output), "--mode", "continuous"]
    ) == 1
    captured = capsys.readouterr()
    assert "Video not found" in captured.err
    assert not captured.out
    assert not output.exists()


def test_existing_output_is_preserved(video_and_replay, tmp_path):
    video, replay = video_and_replay
    output = tmp_path / "prior run"
    output.mkdir()
    original = output / "summary.json"
    original.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run(video, output, "continuous", replay=replay)
    assert original.read_text(encoding="utf-8") == "original"


def test_bad_fps_releases_capture_and_does_not_publish(video_and_replay, tmp_path, monkeypatch):
    class ZeroFpsCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            assert prop == cv2.CAP_PROP_FPS
            return 0

        def release(self):
            self.released = True

    capture = ZeroFpsCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    video, replay = video_and_replay
    output = tmp_path / "zero fps"
    with pytest.raises(ValueError, match="invalid FPS"):
        pipeline.run(video, output, "continuous", replay=replay)
    assert capture.released
    assert not output.exists()
    assert not list(tmp_path.glob(".tracking-run-*"))


def test_bad_writer_releases_writer_and_does_not_publish(video_and_replay, tmp_path, monkeypatch):
    class BrokenWriter:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    writer = BrokenWriter()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    video, replay = video_and_replay
    output = tmp_path / "bad writer"
    with pytest.raises(ValueError, match="Cannot open MP4 writer"):
        pipeline.run(video, output, "continuous", replay=replay)
    assert writer.released
    assert not output.exists()
    assert not list(tmp_path.glob(".tracking-run-*"))


def test_bad_replay_frame_has_no_published_result(video_and_replay, tmp_path):
    video, replay = video_and_replay
    replay.write_text('{"frames": {"25": {}}}', encoding="utf-8")
    output = tmp_path / "bad replay"
    with pytest.raises(ValueError, match="outside the 12-frame video"):
        pipeline.run(video, output, "gated", replay=replay)
    assert not output.exists()
