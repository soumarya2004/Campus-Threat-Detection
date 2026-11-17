"""Verify prediction-independent sampling and fail-closed extraction bookkeeping."""

import json

import numpy as np
import pytest

from scripts import extract_evaluation_frames as extraction


class Capture:
    def __init__(self, count=10, fps=5.0, opened=True):
        self.count, self.fps, self.opened = count, fps, opened
        self.index = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {
            extraction.cv2.CAP_PROP_FPS: self.fps,
            extraction.cv2.CAP_PROP_FRAME_COUNT: 999,  # Incorrect metadata must not drive sampling.
            extraction.cv2.CAP_PROP_POS_MSEC: max(0, self.index - 1) * 200.0,
        }.get(prop, 0)

    def read(self):
        if self.index >= self.count:
            return False, None
        frame = np.full((8, 10, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def fake_video(tmp_path, monkeypatch, first=None, second=None):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video contents are mocked")
    captures = [first or Capture(), second or Capture()]
    iterator = iter(captures)
    monkeypatch.setattr(extraction.cv2, "VideoCapture", lambda path: next(iterator))
    return video, captures


def test_midpoints_use_decoded_count_and_cap_requested_samples():
    assert extraction.midpoint_indices(10, 3) == [1, 5, 8]
    assert extraction.midpoint_indices(3, 12) == [0, 1, 2]


@pytest.mark.parametrize(
    "count,samples", [(0, 12), (5, 0), (5, -1), (5, True), (5, 2.5)]
)
def test_invalid_sample_requests_fail(count, samples):
    with pytest.raises(ValueError):
        extraction.midpoint_indices(count, samples)


def test_extracts_fixed_frames_with_hashes_but_without_annotations(
    tmp_path, monkeypatch
):
    video, captures = fake_video(tmp_path, monkeypatch)
    output = tmp_path / "samples"
    manifest = extraction.extract_frames(video, "source_001", output, samples=3)
    assert manifest["decoded_frame_count"] == 10
    assert manifest["reported_frame_count"] == 999
    assert [row["frame_index_zero_based"] for row in manifest["frames"]] == [1, 5, 8]
    for row in manifest["frames"]:
        image = output / row["image"]
        assert row["image_sha256"] == extraction.file_sha256(image)
        assert (
            extraction.cv2.imread(str(image))[0, 0, 0] == row["frame_index_zero_based"]
        )
    assert json.loads((output / "source_manifest.json").read_text()) == manifest
    assert not list(output.rglob("*.txt"))
    assert all(capture.released for capture in captures)


@pytest.mark.parametrize(
    "source_id", ["../escape", "with space", "", "a/b", "a\\b", ".hidden"]
)
def test_invalid_source_id_does_not_create_output(tmp_path, source_id):
    output = tmp_path / "samples"
    with pytest.raises(ValueError, match="source-id"):
        extraction.extract_frames(tmp_path / "input.mp4", source_id, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "samples"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        extraction.extract_frames(video, "source", output)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("fps", [0, float("nan"), float("inf"), -1])
def test_bad_fps_releases_capture_and_creates_no_output(tmp_path, monkeypatch, fps):
    first = Capture(fps=fps)
    video, _ = fake_video(tmp_path, monkeypatch, first=first)
    output = tmp_path / "samples"
    with pytest.raises(ValueError, match="FPS"):
        extraction.extract_frames(video, "source", output)
    assert first.released and not output.exists()


def test_empty_video_creates_no_output(tmp_path, monkeypatch):
    first = Capture(count=0)
    video, _ = fake_video(tmp_path, monkeypatch, first=first)
    output = tmp_path / "samples"
    with pytest.raises(ValueError, match="decoded frame"):
        extraction.extract_frames(video, "source", output)
    assert first.released and not output.exists()


def test_changed_second_decode_does_not_write_success_manifest(tmp_path, monkeypatch):
    video, captures = fake_video(tmp_path, monkeypatch, second=Capture(count=8))
    output = tmp_path / "samples"
    with pytest.raises(ValueError, match="Decode changed"):
        extraction.extract_frames(video, "source", output, samples=3)
    assert not (output / "source_manifest.json").exists()
    assert all(capture.released for capture in captures)


def test_failed_image_write_releases_second_capture(tmp_path, monkeypatch):
    video, captures = fake_video(tmp_path, monkeypatch)
    monkeypatch.setattr(extraction.cv2, "imwrite", lambda *args: False)
    output = tmp_path / "samples"
    with pytest.raises(OSError, match="Could not save"):
        extraction.extract_frames(video, "source", output)
    assert all(capture.released for capture in captures)
    assert not (output / "source_manifest.json").exists()
