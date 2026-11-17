import pytest
import yaml

from scripts import compare_detector_models as comparison


def test_sample_uses_stable_names_and_does_not_change_global_random_state():
    entries = [{"image": str(index)} for index in range(20)]
    state = comparison.random.getstate()
    first = comparison.choose_sample(entries, 8, 42)
    assert first == comparison.choose_sample(list(reversed(entries)), 8, 42)
    assert len({item["image"] for item in first}) == 8
    assert first != comparison.choose_sample(entries, 8, 43)
    assert comparison.random.getstate() == state


def test_sample_refuses_unavailable_image_count():
    with pytest.raises(ValueError, match="only 1 available"):
        comparison.choose_sample([{"image": "example.jpg"}], 2, 42)


def test_batch_measurement_preserves_tail_and_synchronizes(monkeypatch):
    times = iter([1.0, 1.1, 2.0, 2.2])
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: next(times))
    calls, synchronization = [], []

    def predict(frames):
        calls.append(frames)
        return [None] * len(frames)

    result = comparison.measure_batches(
        predict, [1, 2, 3], 2, lambda: synchronization.append(True)
    )
    assert calls == [[1, 2], [3]]
    assert len(synchronization) == 4
    assert [batch["images"] for batch in result] == [2, 1]
    summary = comparison.summarize_batches(result)
    assert summary["amortized_ms_per_image"] == pytest.approx(100)
    assert summary["processing_images_per_second"] == pytest.approx(10)


def test_measurement_rejects_dropped_results():
    with pytest.raises(RuntimeError, match="result count"):
        comparison.measure_batches(lambda frames: [], [1, 2], 2)


@pytest.fixture
def plan():
    return {
        "split": "val",
        "samples": 8,
        "repetitions": 3,
        "warmup_batches": 2,
        "image_size": 640,
        "cpu_threads": 4,
        "seed": 42,
        "batch_sizes": [1, 4, 8],
        "confidence": 0.25,
        "nms_iou": 0.7,
        "max_detections": 300,
        "models": [{"name": "model", "path": "fake.pt", "sha256": "expected"}],
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("split", "test"),
        ("batch_sizes", [1, 1]),
        ("batch_sizes", [0]),
        ("samples", True),
        ("confidence", float("nan")),
        ("image_size", 641),
    ],
)
def test_invalid_experiment_is_rejected(tmp_path, monkeypatch, plan, key, value):
    plan[key] = value
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(comparison, "file_sha256", lambda path: "expected")
    with pytest.raises(ValueError):
        comparison.load_plan(path)


def test_checkpoint_hash_must_match_plan(tmp_path, monkeypatch, plan):
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(comparison, "file_sha256", lambda path: "changed")
    with pytest.raises(ValueError, match="hash changed"):
        comparison.load_plan(path)


def test_valid_plan(tmp_path, monkeypatch, plan):
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(comparison, "file_sha256", lambda path: "expected")
    assert comparison.load_plan(path) == plan
