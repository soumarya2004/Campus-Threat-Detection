"""Keep latency and GPU memory reports honest without requiring a CUDA machine."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_detector as benchmark


class FakeDetector:
    def __init__(self, device="cpu", fail=False):
        self.device = device
        self.fail = fail

    def get_model_info(self):
        return {"device_resolved": self.device}

    def warmup(self, **kwargs):
        pass

    def predict(self, frame, frame_id=None):
        return {
            "model_status": "inference_error" if self.fail else "ok",
            "error": "backend unavailable" if self.fail else None,
            "detections": [],
            "dropped_detections": 0,
        }


def test_cpu_timing_excludes_warmup_and_does_not_invent_gpu_memory(monkeypatch):
    ticks = iter([0, 0.1, 1, 1.01, 2, 2.03])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    report = benchmark.measure_inference(
        FakeDetector(),
        np.zeros((20, 30, 3), np.uint8),
        runs=2,
        warmup=1,
        torch_module=SimpleNamespace(),
    )
    assert report["warmup"]["elapsed_ms"] == pytest.approx(100)
    assert report["latency"]["p50_ms"] == pytest.approx(20)
    assert report["latency"]["p95_ms"] == pytest.approx(29)
    assert report["cuda_memory"] is None
    assert report["latency"]["synchronized_cuda"] is False


def test_cuda_measurement_synchronizes_and_separates_peak_from_model_memory():
    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda device: calls.append(("sync", device)),
        memory_allocated=lambda device: 100,
        memory_reserved=lambda device: 200,
        reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
        max_memory_allocated=lambda device: 180,
        max_memory_reserved=lambda device: 250,
        get_device_name=lambda device: "Test GPU",
    )
    report = benchmark.measure_inference(
        FakeDetector("1"),
        np.zeros((20, 30, 3), np.uint8),
        runs=2,
        warmup=1,
        torch_module=SimpleNamespace(cuda=cuda, device=lambda value: value),
    )
    assert calls == [
        ("sync", "cuda:1"),
        ("sync", "cuda:1"),
        ("reset", "cuda:1"),
        ("sync", "cuda:1"),
        ("sync", "cuda:1"),
        ("sync", "cuda:1"),
        ("sync", "cuda:1"),
    ]
    assert report["cuda_memory"]["peak_allocated_bytes"] == 180
    assert report["cuda_memory"]["additional_peak_allocated_bytes"] == 80


def test_failed_inference_cannot_produce_a_successful_benchmark():
    with pytest.raises(RuntimeError, match="backend unavailable"):
        benchmark.measure_inference(
            FakeDetector(fail=True),
            np.zeros((20, 30, 3), np.uint8),
            runs=2,
            warmup=1,
            torch_module=SimpleNamespace(),
        )
