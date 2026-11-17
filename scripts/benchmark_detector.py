"""Measure the selected detector's latency and CUDA memory on a real local image.

The same decoded frame is repeated to isolate detector cost. This measures
neither accuracy nor camera-to-display throughput.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import file_sha256, repo_path, validate_report_paths
from src.model_config import DEFAULT_CONFIG, load_detector_config


def cuda_device(device, torch_module):
    """Translate the wrapper's Ultralytics device name to a PyTorch CUDA device."""
    value = str(device).lower()
    if value.isdigit():
        value = f"cuda:{value}"
    if value.startswith("cuda"):
        return torch_module.device(value)
    return None


def measure_inference(detector, frame, *, runs, warmup, torch_module):
    """Time completed wrapper predictions and report peak memory after warm-up."""
    if runs < 1 or warmup < 1:
        raise ValueError("runs and warmup must be positive integers")
    info = detector.get_model_info()
    device = cuda_device(info["device_resolved"], torch_module)

    def synchronize():
        if device is not None:
            torch_module.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter()
    detector.warmup(runs=warmup, size=frame.shape[:2])
    # Warm-up the actual scene too, so postprocessing follows the real workload.
    first = detector.predict(frame)
    if first["model_status"] != "ok":
        raise RuntimeError(f"Warm-up prediction failed: {first['error']}")
    synchronize()
    warmup_ms = (time.perf_counter() - started) * 1000
    memory = None
    if device is not None:
        baseline_allocated = torch_module.cuda.memory_allocated(device)
        baseline_reserved = torch_module.cuda.memory_reserved(device)
        torch_module.cuda.reset_peak_memory_stats(device)

    latencies = []
    detection_counts = []
    dropped = 0
    for index in range(runs):
        synchronize()
        started = time.perf_counter()
        result = detector.predict(frame, frame_id=index)
        synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        if result["model_status"] != "ok":
            raise RuntimeError(f"Prediction {index} failed: {result['error']}")
        detection_counts.append(len(result["detections"]))
        dropped += result["dropped_detections"]

    if device is not None:
        peak_allocated = torch_module.cuda.max_memory_allocated(device)
        memory = {
            "device": str(device),
            "device_name": torch_module.cuda.get_device_name(device),
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": torch_module.cuda.max_memory_reserved(device),
            "additional_peak_allocated_bytes": max(
                0, peak_allocated - baseline_allocated
            ),
            "interpretation": "PyTorch allocator memory after warm-up; includes model tensors, excludes CUDA driver/context and memory owned by other processes.",
        }
    return {
        "warmup": {
            "blank_frame_runs": warmup,
            "real_frame_runs": 1,
            "elapsed_ms": warmup_ms,
        },
        "latency": {
            "runs": runs,
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "synchronized_cuda": device is not None,
            "scope": "Completed YOLOObjectDetector.predict calls, including preprocessing, model inference, postprocessing and output normalization. Excludes decoding, drawing and display.",
        },
        "cuda_memory": memory,
        "detections_per_run": {
            "min": min(detection_counts),
            "max": max(detection_counts),
        },
        "dropped_detections": dropped,
        "latencies_ms": latencies,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Real local image used for every timed prediction.",
    )
    parser.add_argument(
        "--device", help="cpu or CUDA device index; defaults to automatic selection."
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/detector.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1:
        parser.error("--runs and --warmup must be positive")
    image_path, output = repo_path(args.image), repo_path(args.output)
    settings = load_detector_config()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    validate_report_paths(
        [output],
        [image_path, settings["model_path"], DEFAULT_CONFIG],
        overwrite=args.overwrite,
    )

    import cv2
    import torch
    import ultralytics
    from src.models.object_detector import YOLOObjectDetector

    started = time.perf_counter()
    frame = cv2.imread(str(image_path))
    decode_ms = (time.perf_counter() - started) * 1000
    if frame is None:
        parser.error(f"Could not decode image: {image_path}")
    detector = None
    try:
        started = time.perf_counter()
        detector = YOLOObjectDetector(device=args.device)
        info = detector.get_model_info()
        device = cuda_device(info["device_resolved"], torch)
        if device is not None:
            torch.cuda.synchronize(device)
        load_ms = (time.perf_counter() - started) * 1000
        measurements = measure_inference(
            detector, frame, runs=args.runs, warmup=args.warmup, torch_module=torch
        )
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Single-image repeated runtime benchmark; not an accuracy evaluation or video FPS claim.",
            "model": info,
            "checkpoint_sha256": file_sha256(Path(info["model_path"])),
            "image": {
                "path": str(image_path),
                "sha256": file_sha256(image_path),
                "height": frame.shape[0],
                "width": frame.shape[1],
            },
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "opencv": cv2.__version__,
            },
            "setup": {
                "image_decode_ms": decode_ms,
                "model_load_ms": load_ms,
                "scope": "Imports excluded; CUDA initialization/first prediction costs are included in warm-up when deferred by the backend.",
            },
            **measurements,
        }
    finally:
        if detector is not None:
            detector.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"p50: {report['latency']['p50_ms']:.2f} ms; p95: {report['latency']['p95_ms']:.2f} ms"
    )
    if report["cuda_memory"] is not None:
        print(
            f"Peak PyTorch allocation: {report['cuda_memory']['peak_allocated_bytes'] / 1024**2:.1f} MiB"
        )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
