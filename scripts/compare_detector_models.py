"""Compare model and batch inference cost on one frozen validation sample.

This measures compute, not five-class accuracy: the general COCO checkpoints
do not share the fine-tuned model's complete label set. Decoding is excluded and
rect=False gives every configuration the same square input dimensions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import gc
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (
    file_sha256,
    load_split,
    repo_path,
    split_fingerprint,
    validate_report_paths,
)


def load_plan(path):
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("split") != "val":
        raise ValueError("This comparison requires the validation split.")
    for key in (
        "samples",
        "repetitions",
        "warmup_batches",
        "image_size",
        "cpu_threads",
    ):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(plan.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if plan["image_size"] < 32 or plan["image_size"] % 32:
        raise ValueError("image_size must be at least 32 and divisible by 32")
    for key in ("confidence", "nms_iou"):
        if type(plan.get(key)) not in (int, float) or not 0 <= plan[key] <= 1:
            raise ValueError(f"{key} must be a number in [0, 1]")
    if type(plan.get("max_detections")) is not int or plan["max_detections"] < 1:
        raise ValueError("max_detections must be a positive integer")
    batches = plan.get("batch_sizes")
    if (
        not isinstance(batches, list)
        or not batches
        or any(type(b) is not int or b < 1 or b > plan["samples"] for b in batches)
    ):
        raise ValueError(
            "batch_sizes must contain positive integers no larger than samples"
        )
    if len(set(batches)) != len(batches):
        raise ValueError("batch_sizes must be distinct")
    models = plan.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list")
    if len({m["name"] for m in models}) != len(models):
        raise ValueError("model names must be distinct")
    for model in models:
        checkpoint = repo_path(model["path"])
        if file_sha256(checkpoint) != model["sha256"]:
            raise ValueError(f"Checkpoint hash changed: {checkpoint}")
    return plan


def choose_sample(entries, count, seed):
    if count > len(entries):
        raise ValueError(f"Requested {count} images, only {len(entries)} available")
    # Stable ordering makes the sample reproducible even if directory enumeration changes.
    ordered = sorted(entries, key=lambda entry: entry["image"])
    return random.Random(seed).sample(ordered, count)


def measure_batches(predict, frames, batch_size, synchronize=lambda: None):
    batches = []
    for start in range(0, len(frames), batch_size):
        inputs = frames[start : start + batch_size]
        synchronize()
        started = time.perf_counter()
        results = predict(inputs)
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if len(results) != len(inputs):
            raise RuntimeError("Prediction result count does not match the input batch")
        batches.append({"images": len(inputs), "elapsed_ms": elapsed_ms})
    return batches


def summarize_batches(batches):
    images = sum(batch["images"] for batch in batches)
    total_ms = sum(batch["elapsed_ms"] for batch in batches)
    if not images or total_ms <= 0:
        raise ValueError("Cannot summarize an empty or zero-duration measurement")
    times = [batch["elapsed_ms"] for batch in batches]
    return {
        "images": images,
        "batches": len(batches),
        "elapsed_ms": total_ms,
        "amortized_ms_per_image": total_ms / images,
        "processing_images_per_second": 1000 * images / total_ms,
        "batch_latency_p50_ms": float(np.median(times)),
        "batch_latency_p95_ms": float(np.percentile(times, 95)),
    }


def save_plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(row["model"] for row in rows))
    sizes = sorted({row["batch_size"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in names:
        points = sorted(
            (row for row in rows if row["model"] == name),
            key=lambda row: row["batch_size"],
        )
        axes[0].plot(
            sizes,
            [r["summary"]["amortized_ms_per_image"] for r in points],
            marker="o",
            label=name,
        )
        axes[1].plot(
            sizes,
            [r["summary"]["batch_latency_p95_ms"] for r in points],
            marker="o",
            label=name,
        )
    for ax in axes:
        ax.set_xlabel("Inference batch size")
        ax.set_xticks(sizes)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Amortized milliseconds per image")
    axes[1].set_ylabel("Batch-completion p95 (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed 640-pixel validation sample: compute comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/comparison.yaml"))
    parser.add_argument("--device", default="cpu", help="cpu or one CUDA device index")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/comparison/model_batches.json")
    )
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path, output = repo_path(args.config), repo_path(args.output)
    plan = load_plan(config_path)
    data_path = repo_path(plan["data"])
    _, entries = load_split(data_path, "val")
    sample = choose_sample(entries, plan["samples"], plan["seed"])
    outputs = [output] + ([repo_path(args.plot)] if args.plot else [])
    inputs = [config_path, data_path] + [repo_path(m["path"]) for m in plan["models"]]
    inputs += [item[key] for item in entries for key in ("image_path", "label_path")]
    validate_report_paths(outputs, inputs, overwrite=args.overwrite)

    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(plan["cpu_threads"])
    device = torch.device(
        f"cuda:{args.device}" if args.device.isdigit() else args.device
    )
    if device.type not in {"cpu", "cuda"}:
        parser.error("--device must be cpu or one CUDA device index")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no GPU measurement can be made")
    synchronize = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else (lambda: None)
    )
    frames = [cv2.imread(str(item["image_path"])) for item in sample]
    if any(frame is None for frame in frames):
        raise ValueError("A sampled image could not be decoded")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Validation-image model-size/batch compute comparison; not a weapon accuracy comparison or a camera FPS claim.",
        "plan": plan,
        "plan_sha256": file_sha256(config_path),
        "sample": split_fingerprint(data_path, sample),
        "sample_manifest": [
            {
                "image": item["image"],
                "image_sha256": file_sha256(item["image_path"]),
                "label_sha256": file_sha256(item["label_path"]),
            }
            for item in sample
        ],
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "device": str(device),
            "cpu_threads": torch.get_num_threads(),
            "processor": platform.processor(),
        },
        "timing_scope": "Completed predict calls including preprocess, forward pass and postprocess; decoded images already in RAM; rect=False gives square inputs; imports, loading, warm-up, decoding, plotting and serialization excluded.",
        "models": [],
        "trials": [],
        "results": [],
        "limitations": [
            "General pretrained models have COCO classes; predictions are not scored as the five-class project task.",
            "Inference batching is not a training-batch convergence experiment.",
            "Batched throughput requires queued frames and cannot be interpreted as single-camera latency.",
            "CUDA memory is null for CPU runs; PyTorch CUDA memory excludes driver/context and other processes.",
        ],
    }
    for spec in plan["models"]:
        model = YOLO(str(repo_path(spec["path"])))
        if (
            len(model.names) != spec["class_count"]
            or model.names[0] != spec["class_zero"]
        ):
            raise ValueError(f"Unexpected class mapping for {spec['name']}")
        report["models"].append(
            {
                "name": spec["name"],
                "checkpoint": spec["path"],
                "sha256": spec["sha256"],
                "checkpoint_bytes": repo_path(spec["path"]).stat().st_size,
                "parameters": sum(p.numel() for p in model.model.parameters()),
                "class_names": model.names,
            }
        )
        del model
    configurations = [
        (spec["name"], batch)
        for spec in plan["models"]
        for batch in plan["batch_sizes"]
    ]
    for repetition in range(plan["repetitions"]):
        order = configurations.copy()
        random.Random(plan["seed"] + repetition).shuffle(order)
        for name, batch_size in order:
            spec = next(spec for spec in plan["models"] if spec["name"] == name)
            load_started = time.perf_counter()
            model = YOLO(str(repo_path(spec["path"])))
            model_load_ms = (time.perf_counter() - load_started) * 1000

            predict = partial(
                model.predict,
                imgsz=plan["image_size"],
                rect=False,
                device=args.device,
                conf=plan["confidence"],
                iou=plan["nms_iou"],
                max_det=plan["max_detections"],
                augment=False,
                verbose=False,
            )

            warmup_started = time.perf_counter()
            for _ in range(plan["warmup_batches"]):
                predict(frames[:batch_size])
            synchronize()
            warmup_ms = (time.perf_counter() - warmup_started) * 1000
            memory = None
            if device.type == "cuda":
                baseline_allocated = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            batches = measure_batches(predict, frames, batch_size, synchronize)
            if device.type == "cuda":
                memory = {
                    "baseline_allocated_bytes": baseline_allocated,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
            summary = summarize_batches(batches)
            report["trials"].append(
                {
                    "repetition": repetition + 1,
                    "model": name,
                    "batch_size": batch_size,
                    "model_load_ms": model_load_ms,
                    "warmup_ms": warmup_ms,
                    "summary": summary,
                    "cuda_memory": memory,
                    "batch_samples": batches,
                }
            )
            print(
                f"Repeat {repetition + 1}: {name}, batch {batch_size}: {summary['amortized_ms_per_image']:.2f} ms/image",
                flush=True,
            )
            del predict, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for name, batch_size in configurations:
        trials = [
            trial
            for trial in report["trials"]
            if trial["model"] == name and trial["batch_size"] == batch_size
        ]
        report["results"].append(
            {
                "model": name,
                "batch_size": batch_size,
                "summary": summarize_batches(
                    [batch for trial in trials for batch in trial["batch_samples"]]
                ),
                "repeat_amortized_ms_per_image": [
                    trial["summary"]["amortized_ms_per_image"] for trial in trials
                ],
            }
        )
    gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.plot:
        save_plot(report["results"], repo_path(args.plot))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
