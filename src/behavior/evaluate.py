"""Evaluate a trained model on clip-level or source-video manifests."""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.behavior.data.dataset import VideoClipDataset, create_dataloader
from src.behavior.data.video_bag_dataset import VideoBagDataset, create_video_bag_dataloader
from src.behavior.models.temporal_model import build_temporal_model
from src.behavior.train import pool_video_bag_logits
from src.behavior.utils.device import setup_device
from src.behavior.utils.metrics import (
    evaluate_predictions,
    find_optimal_threshold,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EvaluateViolenceModel")


def run_full_evaluation(
    model: nn.Module,
    dataset: VideoClipDataset,
    device: torch.device,
    default_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates model across all test samples, categorizing errors by behavior type.
    """
    model.eval()
    loader = create_dataloader(
        dataset, batch_size=4, is_training=False, balance_classes=False, num_workers=0
    )

    all_targets = []
    all_probs = []
    all_categories = []
    all_video_ids = []
    all_latencies = []

    logger.info("Running evaluation over %d test clips...", len(dataset))
    with torch.no_grad():
        for clips, labels, meta in loader:
            clips = clips.to(device)
            t0 = time.perf_counter()
            logits = model(clips)
            latency = (time.perf_counter() - t0) * 1000.0 / clips.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = labels.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets)
            all_categories.extend(meta["category"])
            all_video_ids.extend(meta["source_video_id"])
            all_latencies.extend([latency] * len(probs))

    y_true = np.array(all_targets).astype(int)
    y_prob = np.array(all_probs).astype(float)

    # 1. Standard Metrics at default threshold
    default_metrics = evaluate_predictions(y_true, y_prob, threshold=default_threshold)

    # 2. Optimal Threshold Search (Maximizing F1 and FPR <= 0.05)
    best_f1_th, best_f1_metrics = find_optimal_threshold(
        y_true, y_prob, criterion="max_f1"
    )
    low_far_th, low_far_metrics = find_optimal_threshold(
        y_true, y_prob, criterion="constrained_fpr", max_acceptable_fpr=0.05
    )

    # 3. Categorical Breakdown (Campus activities, sports, benign crowds)
    unique_cats = sorted(list(set(all_categories)))
    category_breakdown = {}

    for cat in unique_cats:
        cat_indices = [i for i, c in enumerate(all_categories) if c == cat]
        c_true = y_true[cat_indices]
        c_prob = y_prob[cat_indices]
        c_pred = (c_prob >= default_threshold).astype(int)

        tp = int(np.sum((c_true == 1) & (c_pred == 1)))
        fp = int(np.sum((c_true == 0) & (c_pred == 1)))
        tn = int(np.sum((c_true == 0) & (c_pred == 0)))
        fn = int(np.sum((c_true == 1) & (c_pred == 0)))

        category_breakdown[cat] = {
            "total_clips": len(cat_indices),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "mean_probability": float(np.mean(c_prob)),
            "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "recall": float(tp / (tp + fn))
            if (tp + fn) > 0
            else (1.0 if fp == 0 else 0.0),
        }

    # 4. Latency / Benchmark
    avg_latency = float(np.mean(all_latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    report = {
        "summary": {
            "total_samples": len(y_true),
            "inference_latency_ms": round(avg_latency, 2),
            "throughput_clips_per_sec": round(fps, 1),
        },
        "default_threshold_metrics": default_metrics.to_dict(),
        "optimal_f1_tuning": {
            "optimal_threshold": round(best_f1_th, 3),
            "metrics": best_f1_metrics.to_dict(),
        },
        "constrained_low_far_tuning": {
            "optimal_threshold": round(low_far_th, 3),
            "metrics": low_far_metrics.to_dict(),
        },
        "category_breakdown": category_breakdown,
    }

    return report


@torch.no_grad()
def run_video_level_evaluation(
    model: nn.Module,
    dataset: VideoBagDataset,
    device: torch.device,
    threshold: float = 0.50,
    top_k: int = 1,
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, Any]:
    """Evaluate one score per source video; never tune a threshold on the test set."""
    loader = create_video_bag_dataloader(
        dataset,
        batch_size=batch_size,
        is_training=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    all_targets: list[int] = []
    all_probs: list[float] = []
    all_categories: list[str] = []
    all_video_ids: list[str] = []
    all_paths: list[str] = []
    latencies: list[float] = []
    model.eval()
    for bags, labels, meta in loader:
        bags = bags.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = pool_video_bag_logits(model, bags, top_k=top_k)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        batch_size_actual = len(probabilities)
        all_targets.extend(labels.numpy().astype(int).flatten().tolist())
        all_probs.extend(probabilities.astype(float).tolist())
        all_categories.extend(meta["category"])
        all_video_ids.extend(meta["source_video_id"])
        all_paths.extend(meta["video_path"])
        latencies.extend([elapsed_ms / max(1, batch_size_actual)] * batch_size_actual)

    y_true = np.asarray(all_targets, dtype=int)
    y_prob = np.asarray(all_probs, dtype=float)
    metrics = evaluate_predictions(y_true, y_prob, threshold=threshold)
    per_video = [
        {
            "source_video_id": video_id,
            "video_path": path,
            "category": category,
            "label": int(label),
            "violence_probability": round(float(probability), 6),
            "prediction": int(probability >= threshold),
        }
        for video_id, path, category, label, probability in zip(
            all_video_ids, all_paths, all_categories, y_true, y_prob
        )
    ]

    categories: dict[str, Any] = {}
    for category in sorted(set(all_categories)):
        indices = [i for i, value in enumerate(all_categories) if value == category]
        categories[category] = evaluate_predictions(
            y_true[indices], y_prob[indices], threshold
        ).to_dict()

    return {
        "evaluation_unit": "source_video",
        "threshold": float(threshold),
        "summary": {
            "total_videos": len(y_true),
            "mean_inference_ms_per_video": round(float(np.mean(latencies)), 2)
            if latencies
            else 0.0,
            "videos_per_second": round(1000.0 / float(np.mean(latencies)), 2)
            if latencies and np.mean(latencies) > 0
            else 0.0,
        },
        "metrics": metrics.to_dict(),
        "category_breakdown": categories,
        "predictions": per_video,
        "note": "Threshold was selected on validation videos and loaded from the checkpoint. Held-out test videos were not used for threshold selection.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Temporal Behavior Recognition Model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/behavior/training_config.yaml", help="Config file"
    )
    parser.add_argument("--arch", type=str, default=None, help="Model architecture")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/behavior/best_model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--manifest", type=str, default=None, help="Evaluation manifest file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the validation-selected threshold stored in the checkpoint",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/behavior/logs/xd_test_evaluation.json",
        help="Path to save evaluation JSON",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device, dev_mgr = setup_device("auto")

    # Load checkpoint first to detect architecture if available
    ckpt = None
    if os.path.exists(args.checkpoint):
        logger.info("Loading checkpoint from: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "config" in ckpt:
            saved_arch = ckpt["config"].get("model", {}).get("architecture")
            if saved_arch:
                cfg["model"]["architecture"] = saved_arch
                logger.info(
                    "Auto-detected architecture '%s' from checkpoint metadata.",
                    saved_arch,
                )

    if args.arch:
        cfg["model"]["architecture"] = args.arch

    if ckpt is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train a model before evaluation; "
            "the evaluator does not substitute random weights or generated data."
        )

    threshold = args.threshold
    if threshold is None:
        threshold = float(
            ckpt.get(
                "validation_threshold",
                cfg.get("evaluation", {}).get("decision_threshold", 0.50),
            )
        )
    logger.info("Using fixed evaluation threshold %.3f", threshold)

    manifest_path = args.manifest or cfg["dataset"].get(
        "test_manifest", "./data/behavior/splits/xd_test_manifest.json"
    )
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Evaluation manifest not found: {manifest_path}. Prepare the XD-Violence manifests first."
        )

    use_video_mil = cfg["dataset"].get("supervision", "clip") == "weak_video_mil"
    if use_video_mil:
        dataset = VideoBagDataset(
            manifest_path,
            project_root=Path.cwd(),
            clip_length=cfg["dataset"]["clip_length"],
            sampling_stride=cfg["dataset"]["sampling_stride"],
            spatial_resolution=tuple(cfg["dataset"]["spatial_resolution"]),
            clips_per_video=cfg["dataset"].get("clips_per_video", 4),
            is_training=False,
        )
    else:
        dataset = VideoClipDataset(
            manifest_path_or_records=manifest_path,
            clip_length=cfg["dataset"]["clip_length"],
            sampling_stride=cfg["dataset"]["sampling_stride"],
            spatial_resolution=tuple(cfg["dataset"]["spatial_resolution"]),
            is_training=False,
        )

    # The checkpoint supplies the weights; evaluation should not download a
    # second copy of the ImageNet initialization.
    cfg["model"]["pretrained"] = False
    model = build_temporal_model(cfg).to(device)

    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    if use_video_mil:
        report = run_video_level_evaluation(
            model,
            dataset,
            device,
            threshold=threshold,
            top_k=cfg["dataset"].get("mil_top_k", 1),
            batch_size=cfg["training"].get("batch_size", 2),
            num_workers=cfg["dataset"].get("num_workers", 0),
            pin_memory=dev_mgr.is_cuda,
        )
        print("\nXD-VIOLENCE HELD-OUT VIDEO-LEVEL REPORT")
        print(json.dumps(report["metrics"], indent=2))
    else:
        report = run_full_evaluation(
            model, dataset, device, default_threshold=threshold
        )

    report["dataset_provenance"] = ckpt.get(
        "dataset_provenance", {"training_scope": "unknown"}
    )

    # Print summary
    print("\n" + "=" * 60)
    print("           TEMPORAL VIOLENCE CLASSIFICATION REPORT")
    print("=" * 60)
    if use_video_mil:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved held-out video-level evaluation to: %s", output_path)
        return

    m = report["default_threshold_metrics"]
    print(f"Accuracy         : {m['accuracy'] * 100:.2f}%")
    print(f"Precision        : {m['precision'] * 100:.2f}%")
    print(f"Recall           : {m['recall'] * 100:.2f}%")
    print(f"F1-Score         : {m['f1']:.4f}")
    print(f"AUROC            : {m['auroc']:.4f}")
    print(f"AUPRC            : {m['auprc']:.4f}")
    print(f"False Alarm Rate : {m['false_alarm_rate'] * 100:.2f}% (FPR)")
    print(
        f"Inference Latency: {report['summary']['inference_latency_ms']} ms/clip ({report['summary']['throughput_clips_per_sec']} clips/s)"
    )
    print("-" * 60)
    print("Category Breakdown:")
    for cat, c_info in report["category_breakdown"].items():
        print(
            f"  [{cat:<20}] Clips: {c_info['total_clips']:<4} | FAR: {c_info['false_alarm_rate'] * 100:5.1f}% | Recall: {c_info['recall'] * 100:5.1f}% | Mean Prob: {c_info['mean_probability']:.3f}"
        )
    print("=" * 60 + "\n")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to: %s", args.output_json)


if __name__ == "__main__":
    main()
