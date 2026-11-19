"""Run the frozen behavior model on UCF-Crime's official test videos."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_source import CameraConfig, CameraSource, CaptureError  # noqa: E402
from src.external_evaluation.ucf_crime import (  # noqa: E402
    evaluate_ucf_run,
    load_window_predictions,
    read_manifest,
    sha256_file,
)
from src.monitoring.session import DashboardSessionWriter  # noqa: E402


def _load_checkpoint_config(path: Path) -> tuple[dict, float | None, str | None]:
    with path.open("rb") as handle:
        header = handle.read(160)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the checkpoint payload. "
            "Fetch the LFS object or train and select a checkpoint first."
        )
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_config = (
        checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    )
    checkpoint_config = (
        deepcopy(checkpoint_config) if isinstance(checkpoint_config, dict) else {}
    )
    threshold = (
        checkpoint.get("validation_threshold") if isinstance(checkpoint, dict) else None
    )
    threshold_source = (
        "checkpoint validation_threshold" if threshold is not None else None
    )
    if threshold is None:
        threshold = checkpoint_config.get("evaluation", {}).get("decision_threshold")
        threshold_source = "checkpoint config" if threshold is not None else None
    if threshold is not None:
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Checkpoint validation threshold must be between 0 and 1")
        checkpoint_config.setdefault("evaluation", {})["decision_threshold"] = threshold
    return checkpoint_config, threshold, threshold_source


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/behavior/best_model.pt"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/behavior/training_config.yaml"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--infer-every-n-frames", type=int, default=2)
    parser.add_argument("--merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--event-iou-threshold", type=float, default=0.1)
    parser.add_argument(
        "--max-videos", type=int, help="Partial-run limit for a smoke review"
    )
    args = parser.parse_args(argv)
    if args.infer_every_n_frames < 1:
        parser.error("--infer-every-n-frames must be at least 1")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be at least 1")
    if (
        not math.isfinite(args.merge_gap_seconds)
        or args.merge_gap_seconds < 0
        or not math.isfinite(args.event_iou_threshold)
        or not 0 <= args.event_iou_threshold <= 1
    ):
        parser.error("Event settings are outside their valid range")

    try:
        manifest = read_manifest(args.manifest)
        if any(row["split"] != "official_test" for row in manifest):
            raise ValueError("This evaluator accepts only the official_test manifest")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        checkpoint_config, validation_threshold, threshold_source = (
            _load_checkpoint_config(args.checkpoint)
        )
        effective_config = checkpoint_config or config
        if validation_threshold is None:
            validation_threshold = float(config["evaluation"]["decision_threshold"])
            threshold_source = "training configuration fallback"
        if (
            not math.isfinite(validation_threshold)
            or not 0 <= validation_threshold <= 1
        ):
            raise ValueError("Configured threshold must be between 0 and 1")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    all_in_scope = [row for row in manifest if row["target_label"] in (0, 1)]
    selected = (
        all_in_scope[: args.max_videos] if args.max_videos is not None else all_in_scope
    )
    in_scope = [row for row in selected if row["file_exists"]]
    if not selected:
        parser.error("The manifest has no in-scope UCF-Crime test videos")
    if not in_scope:
        parser.error("No selected in-scope video is present on disk")
    missing_video_ids = [row["video_id"] for row in selected if not row["file_exists"]]

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        from src.behavior.models.behavior_classifier import BehaviorClassifier

        device = (
            "cuda:0"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        dataset_config = effective_config.get("dataset", config.get("dataset", {}))
        model_config = deepcopy(effective_config)
        model_config.setdefault("evaluation", {})["decision_threshold"] = (
            validation_threshold
        )
        classifier = BehaviorClassifier(
            checkpoint_path=str(args.checkpoint),
            model_config=model_config,
            clip_length=dataset_config.get("clip_length", 16),
            sampling_stride=dataset_config.get("sampling_stride", 2),
            spatial_resolution=tuple(
                dataset_config.get("spatial_resolution", (112, 112))
            ),
            threshold=validation_threshold,
            device=device,
        )
    except Exception as exc:
        parser.error(
            f"Could not initialize the saved behavior model: {type(exc).__name__}: {exc}"
        )

    run_id = datetime.now(timezone.utc).strftime("ucf-%Y%m%dT%H%M%SZ")
    run_started = time.perf_counter()
    predictions_path = output_dir / "predictions.jsonl"
    failed_videos = [
        {"video_id": video_id, "error": "video file is missing from the local dataset"}
        for video_id in missing_video_ids
    ]
    processed_videos = 0
    total_frames = 0
    total_windows = 0
    status_writer = DashboardSessionWriter(
        output_dir,
        metadata={
            "run_id": run_id,
            "purpose": "external UCF-Crime behavior evaluation; no training",
            "threshold_source": threshold_source,
        },
    )

    try:
        with predictions_path.open(
            "w", encoding="utf-8", newline=""
        ) as predictions_file:
            for index, row in enumerate(in_scope, 1):
                video_id = row["video_id"]
                video_frames = 0
                video_windows = 0
                video_started = time.perf_counter()
                classifier.reset()
                with tempfile.TemporaryFile(
                    mode="w+t", encoding="utf-8", newline=""
                ) as video_predictions:
                    try:
                        source = CameraSource(
                            CameraConfig(
                                source=Path(row["video_path"]), reconnect_attempts=0
                            )
                        )
                        with source:
                            while True:
                                packet = source.read()
                                if packet is None:
                                    break
                                total_frames += 1
                                video_frames += 1
                                classifier.ingest_frame(
                                    packet.frame,
                                    timestamp=packet.metadata["timestamp_s"],
                                )
                                if video_frames % args.infer_every_n_frames:
                                    continue
                                result = classifier.infer()
                                scored_at = time.perf_counter()
                                if result["model_status"] != "valid":
                                    continue
                                record = {
                                    "video_id": video_id,
                                    "frame_id": int(packet.metadata["frame_id"]),
                                    "timestamp_s": float(
                                        packet.metadata["timestamp_s"]
                                    ),
                                    "window_start_s": float(result["window_start"]),
                                    "window_end_s": float(result["window_end"]),
                                    "violence_probability": float(
                                        result["violence_probability"]
                                    ),
                                    "model_status": result["model_status"],
                                    "inference_latency_ms": float(
                                        result["inference_latency_ms"]
                                    ),
                                    "capture_to_score_ms": max(
                                        0.0,
                                        (
                                            scored_at
                                            - float(packet.metadata["monotonic_time"])
                                        )
                                        * 1000.0,
                                    ),
                                }
                                video_predictions.write(
                                    json.dumps(record, allow_nan=False) + "\n"
                                )
                                video_windows += 1
                        if video_frames == 0:
                            raise ValueError("video contains no decodable frames")
                        video_predictions.seek(0)
                        predictions_file.write(video_predictions.read())
                        total_windows += video_windows
                        processed_videos += 1
                    except (CaptureError, OSError, ValueError) as exc:
                        video_windows = 0
                        failed_videos.append(
                            {
                                "video_id": video_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

                elapsed = max(time.perf_counter() - video_started, 1e-9)
                status_writer.publish_frame(
                    None,
                    capture={
                        "status": "processing_video",
                        "source_type": "recorded_video",
                        "source_label": video_id,
                        "fps": row["fps"],
                    },
                    models={
                        "object_detection": "not_run",
                        "behavior_classification": "ready",
                        "tracking": "not_run",
                    },
                    performance={
                        "attempted_videos": index,
                        "successfully_processed_videos": processed_videos,
                        "failed_or_missing_videos": len(failed_videos),
                        "total_selected_videos": len(in_scope),
                        "current_video_frames": video_frames,
                        "current_video_windows": video_windows,
                        "current_video_processing_fps": video_frames / elapsed,
                    },
                    current_frame={"video_id": video_id, "frame_id": video_frames - 1},
                    assessment={
                        "level": "UNASSESSED",
                        "reason": "This run reports a behavior-model score; it does not apply a multi-model threat policy.",
                    },
                )
                predictions_file.flush()
                print(
                    f"[{index}/{len(in_scope)}] {video_id}: {video_frames} frames, {video_windows} scored windows"
                )

        predictions = load_window_predictions(predictions_path)
        elapsed_total = time.perf_counter() - run_started
        run_metadata = {
            "run_id": run_id,
            "completed_videos": processed_videos,
            "selected_videos": len(selected),
            "missing_video_ids": missing_video_ids,
            "failed_videos": failed_videos,
            "total_decoded_frames": total_frames,
            "valid_scored_windows": total_windows,
            "elapsed_wall_seconds": round(elapsed_total, 3),
            "video_processing_fps": round(total_frames / elapsed_total, 3)
            if elapsed_total
            else None,
            "device": str(device),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_file": args.checkpoint.name,
            "threshold_source": threshold_source,
            "manifest_sha256": sha256_file(args.manifest),
            "partial_run": (
                args.max_videos is not None
                or bool(missing_video_ids)
                or bool(failed_videos)
                or processed_videos < len(in_scope)
            ),
        }
        report = evaluate_ucf_run(
            selected,
            predictions,
            threshold=validation_threshold,
            merge_gap_s=args.merge_gap_seconds,
            event_iou_threshold=args.event_iou_threshold,
            run_metadata=run_metadata,
        )
        if run_metadata["partial_run"]:
            report["limitations"].append(
                "This is a partial test split; do not report it as the official full-split result."
            )
        _write_json(output_dir / "report.json", report)
        _write_json(output_dir / "run_metadata.json", run_metadata)
        status_writer.close(status="stopped")
        print(f"Evaluation report: {output_dir / 'report.json'}")
        return 0
    except Exception as exc:
        status_writer.close(status="error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
