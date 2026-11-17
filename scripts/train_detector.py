"""Train a fresh YOLO11n model using the grouped dataset.

Example: python scripts/train_detector.py --config configs/train.yaml
Use --batch 8 --workers 0 --no-amp if a laptop needs the conservative settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (
    file_sha256,
    json_ready,
    load_split,
    prepare_dataset_yaml,
    repo_path,
    split_fingerprint,
)


def training_dataset_provenance(data_path: Path) -> dict:
    """Record the data actually selected by --data, including image and label bytes."""
    provenance = {}
    for split in ("train", "val"):
        _, entries = load_split(data_path, split)
        provenance[split] = split_fingerprint(data_path, entries)
    return provenance


def load_training_settings(config_path: Path, overrides: dict | None = None) -> dict:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Training config must be a YAML mapping.")
    settings = dict(settings)
    settings.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key in ("model", "data", "project"):
        settings[key] = str(repo_path(settings[key]))
    if Path(settings["model"]).name != "yolo11n.pt":
        raise ValueError(
            "Use the original yolo11n.pt pretrained model for this experiment."
        )
    for key in ("model", "data"):
        if not Path(settings[key]).is_file():
            raise FileNotFoundError(f"{key} file not found: {settings[key]}")
    for key in ("epochs", "imgsz", "batch"):
        if (
            not isinstance(settings[key], int)
            or isinstance(settings[key], bool)
            or settings[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer.")
    if settings.get("workers", 0) < 0:
        raise ValueError("workers must be non-negative.")
    if settings.get("resume"):
        raise ValueError(
            "This experiment must start from pretrained weights, not resume a fine-tune."
        )
    settings["exist_ok"] = False
    settings["val"] = True
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    overrides = vars(args).copy()
    config_path = repo_path(overrides.pop("config"))
    settings = load_training_settings(config_path, overrides)
    pretrained = settings.pop("model")
    original_data = Path(settings["data"])
    dataset_provenance = training_dataset_provenance(original_data)
    settings["data"] = str(
        prepare_dataset_yaml(
            original_data, REPO_ROOT / "outputs" / "experiment_configs"
        )
    )

    # Import here so --help and config tests do not load PyTorch or the GPU.
    import ultralytics
    from ultralytics import YOLO

    record = {
        "pretrained_model": pretrained,
        "pretrained_sha256": file_sha256(Path(pretrained)),
        "original_dataset_yaml": str(original_data),
        "dataset_yaml_sha256": file_sha256(original_data),
        "resolved_dataset_yaml_sha256": file_sha256(Path(settings["data"])),
        "dataset_splits": dataset_provenance,
        "ultralytics_version": ultralytics.__version__,
        "settings": settings,
        "note": "Fresh pretrained initialization; historical fine-tuned checkpoints are not used.",
    }
    print(json.dumps(json_ready(record), indent=2))
    model = YOLO(pretrained)
    model.train(**settings)
    output_dir = Path(model.trainer.save_dir)
    record["output_dir"] = str(output_dir)
    (output_dir / "experiment.json").write_text(
        json.dumps(json_ready(record), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Training complete: {output_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
