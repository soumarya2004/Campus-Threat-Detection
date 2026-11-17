"""Read the detector defaults shared by the wrapper and command-line tools."""

import numbers
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "detector.yaml"


def validate_confidence(value, name="confidence") -> float:
    """Use the same threshold rules for YAML settings and Python overrides."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a number between 0 and 1")
    return float(value)


def validate_image_size(value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 32:
        raise ValueError("image_size must be an integer of at least 32 pixels")
    return int(value)


def load_detector_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"Expected detector settings in {path}")

    missing = {"model", "confidence"} - settings.keys()
    if missing:
        raise ValueError(
            f"Missing detector settings in {path}: {', '.join(sorted(missing))}"
        )
    if not isinstance(settings["model"], str) or not settings["model"].strip():
        raise ValueError("model must be a nonempty checkpoint path")
    model_path = Path(settings["model"]).expanduser()
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    return {
        "model_path": model_path.resolve(),
        "confidence": validate_confidence(settings["confidence"]),
        "image_size": validate_image_size(settings.get("image_size", 640)),
    }
