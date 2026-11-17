import pytest

from src.model_config import REPO_ROOT, load_detector_config


def test_model_path_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    config = tmp_path / "detector.yaml"
    config.write_text("model: weights/example.pt\nconfidence: 0.4\nimage_size: 640\n")
    monkeypatch.chdir(tmp_path)
    settings = load_detector_config(config)
    assert settings["model_path"] == REPO_ROOT / "weights" / "example.pt"
    assert settings["confidence"] == 0.4


@pytest.mark.parametrize("value", [".nan", "-0.1", "1.1", "true", "wrong"])
def test_bad_config_threshold_is_rejected(tmp_path, value):
    config = tmp_path / "detector.yaml"
    config.write_text(f"model: weights/example.pt\nconfidence: {value}\n")
    with pytest.raises(ValueError, match="confidence"):
        load_detector_config(config)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("model: weights/example.pt\n", "confidence"),
        ("confidence: 0.4\n", "model"),
        ("model: null\nconfidence: 0.4\n", "checkpoint path"),
        ("model: ''\nconfidence: 0.4\n", "checkpoint path"),
        (
            "model: weights/example.pt\nconfidence: 0.4\nimage_size: true\n",
            "image_size",
        ),
        (
            "model: weights/example.pt\nconfidence: 0.4\nimage_size: 640.5\n",
            "image_size",
        ),
    ],
)
def test_invalid_config_reports_the_setting(tmp_path, content, message):
    config = tmp_path / "detector.yaml"
    config.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_detector_config(config)
