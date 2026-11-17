"""Prevent the demo defaults from drifting away from the evaluated model."""

import json
import shutil

import pytest
import yaml

from scripts.check_model_release import REPO_ROOT, check_release


@pytest.fixture
def release(tmp_path):
    for directory in ("configs", "models", "reports/dataset", "reports/experiments"):
        shutil.copytree(REPO_ROOT / directory, tmp_path / directory)
    return tmp_path


def test_shipped_release_is_consistent():
    result = check_release()
    assert result["passed"], [item for item in result["checks"] if not item["passed"]]


@pytest.mark.parametrize(("key", "value"), [("confidence", 0.35), ("image_size", 960)])
def test_changed_defaults_fail(release, key, value):
    path = release / "configs/detector.yaml"
    settings = yaml.safe_load(path.read_text())
    settings[key] = value
    path.write_text(yaml.safe_dump(settings))
    assert not check_release(release)["passed"]


def test_replaced_checkpoint_fails_even_at_same_path(release):
    (release / "models/dangerous_items_yolo11n.pt").write_bytes(b"different checkpoint")
    assert not check_release(release)["passed"]


def test_changed_model_card_metrics_fail(release):
    path = release / "models/model_card.json"
    card = json.loads(path.read_text())
    card["test"]["map50"] = 0.99
    path.write_text(json.dumps(card))
    assert not check_release(release)["passed"]


def test_check_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert check_release(REPO_ROOT)["passed"]
