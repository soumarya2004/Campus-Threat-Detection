"""Submission export includes both verified models and excludes local working data."""

import json
import shutil
import zipfile

import pytest

from scripts import package_person1 as packaging


@pytest.fixture(scope="module")
def source_snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("submission_source")
    for name, content in packaging.collect_sources(packaging.REPO_ROOT).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.fixture
def project(source_snapshot, tmp_path):
    root = tmp_path / "project"
    shutil.copytree(source_snapshot, root)
    return root


def test_package_contains_both_models_and_complete_payload_hash_manifest(
    project, tmp_path
):
    for excluded in (
        ".git/config",
        ".venv/private.py",
        "data/raw/photo.jpg",
        "runs/best.pt",
        "outputs/frame.json",
        "tmp/private.py",
        "src/__pycache__/cached.py",
        ".env",
        "yolo11s.pt",
    ):
        path = project / excluded
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not be exported")
    output = tmp_path / "submission.zip"
    result = packaging.package_project(project, output)
    assert result["sha256"] == packaging.digest(output.read_bytes())
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "yolo11n.pt" in names
        assert "models/dangerous_items_yolo11n.pt" in names
        assert "models/person_model_card.json" in names
        assert "scripts/run_camera_demo.py" in names
        assert "BUNDLE_README.txt" in names
        assert "data.yaml" in names
        assert not any(
            set(name.split("/")) & packaging.EXCLUDED_PARTS for name in names
        )
        assert ".env" not in names and "yolo11s.pt" not in names
        manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
        assert {item["path"] for item in manifest["files"]} == set(names) - {
            "SOURCE_MANIFEST.json"
        }
        for item in manifest["files"]:
            content = archive.read(item["path"])
            assert item["bytes"] == len(content)
            assert item["sha256"] == packaging.digest(content)


def test_archive_bytes_are_repeatable_for_identical_sources(project, tmp_path):
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    packaging.package_project(project, first)
    packaging.package_project(project, second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    "checkpoint", ["models/dangerous_items_yolo11n.pt", "yolo11n.pt"]
)
def test_corrupted_checkpoint_refuses_export(project, tmp_path, checkpoint):
    (project / checkpoint).write_bytes(b"corrupt checkpoint")
    output = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="hash|release checks"):
        packaging.package_project(project, output)
    assert not output.exists()


def test_missing_required_source_refuses_export(project, tmp_path):
    (project / "scripts/run_camera_demo.py").unlink()
    output = tmp_path / "submission.zip"
    with pytest.raises(FileNotFoundError, match="run_camera_demo"):
        packaging.package_project(project, output)
    assert not output.exists()


def test_existing_archive_is_preserved_by_default(project, tmp_path):
    output = tmp_path / "submission.zip"
    output.write_bytes(b"existing submission")
    with pytest.raises(FileExistsError, match="already exists"):
        packaging.package_project(project, output)
    assert output.read_bytes() == b"existing submission"


def test_output_must_be_a_zip(project, tmp_path):
    with pytest.raises(ValueError, match=".zip extension"):
        packaging.package_project(project, tmp_path / "README.md")
