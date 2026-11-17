"""Small real-file tests for the dataset's scientifically important guarantees."""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.build_grouped_dataset import (
    assign_splits,
    build_dataset,
    inventory,
    make_groups,
)
from scripts.verify_clean_splits import verify_dataset


def make_source(tmp_path):
    root = tmp_path / "source"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    config = root / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "knife", 1: "bat"},
            }
        )
    )
    return root, config


def add_image(root, split, name, value, label="0 0.5 0.5 0.5 0.5\n"):
    assert cv2.imwrite(
        str(root / "images" / split / name), np.full((12, 16, 3), value, np.uint8)
    )
    (root / "labels" / split / Path(name).with_suffix(".txt")).write_text(label)


def test_transitive_hash_and_augmentation_groups_choose_original(tmp_path):
    root, config = make_source(tmp_path)
    add_image(root, "train", "first.png", 10)
    add_image(root, "val", "A6_first.png", 20)
    add_image(root, "test", "second.png", 20)
    add_image(root, "test", "A4_second.png", 30)
    records, _, _ = inventory(config)
    groups, excluded = make_groups(records)
    assert not excluded
    assert len(groups) == 1
    assert len(groups[0]["records"]) == 4
    assert not groups[0]["representative"]["image"].stem.startswith("A")


def test_conflicting_exact_labels_exclude_whole_family(tmp_path):
    root, config = make_source(tmp_path)
    add_image(root, "train", "first.png", 10)
    add_image(root, "val", "duplicate.png", 10, "1 0.5 0.5 0.5 0.5\n")
    add_image(root, "test", "A6_first.png", 20)
    groups, excluded = make_groups(inventory(config)[0])
    assert groups == []
    assert len(excluded) == 1 and len(excluded[0]["records"]) == 3


def test_assignment_is_seeded_and_balanced():
    groups = [{"id": str(i), "counts": Counter({i % 2: 1})} for i in range(100)]
    first = assign_splits(groups)
    assert first == assign_splits(groups)
    assert first != assign_splits(groups, seed=9)
    assert Counter(first.values()) == {"train": 70, "val": 15, "test": 15}
    for split, target in (("train", 35), ("val", 7.5), ("test", 7.5)):
        assert abs(sum(first[str(i)] == split for i in range(0, 100, 2)) - target) <= 1


def test_builder_and_manifest_detect_modified_labels(tmp_path):
    root, config = make_source(tmp_path)
    for i in range(20):
        add_image(
            root,
            ("train", "val", "test")[i % 3],
            f"photo{i}.png",
            i + 1,
            "" if i % 4 == 0 else "0 0.5 0.5 0.5 0.5\n",
        )
    add_image(root, "test", "A6_photo0.png", 80, "")
    output, reports = tmp_path / "output", tmp_path / "reports"
    summary = build_dataset(config, output, reports)
    assert summary["selected_images"] == 20
    second = build_dataset(
        config, tmp_path / "second_output", tmp_path / "second_reports"
    )
    assert summary["manifest_sha256"] == second["manifest_sha256"]
    manifest = reports / "grouped_v1_manifest.csv"
    assert verify_dataset(output / "data.yaml", manifest)["problems"] == []
    label = next((output / "labels" / "test").glob("*.txt"))
    label.write_text("1 0.5 0.5 0.5 0.5\n")
    with pytest.raises(ValueError, match="Manifest mismatch"):
        verify_dataset(output / "data.yaml", manifest)
    with pytest.raises(ValueError, match="already exists"):
        build_dataset(config, output, reports)


def test_verifier_rejects_filename_leak_even_when_hashes_differ(tmp_path):
    root, config = make_source(tmp_path)
    add_image(root, "train", "first.png", 1)
    add_image(root, "val", "another.png", 2)
    add_image(root, "test", "A16_first.png", 3)
    with pytest.raises(ValueError, match="augmentation filename families cross splits"):
        verify_dataset(config)


def test_verifier_rejects_hash_leak_with_unrelated_names(tmp_path):
    root, config = make_source(tmp_path)
    add_image(root, "train", "first.png", 1)
    add_image(root, "val", "another.png", 2)
    add_image(root, "test", "unrelated.png", 1)
    with pytest.raises(ValueError, match="exact image hashes cross splits"):
        verify_dataset(config)


def test_empty_source_fails_without_creating_output(tmp_path):
    _, config = make_source(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Missing or empty"):
        build_dataset(config, output, tmp_path / "reports")
    assert not output.exists()


def test_missing_label_fails_preflight(tmp_path):
    root, config = make_source(tmp_path)
    for i, split in enumerate(("train", "val", "test")):
        add_image(root, split, f"photo{i}.png", i + 1)
    (root / "labels" / "val" / "photo1.txt").unlink()
    with pytest.raises(ValueError, match="Missing label"):
        inventory(config)


def test_nonfinite_annotation_fails_preflight(tmp_path):
    root, config = make_source(tmp_path)
    for i, split in enumerate(("train", "val", "test")):
        add_image(root, split, f"photo{i}.png", i + 1)
    (root / "labels" / "train" / "photo0.txt").write_text("0 nan 0.5 0.5 0.5\n")
    with pytest.raises(ValueError, match="non-finite"):
        inventory(config)


def test_rebuild_reuses_identical_frozen_reports_without_overwriting(tmp_path):
    root, config = make_source(tmp_path)
    for i in range(10):
        add_image(root, ("train", "val", "test")[i % 3], f"photo{i}.png", i + 1)
    reports = tmp_path / "reports"
    first = build_dataset(config, tmp_path / "first", reports)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in reports.iterdir()}
    second = build_dataset(config, tmp_path / "second", reports)
    assert first == second
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in reports.iterdir()
    }
    assert (
        verify_dataset(
            tmp_path / "second" / "data.yaml", reports / "grouped_v1_manifest.csv"
        )["problems"]
        == []
    )


@pytest.mark.parametrize(
    "artifact", ["grouped_v1_manifest.csv", "grouped_v1_summary.json"]
)
def test_changed_frozen_report_blocks_rebuild_before_creating_dataset(
    tmp_path, artifact
):
    root, config = make_source(tmp_path)
    for i in range(10):
        add_image(root, ("train", "val", "test")[i % 3], f"photo{i}.png", i + 1)
    reports = tmp_path / "reports"
    build_dataset(config, tmp_path / "first", reports)
    path = reports / artifact
    changed = path.read_bytes() + b"changed"
    path.write_bytes(changed)
    output = tmp_path / "rebuild"
    with pytest.raises(ValueError, match="Existing report differs"):
        build_dataset(config, output, reports)
    assert not output.exists()
    assert path.read_bytes() == changed
