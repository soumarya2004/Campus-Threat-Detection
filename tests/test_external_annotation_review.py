import hashlib
import json

from PIL import Image
import pytest

from scripts.review_external_annotations import ambiguity_diagnostic, validate_records


@pytest.fixture
def sample(tmp_path):
    images = tmp_path / "images/test"
    labels = tmp_path / "labels/test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image = images / "sample.png"
    Image.new("RGB", (100, 80), "white").save(image)
    label = labels / "sample.txt"
    label.write_text("1 0.3 0.5 0.4 0.5\n")
    record = {
        "image": image.name,
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
        "width": 100,
        "height": 80,
        "annotations": [{"class_id": 1, "bbox_xyxy": [10, 20, 50, 60]}],
        "explicit_background_review": False,
    }
    manifest = tmp_path / "annotations.json"
    manifest.write_text(json.dumps({"records": [record]}))
    return tmp_path, manifest, record, label


def test_consistent_frozen_labels_pass(sample):
    dataset, manifest, record, _ = sample
    assert validate_records(manifest, dataset) == [record]


def test_empty_manifest_is_not_a_successful_review(sample):
    dataset, manifest, _, _ = sample
    manifest.write_text('{"records": []}')
    with pytest.raises(ValueError, match="no records"):
        validate_records(manifest, dataset)


def test_modified_label_bytes_are_rejected(sample):
    dataset, manifest, _, label = sample
    label.write_text("1 0.3 0.5 0.8 0.5\n")
    with pytest.raises(ValueError, match="Label hash"):
        validate_records(manifest, dataset)


def test_rehashed_incorrect_yolo_coordinates_are_rejected(sample):
    dataset, manifest, record, label = sample
    label.write_text("1 0.3 0.5 0.8 0.5\n")
    record["label_sha256"] = hashlib.sha256(label.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"records": [record]}))
    with pytest.raises(ValueError, match="YOLO coordinates"):
        validate_records(manifest, dataset)


def test_manifest_geometry_cannot_extend_outside_image(sample):
    dataset, manifest, record, _ = sample
    record["annotations"][0]["bbox_xyxy"] = [-1, 20, 50, 60]
    manifest.write_text(json.dumps({"records": [record]}))
    with pytest.raises(ValueError, match="outside image"):
        validate_records(manifest, dataset)


def test_duplicate_images_are_rejected(sample):
    dataset, manifest, record, _ = sample
    manifest.write_text(json.dumps({"records": [record, record]}))
    with pytest.raises(ValueError, match="unique"):
        validate_records(manifest, dataset)


def test_diagnostic_uses_original_flags_and_stored_matches():
    records = [
        {
            "image": "one.png",
            "annotations": [{"class_id": 1, "bbox_xyxy": [0, 0, 10, 10]}],
            "boundary_ambiguity": True,
        }
    ]
    evaluation = {
        "fixed_operating_point": {
            "confidence_threshold": 0.45,
            "matching_iou": 0.5,
            "images": [
                {
                    "image": "one.png",
                    "ground_truth": [{"class_id": 1, "bbox": [0, 0, 10, 10]}],
                    "matches": [[0, 0]],
                    "false_positive_indices": [],
                    "missed_ground_truth_indices": [],
                }
            ],
        }
    }
    result = ambiguity_diagnostic(records, evaluation)
    assert result["groups"]["ambiguous_positive"] == {
        "images": 1,
        "tp": 1,
        "fp": 0,
        "fn": 0,
    }
    assert result["groups"]["unflagged_positive"]["images"] == 0


def test_diagnostic_rejects_evaluation_of_different_images():
    with pytest.raises(ValueError, match="differ"):
        ambiguity_diagnostic(
            [], {"fixed_operating_point": {"images": [{"image": "extra.png"}]}}
        )
