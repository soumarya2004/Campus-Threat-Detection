from types import SimpleNamespace

import pytest

from scripts.compare_pretrained_baseline import (
    COCO_TO_PROJECT,
    COMMON_NAMES,
    collect_common_predictions,
    filtered_ground_truth,
    remap_prediction,
    validate_class_mapping,
)
from src.evaluation import evaluate_at_threshold


def test_coco_ids_are_mapped_by_meaning():
    validate_class_mapping({43: "knife", 34: "baseball bat"}, COCO_TO_PROJECT)
    assert remap_prediction([0, 0, 10, 10], 0.8, 43, COCO_TO_PROJECT)["class_id"] == 1
    assert remap_prediction([0, 0, 10, 10], 0.8, 34, COCO_TO_PROJECT)["class_id"] == 2
    with pytest.raises(ValueError, match="must mean"):
        validate_class_mapping({43: "fork", 34: "baseball bat"}, COCO_TO_PROJECT)


@pytest.mark.parametrize("source_id", [0, 43.5, float("nan")])
def test_unrequested_or_malformed_prediction_is_rejected(source_id):
    with pytest.raises(ValueError):
        remap_prediction([0, 0, 10, 10], 0.8, source_id, COCO_TO_PROJECT)


def test_ground_truth_is_filtered_after_validating_original_labels(tmp_path):
    label = tmp_path / "sample.txt"
    names = {0: "machete", 1: "knife", 2: "baseball bat", 3: "rifle", 4: "gun"}
    label.write_text("4 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    truth = filtered_ground_truth(label, 100, 100, names)
    assert [obj["class_id"] for obj in truth] == [1]
    label.write_text("4 0.5 0.5 -0.2 0.2\n")
    with pytest.raises(ValueError, match="invalid normalized box"):
        filtered_ground_truth(label, 100, 100, names)


def test_other_class_images_remain_negatives_and_count_false_positives(tmp_path):
    label = tmp_path / "sample.txt"
    label.write_text("4 0.5 0.5 0.2 0.2\n")
    image = tmp_path / "sample.jpg"
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(path=str(image), orig_shape=(100, 100), boxes=None)]

    model = SimpleNamespace(names={43: "knife", 34: "baseball bat"}, predict=predict)
    records = collect_common_predictions(
        model,
        [{"image": "sample.jpg", "image_path": image, "label_path": label}],
        {4: "gun", **COMMON_NAMES},
        COCO_TO_PROJECT,
        device="cpu",
        batch=8,
    )
    assert len(records) == 1 and records[0]["ground_truth"] == []
    assert calls[0]["classes"] == [34, 43]
    assert calls[0]["rect"] is False
    records[0]["predictions"] = [
        remap_prediction([0, 0, 10, 10], 0.8, 43, COCO_TO_PROJECT)
    ]
    aggregate = evaluate_at_threshold(records, COMMON_NAMES, 0.25)["aggregate"]
    assert aggregate["negative_images"] == 1
    assert aggregate["negative_images_with_false_positives"] == 1
    assert aggregate["fp"] == 1
