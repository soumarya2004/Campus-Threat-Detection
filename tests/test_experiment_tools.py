"""Check experiment bookkeeping and detection matching without loading a model."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.evaluate_detector import read_threshold_decision, verify_threshold_context
from scripts.train_detector import load_training_settings, training_dataset_provenance
from src.evaluation import (
    box_iou,
    collect_predictions,
    evaluate_at_threshold,
    json_ready,
    load_split,
    match_detections,
    prepare_dataset_yaml,
    read_ground_truth,
    select_threshold,
    split_fingerprint,
    timing_summary,
    validate_report_paths,
)

NAMES = {0: "knife", 1: "gun"}


def target(class_id=0, bbox=(0, 0, 10, 10)):
    return {"class_id": class_id, "bbox": list(bbox)}


def prediction(confidence=0.8, class_id=0, bbox=(0, 0, 10, 10)):
    return {**target(class_id, bbox), "confidence": confidence}


def record(ground_truth, predictions, name="frame.jpg"):
    return {"image": name, "ground_truth": ground_truth, "predictions": predictions}


def test_iou_uses_intersection_over_union():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0


def test_duplicate_predictions_cannot_match_one_target_twice():
    matches, fp, fn = match_detections([prediction(0.3), prediction(0.9)], [target()])
    assert matches == [(1, 0)]
    assert fp == [0]
    assert fn == []


def test_each_prediction_can_match_only_one_target():
    matches, fp, fn = match_detections([prediction()], [target(), target()])
    assert len(matches) == 1
    assert len(fn) == 1
    assert not fp


def test_wrong_class_is_false_positive_and_false_negative():
    result = evaluate_at_threshold(
        [record([target(0)], [prediction(class_id=1)])], NAMES, 0.35
    )
    assert result["aggregate"]["tp"] == 0
    assert result["aggregate"]["fp"] == 1
    assert result["aggregate"]["fn"] == 1
    assert result["confusion_matrix"]["counts"][1][0] == 1
    assert result["images"][0]["missed_ground_truth_indices"] == [0]


def test_best_available_same_class_target_is_used():
    targets = [target(bbox=(0, 0, 10, 10)), target(bbox=(20, 20, 30, 30))]
    predictions = [prediction(0.9, bbox=(20, 20, 30, 30)), prediction(0.8)]
    matches, fp, fn = match_detections(predictions, targets)
    assert matches == [(0, 1), (1, 0)]
    assert not fp and not fn


def test_threshold_filters_predictions_before_matching_and_counts_negatives():
    records = [
        record([target()], [prediction(0.7)]),
        record([], [prediction(0.3)], "normal.jpg"),
    ]
    low = evaluate_at_threshold(records, NAMES, 0.2)
    high = evaluate_at_threshold(records, NAMES, 0.5)
    assert low["aggregate"]["precision"] == 0.5
    assert high["aggregate"]["precision"] == 1
    assert low["aggregate"]["negative_image_false_positive_rate"] == 1
    assert high["aggregate"]["negative_image_false_positive_rate"] == 0
    assert high["aggregate"]["recall"] == 1


def test_negative_rate_is_not_zero_when_no_negative_images_exist():
    result = evaluate_at_threshold([record([target()], [])], NAMES, 0.35)
    assert result["aggregate"]["negative_image_false_positive_rate"] is None


def test_confidence_boundary_matches_detector_strict_cutoff():
    result = evaluate_at_threshold([record([target()], [prediction(0.5)])], NAMES, 0.5)
    assert result["aggregate"]["tp"] == 0
    assert result["aggregate"]["fn"] == 1
    assert result["aggregate"]["macro_classes"] == [0]
    assert result["aggregate"]["fn"] == 1


def test_threshold_selection_optimizes_validation_macro_f1():
    records = [
        record([target()], [prediction(0.7)]),
        record([], [prediction(0.3)], "negative.jpg"),
    ]
    best, candidates = select_threshold(records, NAMES, [0.2, 0.5, 0.8], split="val")
    assert best["confidence_threshold"] == 0.5
    assert len(candidates) == 3
    assert "images" not in best


@pytest.mark.parametrize("split", ["test", "train"])
def test_threshold_selection_refuses_test_or_training_data(split):
    with pytest.raises(ValueError, match="validation"):
        select_threshold(
            [record([target()], [prediction()])], NAMES, [0.2, 0.5], split=split
        )


def test_threshold_tie_break_is_deterministic_and_prefers_lower_threshold():
    best, _ = select_threshold(
        [record([target()], [prediction(0.9)])], NAMES, [0.7, 0.2, 0.5], split="val"
    )
    assert best["confidence_threshold"] == 0.2


def test_empty_ground_truth_cannot_select_a_useful_threshold():
    with pytest.raises(ValueError, match="labelled"):
        select_threshold([record([], [])], NAMES, [0.2], split="val")


def test_json_export_handles_numpy_and_nonfinite_values():
    data = {
        "array": np.array([1, 2]),
        "number": np.float32(0.5),
        "count": np.int64(4),
        "missing": float("nan"),
    }
    encoded = json.dumps(json_ready(data), allow_nan=False)
    assert json.loads(encoded) == {
        "array": [1, 2],
        "number": 0.5,
        "count": 4,
        "missing": None,
    }


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_prediction_confidence_cannot_be_hidden_as_an_empty_frame(confidence):
    with pytest.raises(ValueError, match="confidence"):
        evaluate_at_threshold([record([], [prediction(confidence)])], NAMES, 0.35)


@pytest.mark.parametrize("class_id", [0.5, True, 99])
def test_invalid_predicted_class_ids_are_not_coerced(class_id):
    with pytest.raises(ValueError, match="class ID"):
        evaluate_at_threshold(
            [record([], [prediction(class_id=class_id)])], NAMES, 0.35
        )


@pytest.mark.parametrize("bbox", [(0, 0, float("inf"), 10), (0, 0, 0, 10)])
def test_invalid_predicted_geometry_stops_evaluation(bbox):
    with pytest.raises(ValueError, match="Bounding boxes"):
        evaluate_at_threshold([record([], [prediction(bbox=bbox)])], NAMES, 0.35)


def test_prediction_collection_rejects_wrong_image_pairing(tmp_path):
    from types import SimpleNamespace

    class WrongImageModel:
        names = NAMES

        def predict(self, **kwargs):
            return iter([SimpleNamespace(path=str(tmp_path / "other.jpg"))])

    entry = {
        "image": "expected.jpg",
        "image_path": tmp_path / "expected.jpg",
        "label_path": tmp_path / "expected.txt",
    }
    with pytest.raises(ValueError, match="path mismatch"):
        collect_predictions(
            WrongImageModel(),
            [entry],
            NAMES,
            confidence=0.001,
            imgsz=640,
            device="cpu",
            warmup_runs=0,
        )


def test_split_loading_and_fingerprint_detect_changed_labels(tmp_path):
    data = tmp_path / "data"
    (data / "images" / "val").mkdir(parents=True)
    (data / "labels" / "val").mkdir(parents=True)
    (data / "images" / "val" / "a.jpg").write_bytes(b"image content")
    label = data / "labels" / "val" / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump({"path": "data", "val": "images/val", "names": NAMES})
    )
    info, entries = load_split(config, "val")
    before = split_fingerprint(config, entries)
    label.write_text("1 0.5 0.5 0.2 0.2\n")
    after = split_fingerprint(config, entries)
    assert info["root"] == data.resolve()
    assert before["split_content_sha256"] != after["split_content_sha256"]
    assert before["dataset_yaml_sha256"] == after["dataset_yaml_sha256"]
    decision = {
        "selected_threshold": 0.4,
        "provenance": before,
        "inference": {
            "nms_iou": 0.7,
            "max_det": 300,
            "prediction_batch": 1,
            "cache_confidence": 0.001,
        },
    }
    with pytest.raises(ValueError, match="changed after threshold"):
        verify_threshold_context(decision, config)
    decision["provenance"] = after
    verify_threshold_context(decision, config)
    decision["inference"]["max_det"] = 10
    with pytest.raises(ValueError, match="max_det"):
        verify_threshold_context(decision, config)


def test_yolo_labels_convert_using_original_image_size(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.4\n")
    result = read_ground_truth(label, width=200, height=100, names=NAMES)
    assert result[0]["bbox"] == pytest.approx([80, 30, 120, 70])


def test_portable_config_is_resolved_for_ultralytics_without_mutating_source(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "dataset.yaml"
    content = "path: ../data\ntrain: images/train\nval: images/val\nnames: [knife]\n"
    config.write_text(content)
    copy = prepare_dataset_yaml(config, tmp_path / "resolved")
    assert yaml.safe_load(copy.read_text())["path"] == str(
        (tmp_path / "data").resolve()
    )
    assert config.read_text() == content
    assert prepare_dataset_yaml(config, tmp_path / "resolved") == copy


def test_timing_summary_combines_model_stages_only():
    records = [
        {"model_timing_ms": {"preprocess": 1, "inference": 6, "postprocess": 3}},
        {"model_timing_ms": {"preprocess": 2, "inference": 10, "postprocess": 8}},
    ]
    timing = timing_summary(records)
    assert timing["images"] == 2
    assert timing["mean_ms"] == 15
    assert timing["median_ms"] == 15


@pytest.mark.parametrize(
    "row", ["0 nan 0.5 0.2 0.4", "0 0.99 0.5 0.5 0.5", "4 0.5 0.5 0.2 0.2"]
)
def test_invalid_labels_stop_evaluation_instead_of_silently_changing_metrics(
    tmp_path, row
):
    label = tmp_path / "a.txt"
    label.write_text(row)
    with pytest.raises(ValueError, match="a.txt:1"):
        read_ground_truth(label, 100, 100, NAMES)


def test_threshold_artifact_binds_the_model_and_image_size(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(
        json.dumps(
            {
                "split": "val",
                "selected_threshold": 0.4,
                "provenance": {"model_sha256": "abc"},
                "inference": {"image_size": 640},
            }
        )
    )
    assert read_threshold_decision(path, "abc", 640)[0] == 0.4
    with pytest.raises(ValueError, match="different model"):
        read_threshold_decision(path, "other", 640)
    with pytest.raises(ValueError, match="image size"):
        read_threshold_decision(path, "abc", 320)


def test_training_config_disallows_old_finetuned_checkpoint(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {"model": "runs/old/best.pt", "data": "data.yaml", "project": "runs"}
        )
    )
    with pytest.raises(ValueError, match="original yolo11n.pt"):
        load_training_settings(config)


def test_training_overrides_keep_fresh_training_and_repo_paths(tmp_path):
    pretrained, data = tmp_path / "yolo11n.pt", tmp_path / "data.yaml"
    pretrained.write_bytes(b"mock checkpoint")
    data.write_text("names: [knife]\n")
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": str(pretrained),
                "data": str(data),
                "project": str(tmp_path / "runs"),
                "epochs": 50,
                "batch": 16,
                "workers": 2,
                "imgsz": 640,
                "amp": True,
            }
        )
    )
    settings = load_training_settings(
        config, {"batch": 8, "amp": False, "workers": 0, "epochs": None}
    )
    assert (
        settings["batch"] == 8 and settings["workers"] == 0 and settings["amp"] is False
    )
    assert settings["epochs"] == 50
    assert settings["exist_ok"] is False and settings["val"] is True
    assert Path(settings["model"]).is_absolute()


def test_training_provenance_tracks_selected_dataset_contents(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "val": "images/val", "names": NAMES}
        )
    )
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        (tmp_path / "images" / split / "frame.jpg").write_bytes(split.encode())
        (tmp_path / "labels" / split / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    before = training_dataset_provenance(config)
    (tmp_path / "labels" / "train" / "frame.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    after = training_dataset_provenance(config)
    assert (
        before["train"]["split_content_sha256"]
        != after["train"]["split_content_sha256"]
    )
    assert before["val"] == after["val"]
    assert (
        before["train"]["dataset_yaml_sha256"] == after["train"]["dataset_yaml_sha256"]
    )


def test_report_preflight_protects_model_even_with_overwrite(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"irreplaceable checkpoint")
    with pytest.raises(ValueError, match="experiment input"):
        validate_report_paths([model], [model], overwrite=True)
    assert model.read_bytes() == b"irreplaceable checkpoint"


def test_report_and_cache_cannot_replace_each_other(tmp_path):
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="different output paths"):
        validate_report_paths([output, output], [])
    assert not output.exists()


def test_existing_evidence_needs_explicit_overwrite(tmp_path):
    report = tmp_path / "validation.json"
    report.write_text('{"selected_threshold": 0.45}')
    with pytest.raises(FileExistsError, match="--overwrite"):
        validate_report_paths([report], [])
    validate_report_paths([report], [], overwrite=True)
    assert report.read_text() == '{"selected_threshold": 0.45}'


def test_report_cannot_overwrite_a_hard_link_to_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"trained weights")
    output = tmp_path / "report.json"
    output.hardlink_to(checkpoint)
    with pytest.raises(ValueError, match="experiment input"):
        validate_report_paths([output], [checkpoint], overwrite=True)
    assert checkpoint.read_bytes() == b"trained weights"


def test_report_and_cache_cannot_be_hard_links(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("previous report")
    cache = tmp_path / "cache.json"
    cache.hardlink_to(report)
    with pytest.raises(ValueError, match="different output files"):
        validate_report_paths([report, cache], [], overwrite=True)
