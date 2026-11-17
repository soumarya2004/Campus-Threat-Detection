"""The review must surface mistakes as well as correct predictions."""

from scripts.review_model_errors import select_examples


def example(name, misses=0, false_positives=0, correct=0):
    return {
        "image": name,
        "missed_ground_truth_indices": list(range(misses)),
        "false_positive_indices": list(range(false_positives)),
        "matches": [(i, i) for i in range(correct)],
    }


def test_review_selects_errors_and_breaks_severity_ties_by_filename():
    records = [
        example("success.jpg", correct=4),
        example("z.jpg", misses=2),
        example("a.jpg", misses=2),
        example("fp.jpg", false_positives=3),
        example("minor.jpg", misses=1),
    ]
    selected = select_examples(records, limit=2)
    assert [row["image"] for row in selected["missed"]] == ["a.jpg", "z.jpg"]
    assert [row["image"] for row in selected["false_positive"]] == ["fp.jpg"]
    assert [row["image"] for row in selected["correct"]] == ["success.jpg"]
    assert select_examples(list(reversed(records)), limit=2) == selected
