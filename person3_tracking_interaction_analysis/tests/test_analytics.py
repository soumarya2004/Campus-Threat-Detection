import pytest

from src.tracking_analysis.analytics import (
    MOVEMENT_FIELDS,
    PAIR_FIELDS,
    PROXIMITY_FIELDS,
    MovementAnalyzer,
    analyze_interactions,
)


def observe(analyzer, event_id, frame_id, track_id, x, y=0):
    return analyzer.observe(
        event_id, frame_id, frame_id / 20, track_id, (x - 1, y - 1, x + 1, y + 1)
    )


def test_speed_heading_gap_and_scoped_id():
    analyzer = MovementAnalyzer()

    first = observe(analyzer, 1, 0, 0, 50)
    east = observe(analyzer, 1, 1, 0, 53)
    down = observe(analyzer, 1, 2, 0, 53, 4)
    gap = observe(analyzer, 1, 4, 0, 53, 8)
    next_event = observe(analyzer, 2, 5, 0, 500)

    assert tuple(first) == MOVEMENT_FIELDS
    assert (first["x1"], first["y1"], first["x2"], first["y2"]) == (49, -1, 51, 1)
    assert first["movement_status"] == "initial"
    assert east["speed_px_s"] == pytest.approx(60)
    assert east["direction_deg"] == 0
    assert down["speed_px_s"] == pytest.approx(80)
    assert down["direction_deg"] == 90
    assert down["direction_change_deg"] == 90
    assert gap["movement_status"] == "gap"
    assert gap["raw_speed_px_s"] is None
    assert next_event["movement_status"] == "initial"
    assert next_event["speed_px_s"] is None


def test_stationary_track_has_speed_without_heading():
    analyzer = MovementAnalyzer()
    observe(analyzer, 1, 0, 0, 0)
    still = observe(analyzer, 1, 1, 0, 0)

    assert still["speed_px_s"] == 0
    assert still["direction_deg"] is None


def test_canonical_pairs_split_at_dropped_frames_and_events():
    analyzer = MovementAnalyzer()
    rows = [
        observe(analyzer, 1, 0, 0, 0), observe(analyzer, 1, 0, 1, 20),
        observe(analyzer, 1, 1, 1, 14), observe(analyzer, 1, 1, 0, 0),
        observe(analyzer, 1, 3, 0, 0), observe(analyzer, 1, 3, 1, 8),
        observe(analyzer, 2, 4, 1, 2), observe(analyzer, 2, 4, 0, 0),
    ]
    pairs, episodes = analyze_interactions(reversed(rows), fps=20, proximity_px=20)

    assert tuple(pairs[0]) == PAIR_FIELDS
    assert all((pair["track_id_a"], pair["track_id_b"]) == (0, 1) for pair in pairs)
    assert [pair["interaction_status"] for pair in pairs] == [
        "insufficient_data", "approaching", "insufficient_data", "insufficient_data",
    ]
    assert [pair["distance_change_px"] for pair in pairs] == [None, -6, None, None]
    assert tuple(episodes[0]) == PROXIMITY_FIELDS
    assert [episode["event_id"] for episode in episodes] == [1, 1, 2]
    assert [episode["proximity_episode_id"] for episode in episodes] == [1, 2, 1]
    assert [episode["frames_close"] for episode in episodes] == [2, 1, 1]
    assert [episode["duration_s"] for episode in episodes] == [0.1, 0.05, 0.05]


def test_outlier_splits_pair_history_and_movement_recovers():
    analyzer = MovementAnalyzer(max_speed_px_s=600)
    rows = []
    for frame_id, x in enumerate((10, 11, 100, 12, 13, 14)):
        rows.append(observe(analyzer, 1, frame_id, 0, x))
        rows.append(observe(analyzer, 1, frame_id, 1, 16))

    jump, recovery, next_motion = rows[4:9:2]
    assert jump["movement_status"] == "outlier"
    assert jump["raw_speed_px_s"] == pytest.approx(1780)
    assert jump["speed_px_s"] is None
    assert recovery["movement_status"] == "recovered"
    assert recovery["speed_px_s"] is None
    assert next_motion["speed_px_s"] == pytest.approx(20)
    assert next_motion["direction_change_deg"] is None

    pairs, episodes = analyze_interactions(rows, fps=20, proximity_px=10)

    assert pairs[2]["distance_px"] is None
    assert pairs[2]["is_close"] is None
    assert pairs[2]["interaction_status"] == "insufficient_data"
    assert pairs[3]["distance_change_px"] is None
    assert pairs[4]["interaction_status"] == "stable"
    assert [episode["frames_close"] for episode in episodes] == [2, 3]


def test_persistent_new_position_initializes_fresh_heading():
    analyzer = MovementAnalyzer()
    observe(analyzer, 1, 0, 0, 0)
    assert observe(analyzer, 1, 1, 0, 200)["movement_status"] == "outlier"
    assert observe(analyzer, 1, 2, 0, 201)["movement_status"] == "recovered"
    resumed = observe(analyzer, 1, 3, 0, 201, 1)

    assert resumed["speed_px_s"] == pytest.approx(20)
    assert resumed["direction_deg"] == 90
    assert resumed["direction_change_deg"] is None


def test_duplicate_rejected_and_single_person_cannot_make_self_pair():
    analyzer = MovementAnalyzer()
    only = observe(analyzer, 1, 0, 0, 0)

    assert analyze_interactions([], fps=20) == ([], [])
    assert analyze_interactions([only], fps=20) == ([], [])
    with pytest.raises(ValueError, match="Duplicate track ID"):
        analyze_interactions([only, only], fps=20)
    with pytest.raises(ValueError, match="increase"):
        observe(analyzer, 1, 0, 0, 0)


@pytest.mark.parametrize("fps", [0, -20, float("inf")])
def test_reject_invalid_frame_rates(fps):
    with pytest.raises(ValueError, match="FPS"):
        analyze_interactions([], fps=fps)
