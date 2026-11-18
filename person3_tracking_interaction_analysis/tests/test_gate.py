import pytest

from src.tracking_analysis.gate import EventGate


def test_inclusive_grace_and_new_event_after_exact_boundary():
    gate = EventGate(fps=25, grace_s=0.8)

    assert gate.update(56, trigger=False) is None
    assert gate.update(57, trigger=True) == 1
    assert gate.update(77, trigger=False) == 1
    assert gate.update(78, trigger=False) is None
    assert gate.update(79, trigger=True) == 2


def test_errors_keep_existing_grace_without_refreshing_it():
    gate = EventGate(fps=20, grace_s=0.1)

    assert gate.update(0, trigger=True, inference_ok=False) is None
    assert gate.update(1, trigger=True) == 1
    assert gate.update(2, trigger=True) == 1
    assert gate.update(4, trigger=True, inference_ok=False) == 1
    assert gate.update(5, trigger=True, inference_ok=False) is None
    assert gate.update(6, trigger=True) == 2


def test_trigger_after_skipped_expiry_starts_new_event():
    gate = EventGate(fps=20, grace_s=0.08)

    assert gate.update(5, trigger=True) == 1
    assert gate.update(7, trigger=True) == 2
    with pytest.raises(ValueError, match="increasing order"):
        gate.update(7, trigger=False)


@pytest.mark.parametrize("fps,grace_s", [(0, 0.8), (25, -0.8), (float("nan"), 0.8)])
def test_reject_invalid_timing(fps, grace_s):
    with pytest.raises(ValueError):
        EventGate(fps=fps, grace_s=grace_s)
