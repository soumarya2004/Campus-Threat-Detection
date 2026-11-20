"""Conservative, multi-signal scene assessment for the local demo pipeline."""

from __future__ import annotations

import math
from collections import deque


class ThreatAssessmentEngine:
    """Combine detections and temporal signals without claiming actor identity.

    The behavior classifier returns a clip-level score, while item/person
    detections are frame-level observations. A high-risk state therefore
    requires a persistent high-confidence item detection and a positive
    clip-level violence indication. All other individual signals remain
    suspicious evidence for human review.
    """

    _SEVERITY = {"UNASSESSED": -1, "NORMAL": 0, "SUSPICIOUS": 1, "HIGH_RISK": 2}

    def __init__(
        self,
        *,
        high_confidence_threshold: float = 0.60,
        item_history_size: int = 5,
        interaction_history_size: int = 15,
        interaction_required_hits: int = 8,
    ) -> None:
        if not 0 <= high_confidence_threshold <= 1:
            raise ValueError("high_confidence_threshold must be between 0 and 1")
        if item_history_size < 1 or interaction_history_size < 1:
            raise ValueError("Signal history sizes must be positive")
        if not 1 <= interaction_required_hits <= interaction_history_size:
            raise ValueError("interaction_required_hits must fit the history window")

        self.high_confidence_threshold = high_confidence_threshold
        self._item_history: deque[bool] = deque(maxlen=item_history_size)
        self._interaction_history: deque[set[tuple[int, int]]] = deque(
            maxlen=interaction_history_size
        )
        self._interaction_required_hits = interaction_required_hits
        self._episode_active = False
        self._episode_level = "NORMAL"
        self._episode_start_s: float | None = None

    def update(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        item_detections: list[dict],
        behavior: dict | None,
        interaction_rows: list[dict],
        item_model_ok: bool,
        person_model_ok: bool,
    ) -> tuple[dict, dict | None]:
        """Return the current assessment and an optional alert transition."""
        item_present = bool(item_detections)
        high_confidence_item = any(
            isinstance(detection.get("confidence"), (int, float))
            and math.isfinite(float(detection["confidence"]))
            and float(detection["confidence"]) >= self.high_confidence_threshold
            for detection in item_detections
        )
        self._item_history.append(high_confidence_item)

        approaching_pairs = {
            tuple(sorted((int(row["track_id_a"]), int(row["track_id_b"]))))
            for row in interaction_rows
            if row.get("interaction_status") == "approaching"
            and row.get("is_close") is True
        }
        self._interaction_history.append(approaching_pairs)
        pair_hits: dict[tuple[int, int], int] = {}
        for frame_pairs in self._interaction_history:
            for pair in frame_pairs:
                pair_hits[pair] = pair_hits.get(pair, 0) + 1
        persistent_pairs = {
            pair
            for pair, hits in pair_hits.items()
            if hits >= self._interaction_required_hits
        }
        approaching_pair = bool(approaching_pairs)

        valid_behavior = (
            isinstance(behavior, dict)
            and behavior.get("model_status") == "valid"
            and isinstance(behavior.get("violence_probability"), (int, float))
            and math.isfinite(float(behavior["violence_probability"]))
        )
        violence_probability = (
            float(behavior["violence_probability"]) if valid_behavior else None
        )
        violent_activity = bool(
            valid_behavior
            and behavior.get("is_violence", violence_probability >= 0.5)
        )
        persistent_item = sum(self._item_history) >= 2
        persistent_interaction = bool(persistent_pairs)

        categories: list[str] = []
        if item_present:
            categories.append("dangerous_item_detection")
        if persistent_item:
            categories.append("persistent_high_confidence_item_detection")
        if violent_activity:
            categories.append("clip_level_violence_indication")
        if persistent_interaction:
            categories.append("repeated_close_approach")

        if persistent_item and violent_activity:
            level = "HIGH_RISK"
            reason = (
                "A high-confidence dangerous-item detection persisted and the "
                "clip-level behavior model indicated violence. Human review is required."
            )
        elif item_present or violent_activity or persistent_interaction:
            level = "SUSPICIOUS"
            reasons = []
            if item_present:
                reasons.append("a dangerous item was detected")
            if violent_activity:
                reasons.append("the clip-level behavior model indicated violence")
            if persistent_interaction:
                reasons.append("a close approach repeated across frames")
            reason = "; ".join(reasons).capitalize() + ". These are unverified signals for human review."
        else:
            missing = []
            if not item_model_ok:
                missing.append("item detector unavailable")
            if not person_model_ok:
                missing.append("person detector unavailable")
            if not valid_behavior:
                missing.append("behavior model has no valid score yet")
            if missing:
                level = "UNASSESSED"
                reason = "Assessment incomplete: " + "; ".join(missing) + "."
            else:
                level = "NORMAL"
                reason = "No suspicious signals from the available models in this frame."

        assessment = {
            "level": level,
            "reason": reason,
            "frame_id": int(frame_id),
            "timestamp_s": float(timestamp_s),
            "evidence_categories": categories,
            "signals": {
                "item_detected": item_present,
                "high_confidence_item_detected": high_confidence_item,
                "persistent_item_detected": persistent_item,
                "violence_probability": violence_probability,
                "clip_level_violence_indication": violent_activity,
                "approaching_pair": approaching_pair,
                "repeated_close_approach": persistent_interaction,
                "repeated_close_approach_pairs": [
                    list(pair) for pair in sorted(persistent_pairs)
                ],
            },
            "thresholds": {
                "high_confidence_item": self.high_confidence_threshold,
                "item_persistence": "at least 2 of the last 5 frames",
                "interaction_persistence": (
                    f"at least {self._interaction_required_hits} of the last "
                    f"{self._interaction_history.maxlen} frames"
                ),
            },
        }

        event = self._transition_event(level, timestamp_s, categories)
        return assessment, event

    def _transition_event(
        self, level: str, timestamp_s: float, categories: list[str]
    ) -> dict | None:
        if level in {"SUSPICIOUS", "HIGH_RISK"}:
            if not self._episode_active:
                self._episode_active = True
                self._episode_level = level
                self._episode_start_s = timestamp_s
                phase = "opened"
            elif self._SEVERITY[level] > self._SEVERITY[self._episode_level]:
                self._episode_level = level
                phase = "escalated"
            else:
                return None
        elif self._episode_active:
            phase = "resolved" if level == "NORMAL" else "assessment_interrupted"
            self._episode_active = False
            self._episode_level = level
        else:
            return None

        summary = {
            "opened": f"{level.replace('_', ' ')} review signal opened",
            "escalated": f"Review signal escalated to {level.replace('_', ' ')}",
            "resolved": "Review signal cleared by the current assessment",
            "assessment_interrupted": "Assessment became unavailable while a review signal was active",
        }[phase]
        return {
            "level": level,
            "phase": phase,
            "start_s": self._episode_start_s,
            "end_s": timestamp_s,
            "evidence_categories": categories or ["signal_state_change"],
            "summary": summary,
        }
