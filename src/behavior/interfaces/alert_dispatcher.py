"""Create and log alert events from rolling-window model predictions."""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ViolenceClassifier.AlertDispatcher")


@dataclass
class AlertEvent:
    event_id: str
    timestamp: float
    window_start: float
    window_end: float
    violence_probability: float
    alert_level: str  # 'NORMAL', 'MONITOR', 'ALARM'
    status: str  # 'active', 'cleared'
    verification_status: str  # 'pending', 'confirmed_violence', 'false_alarm'
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertDispatcher:
    """
    Apply thresholds and cooldowns before publishing model alerts.

    Hysteresis & Cooldown:
    - Prevents alert spam by enforcing consecutive positive clips before raising ALARM.
    - Implements a cooldown period between alarms.
    - Logs events for post-incident auditing and false positive analysis.
    """

    def __init__(
        self,
        threshold_monitor: float = 0.50,
        threshold_alarm: float = 0.75,
        required_consecutive_triggers: int = 2,
        cooldown_seconds: float = 5.0,
        log_file: Optional[str] = "./outputs/behavior/logs/alerts_audit.jsonl",
    ):
        self.thresh_monitor = threshold_monitor
        self.thresh_alarm = threshold_alarm
        self.consecutive_required = required_consecutive_triggers
        self.cooldown_sec = cooldown_seconds
        self.log_file = log_file

        self.consecutive_count = 0
        self.last_alarm_time = -999.0
        self.current_state = "NORMAL"
        self.subscribers: List[Callable[[AlertEvent], None]] = []

        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def subscribe(self, callback: Callable[[AlertEvent], None]) -> None:
        """Register a callback for alert notifications."""
        self.subscribers.append(callback)

    def process_prediction(self, pred: Dict[str, Any]) -> AlertEvent:
        """
        Evaluates a prediction dictionary from BehaviorClassifier and returns an AlertEvent.
        """
        prob = pred.get("violence_probability", 0.0)
        ts = pred.get("timestamp", time.time())
        w_start = pred.get("window_start", ts)
        w_end = pred.get("window_end", ts)

        # Check thresholds
        if prob >= self.thresh_alarm:
            self.consecutive_count += 1
        elif prob >= self.thresh_monitor:
            # Maintain count or increment cautiously
            self.consecutive_count = max(1, self.consecutive_count)
        else:
            self.consecutive_count = 0

        # Determine state
        time_since_last_alarm = ts - self.last_alarm_time
        if (
            self.consecutive_count >= self.consecutive_required
            and time_since_last_alarm > self.cooldown_sec
        ):
            new_state = "ALARM"
            self.last_alarm_time = ts
        elif prob >= self.thresh_monitor:
            new_state = "MONITOR"
        else:
            new_state = "NORMAL"

        self.current_state = new_state
        event_id = f"evt_{int(ts * 1000)}"

        event = AlertEvent(
            event_id=event_id,
            timestamp=ts,
            window_start=w_start,
            window_end=w_end,
            violence_probability=prob,
            alert_level=new_state,
            status="active" if new_state == "ALARM" else "normal",
            verification_status="pending",
            details={
                "consecutive_triggers": self.consecutive_count,
                "model_status": pred.get("model_status", "valid"),
                "inference_latency_ms": pred.get("inference_latency_ms", 0.0),
            },
        )

        # Dispatch to subscribers
        if new_state in ("MONITOR", "ALARM"):
            for sub in self.subscribers:
                try:
                    sub(event)
                except Exception as e:
                    logger.error("Error in subscriber callback: %s", e)

            # Audit log
            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")

        return event

    def log_operator_feedback(self, event_id: str, label: str, notes: str = "") -> None:
        """
        Record an operator label and optional notes for a past alert.
        """
        feedback_entry = {
            "feedback_timestamp": time.time(),
            "event_id": event_id,
            "operator_label": label,  # 'confirmed_violence', 'false_alarm_sports', 'false_alarm_crowd', etc.
            "notes": notes,
        }
        feedback_file = "./outputs/behavior/logs/operator_feedback.jsonl"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        logger.info("Recorded operator feedback for event %s: %s", event_id, label)
