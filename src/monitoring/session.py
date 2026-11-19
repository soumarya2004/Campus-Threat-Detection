"""File-based bridge between inference producers and the Streamlit viewer.

The dashboard reads these files and never loads or runs detection models.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

SCHEMA_VERSION = 1
ASSESSMENT_LEVELS = {"NORMAL", "SUSPICIOUS", "HIGH_RISK", "UNASSESSED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        for attempt in range(7):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _safe_child(root: Path, relative: str | None) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


class DashboardSessionWriter:
    """Publish processed frames, status and upstream-assessed events."""

    def __init__(self, directory: str | Path, *, metadata: dict | None = None):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.status_path = self.directory / "status.json"
        self.events_path = self.directory / "events.jsonl"
        if any(self.directory.iterdir()):
            raise FileExistsError(
                f"Dashboard session folder is not empty: {self.directory}. "
                "Choose a new empty folder for each run."
            )
        self.events_path.touch(exist_ok=False)
        self._lock = threading.Lock()
        now = utc_now()
        self._state = {
            "schema_version": SCHEMA_VERSION,
            "run_id": uuid4().hex,
            "status": "starting",
            "created_at_utc": now,
            "updated_at_utc": now,
            "metadata": metadata or {},
            "capture": {"status": "unknown"},
            "models": {},
            "performance": {},
            "current_frame": None,
            "assessment": {
                "level": "UNASSESSED",
                "reason": "No system assessment has been supplied.",
            },
            "events_file": "events.jsonl",
        }
        self._write_state()

    def _write_state(self) -> None:
        _atomic_write(self.status_path, _json_bytes(self._state))

    def publish_frame(
        self,
        frame: np.ndarray | None,
        *,
        capture: dict,
        models: dict,
        performance: dict,
        current_frame: dict,
        assessment: dict | None = None,
    ) -> None:
        """Write the newest annotated frame and its status snapshot."""
        if not all(
            isinstance(value, dict)
            for value in (capture, models, performance, current_frame)
        ):
            raise TypeError(
                "capture, models, performance and current_frame must be mappings"
            )

        assessment = assessment or {
            "level": "UNASSESSED",
            "reason": "No system assessment has been supplied.",
        }
        level = assessment.get("level", "UNASSESSED")
        if level not in ASSESSMENT_LEVELS:
            raise ValueError(f"Unknown assessment level: {level!r}")

        image_name = "latest.jpg" if (self.directory / "latest.jpg").is_file() else None
        if frame is not None:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                raise ValueError("dashboard frames must be uint8 OpenCV images")
            import cv2

            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise OSError("Could not encode the latest dashboard frame")
            _atomic_write(self.directory / "latest.jpg", encoded.tobytes())
            image_name = "latest.jpg"

        with self._lock:
            now = utc_now()
            self._state.update(
                status="running",
                updated_at_utc=now,
                capture=capture,
                models=models,
                performance=performance,
                current_frame={
                    **current_frame,
                    "updated_at_utc": now,
                    "image": image_name,
                },
                assessment=assessment,
            )
            self._write_state()

    def record_event(self, event: dict) -> None:
        """Append an event produced by the upstream assessment policy."""
        if not isinstance(event, dict):
            raise TypeError("event must be a mapping")
        record = dict(event)
        level = record.get("level", "UNASSESSED")
        if level not in ASSESSMENT_LEVELS:
            raise ValueError(f"Unknown event level: {level!r}")
        record.setdefault("event_id", uuid4().hex)
        record.setdefault("recorded_at_utc", utc_now())
        record.setdefault("evidence_categories", [])
        if not isinstance(record["evidence_categories"], list):
            raise ValueError("evidence_categories must be a list")
        for name in ("start_s", "end_s"):
            value = record.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number or None")
        if (
            record.get("start_s") is not None
            and record.get("end_s") is not None
            and record["end_s"] < record["start_s"]
        ):
            raise ValueError("event end_s must be greater than or equal to start_s")
        encoded = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
        with (
            self._lock,
            self.events_path.open("a", encoding="utf-8", newline="") as handle,
        ):
            handle.write(encoded)
            handle.flush()

    def close(self, *, status: str = "stopped", error: str | None = None) -> None:
        if status not in {"stopped", "error"}:
            raise ValueError(f"Invalid final session status: {status!r}")
        with self._lock:
            self._state["status"] = status
            self._state["updated_at_utc"] = utc_now()
            if error:
                self._state["error"] = str(error)
            self._write_state()

    def __enter__(self) -> DashboardSessionWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(
            status="error" if exc_type else "stopped",
            error=str(exc_value) if exc_value else None,
        )


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"Missing {path.name}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not read {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} must contain a JSON object"
    return value, None


def _read_event_tail(
    path: Path, *, max_events: int = 1000, max_bytes: int = 2_000_000
) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], ["Event log is unavailable"]
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - max_bytes)
            handle.seek(start)
            data = handle.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [], [f"Could not read event log: {exc}"]
    if start:
        lines = lines[1:]
    events, warnings = [], []
    for line_number, line in enumerate(lines[-max_events:], 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
            else:
                warnings.append(f"Skipped non-object event at tail line {line_number}")
        except json.JSONDecodeError:
            warnings.append(f"Skipped malformed event at tail line {line_number}")
    return events, warnings


def load_dashboard_session(directory: str | Path) -> dict:
    """Read a dashboard session without executing models or following image paths."""
    root = Path(directory).expanduser().resolve()
    warnings: list[str] = []
    state, state_warning = _read_json(root / "status.json")
    if state_warning and (root / "status.json").exists():
        warnings.append(state_warning)

    events, event_warnings = _read_event_tail(root / "events.jsonl")
    warnings.extend(event_warnings)

    report = None
    for candidate in (root / "evaluation" / "report.json", root / "report.json"):
        if candidate.is_file():
            report, report_warning = _read_json(candidate)
            if report_warning:
                warnings.append(report_warning)
            break

    image_path = None
    if state:
        current = state.get("current_frame")
        if isinstance(current, dict):
            image_path = _safe_child(root, current.get("image"))
            if image_path is not None and not image_path.is_file():
                warnings.append("Latest frame image is missing")
                image_path = None

    return {
        "directory": root,
        "status": state,
        "events": events,
        "events_available": (root / "events.jsonl").is_file(),
        "evaluation": report,
        "latest_image": image_path,
        "warnings": warnings,
    }
