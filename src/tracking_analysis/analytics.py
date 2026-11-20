from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, groupby
from math import atan2, degrees, hypot, isfinite
from numbers import Integral


MOVEMENT_FIELDS = (
    "event_id", "frame_id", "timestamp_s", "track_id",
    "x1", "y1", "x2", "y2", "center_x_px", "center_y_px",
    "raw_speed_px_s", "speed_px_s", "acceleration_px_s2",
    "body_heights_per_s", "motion_band", "abrupt_turn",
    "vx_px_s", "vy_px_s", "direction_deg", "direction_change_deg",
    "movement_status",
)
PAIR_FIELDS = (
    "event_id", "frame_id", "timestamp_s", "track_id_a", "track_id_b",
    "distance_px", "distance_change_px", "interaction_status", "is_close",
    "possible_following",
)
PROXIMITY_FIELDS = (
    "event_id", "track_id_a", "track_id_b", "proximity_episode_id",
    "start_frame", "end_frame", "start_time_s", "end_time_s",
    "frames_close", "duration_s",
)


@dataclass
class _Observation:
    frame_id: int
    timestamp_s: float
    x: float
    y: float
    direction_deg: float | None = None
    speed_px_s: float | None = None
    height_px: float | None = None


@dataclass
class _TrackState:
    last_observed: _Observation
    last_accepted: _Observation


class MovementAnalyzer:
    """Measure image-plane motion within each event and track."""

    def __init__(self, max_speed_px_s: float = 600):
        if not isfinite(max_speed_px_s) or max_speed_px_s < 0:
            raise ValueError("The speed threshold must be finite and nonnegative.")
        self.max_speed_px_s = max_speed_px_s
        self._history: dict[tuple[int, int], _TrackState] = {}

    def observe(
        self,
        event_id: int,
        frame_id: int,
        timestamp_s: float,
        track_id: int,
        bbox: Sequence[float],
    ) -> dict:
        for value in (event_id, frame_id, track_id):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError("Event, frame, and track IDs must be nonnegative integers.")
        if not isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("The timestamp must be finite and nonnegative.")

        x1, y1, x2, y2 = map(float, bbox)
        if not all(map(isfinite, (x1, y1, x2, y2))) or x1 > x2 or y1 > y2:
            raise ValueError("The bounding box must have finite, ordered coordinates.")
        current = _Observation(
            frame_id,
            timestamp_s,
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            height_px=max(1.0, y2 - y1),
        )
        row = {
            "event_id": event_id,
            "frame_id": frame_id,
            "timestamp_s": timestamp_s,
            "track_id": track_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x_px": current.x,
            "center_y_px": current.y,
            "raw_speed_px_s": None,
            "speed_px_s": None,
            "acceleration_px_s2": None,
            "body_heights_per_s": None,
            "motion_band": "unknown",
            "abrupt_turn": False,
            "vx_px_s": None,
            "vy_px_s": None,
            "direction_deg": None,
            "direction_change_deg": None,
            "movement_status": "initial",
        }

        key = (event_id, track_id)
        state = self._history.get(key)
        if state is None:
            self._history[key] = _TrackState(current, current)
            return row

        previous = state.last_observed
        dt = timestamp_s - previous.timestamp_s
        if frame_id <= previous.frame_id or dt <= 0:
            raise ValueError("Frame IDs and timestamps must increase within each track.")

        if frame_id != previous.frame_id + 1:
            row["movement_status"] = "gap"
        else:
            vx = (current.x - previous.x) / dt
            vy = (current.y - previous.y) / dt
            raw_speed = hypot(vx, vy)
            row["raw_speed_px_s"] = raw_speed

            anchor = state.last_accepted
            if previous is not anchor:
                # Reinitialize after a return to the accepted path or a stable new box.
                anchor_speed = hypot(current.x - anchor.x, current.y - anchor.y) / (
                    timestamp_s - anchor.timestamp_s
                )
                row["movement_status"] = "outlier"
                if min(anchor_speed, raw_speed) <= self.max_speed_px_s:
                    row["movement_status"] = "recovered"
            elif raw_speed > self.max_speed_px_s:
                row["movement_status"] = "outlier"
            else:
                current.direction_deg = degrees(atan2(vy, vx)) if raw_speed else None
                current.speed_px_s = raw_speed
                row.update(
                    speed_px_s=raw_speed,
                    vx_px_s=vx,
                    vy_px_s=vy,
                    direction_deg=current.direction_deg,
                    movement_status="ok",
                )
                body_heights_per_s = raw_speed / current.height_px
                row["body_heights_per_s"] = body_heights_per_s
                row["motion_band"] = (
                    "near_stationary"
                    if body_heights_per_s < 0.08
                    else "low_image_motion"
                    if body_heights_per_s < 0.50
                    else "high_image_motion"
                )
                if previous.speed_px_s is not None:
                    row["acceleration_px_s2"] = (raw_speed - previous.speed_px_s) / dt
                if current.direction_deg is not None and previous.direction_deg is not None:
                    row["direction_change_deg"] = (
                        (current.direction_deg - previous.direction_deg + 180) % 360 - 180
                    )
                    row["abrupt_turn"] = (
                        abs(row["direction_change_deg"]) >= 75
                        and body_heights_per_s >= 0.08
                    )

        state.last_observed = current
        if row["movement_status"] != "outlier":
            state.last_accepted = current
        return row


def analyze_interactions(
    movement_rows: Iterable[dict],
    fps: float,
    proximity_px: float = 150,
    distance_change_px: float = 5,
) -> tuple[list[dict], list[dict]]:
    """Compare each unordered pair of tracks in adjacent frames of one event."""
    if not isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive.")
    if any(not isfinite(value) or value < 0 for value in (proximity_px, distance_change_px)):
        raise ValueError("Distance thresholds must be finite and nonnegative.")

    def event_and_frame(row: dict) -> tuple[int, int]:
        return row["event_id"], row["frame_id"]

    rows = sorted(movement_rows, key=event_and_frame)
    pair_rows = []
    episodes = []
    previous_pairs = {}
    latest_episodes = {}

    for (event_id, frame_id), frame_rows in groupby(rows, key=event_and_frame):
        people = sorted(frame_rows, key=lambda row: row["track_id"])
        track_ids = {person["track_id"] for person in people}
        if len(track_ids) != len(people):
            raise ValueError(f"Duplicate track ID in event {event_id}, frame {frame_id}.")

        timestamp_s = people[0]["timestamp_s"]
        if any(person["timestamp_s"] != timestamp_s for person in people):
            raise ValueError(f"Inconsistent timestamps in event {event_id}, frame {frame_id}.")

        for person_a, person_b in combinations(people, 2):
            track_a, track_b = person_a["track_id"], person_b["track_id"]
            key = (event_id, track_a, track_b)
            distance = None
            if "outlier" not in (person_a["movement_status"], person_b["movement_status"]):
                distance = hypot(
                    person_b["center_x_px"] - person_a["center_x_px"],
                    person_b["center_y_px"] - person_a["center_y_px"],
                )

            previous = previous_pairs.get(key)
            change = None
            status = "insufficient_data"
            if (
                distance is not None
                and previous is not None
                and previous["frame_id"] + 1 == frame_id
                and previous["distance_px"] is not None
            ):
                change = distance - previous["distance_px"]
                if change < -distance_change_px:
                    status = "approaching"
                elif change > distance_change_px:
                    status = "moving_apart"
                else:
                    status = "stable"

            close = None if distance is None else distance <= proximity_px
            possible_following = False
            speed_a, speed_b = person_a["speed_px_s"], person_b["speed_px_s"]
            if (
                distance is not None
                and distance <= proximity_px * 2
                and change is not None
                and abs(change) <= distance_change_px * 2
                and speed_a is not None
                and speed_b is not None
                and speed_a >= 10
                and speed_b >= 10
            ):
                dot = (
                    person_a["vx_px_s"] * person_b["vx_px_s"]
                    + person_a["vy_px_s"] * person_b["vy_px_s"]
                )
                possible_following = dot / (speed_a * speed_b) >= 0.85
            pair = {
                "event_id": event_id,
                "frame_id": frame_id,
                "timestamp_s": timestamp_s,
                "track_id_a": track_a,
                "track_id_b": track_b,
                "distance_px": distance,
                "distance_change_px": change,
                "interaction_status": status,
                "is_close": close,
                "possible_following": possible_following,
            }
            pair_rows.append(pair)
            previous_pairs[key] = pair

            if close:
                episode = latest_episodes.get(key)
                if episode is None or episode["end_frame"] + 1 != frame_id:
                    episode = {
                        "event_id": event_id,
                        "track_id_a": track_a,
                        "track_id_b": track_b,
                        "proximity_episode_id": (
                            1 if episode is None else episode["proximity_episode_id"] + 1
                        ),
                        "start_frame": frame_id,
                        "end_frame": frame_id,
                        "start_time_s": timestamp_s,
                        "end_time_s": timestamp_s,
                        "frames_close": 1,
                        "duration_s": 1 / fps,
                    }
                    episodes.append(episode)
                    latest_episodes[key] = episode
                else:
                    episode["end_frame"] = frame_id
                    episode["end_time_s"] = timestamp_s
                    episode["frames_close"] += 1
                    episode["duration_s"] = episode["frames_close"] / fps

    return pair_rows, episodes
