"""Compare two completed runs against the fixed pedestrian sample."""

import argparse
import csv
from collections import defaultdict
from itertools import combinations, pairwise
import json
import math
from pathlib import Path


ANNOTATIONS = Path(__file__).with_name("pedestrian_sample.csv")
MODE_NAMES = ("continuous", "gated")
CLIP_SHAPE = (250, 25.0, 1280, 720)
CLIP_SHA256 = "1a5d9ba0917355803a6e7f662daf5f7a372abfe7f56b015c846159b8eea63098"
MIN_IOU = 0.50
PROXIMITY_PX = 150.0
LABEL_MARGIN_PX = 20.0


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def box(row):
    return tuple(float(row[field]) for field in ("x1", "y1", "x2", "y2"))


def iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union if union > 0 else 0.0


def center(bounds):
    return (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2


def match_frame(labels, predictions):
    """Greedily assign one label and one prediction per overlap, largest IoU first."""
    candidates = []
    best = {}
    for label in labels:
        subject = label["subject_id"]
        label_box = box(label)
        for prediction in predictions:
            track = int(prediction["event_id"]), int(prediction["track_id"])
            score = iou(label_box, box(prediction))
            best[subject] = max(best.get(subject, 0), score)
            if score >= MIN_IOU:
                candidates.append((-score, subject, track))
    assignments = {}
    used_tracks = set()
    for negative_score, subject, track in sorted(candidates):
        if subject not in assignments and track not in used_tracks:
            assignments[subject] = {"event_id": track[0], "track_id": track[1], "iou": -negative_score}
            used_tracks.add(track)
    return assignments, best


def read_run(directory, mode, sampled_frames):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    shape = tuple(summary[key] for key in ("frame_count", "fps", "width", "height"))
    if (summary["mode"] != mode or summary["replay"] or shape != CLIP_SHAPE
            or summary["video_sha256"].lower() != CLIP_SHA256):
        raise ValueError(f"{directory} needs a real {mode} run on the full 250-frame clip")
    if summary["config"]["proximity_px"] != PROXIMITY_PX:
        raise ValueError(f"{directory} needs the fixed {PROXIMITY_PX:g} px proximity threshold")

    frame_rows = read_csv(directory / "frames.csv")
    frames = {int(row["frame_id"]): row for row in frame_rows}
    if len(frame_rows) != CLIP_SHAPE[0] or set(frames) != set(range(CLIP_SHAPE[0])):
        raise ValueError(f"{directory} must cover all frame indices in the sampled clip")

    tracks = defaultdict(list)
    for row in read_csv(directory / "movement.csv"):
        frame_id = int(row["frame_id"])
        if frame_id in sampled_frames:
            tracks[frame_id].append(row)
    pairs = {}
    for row in read_csv(directory / "pairs.csv"):
        frame_id = int(row["frame_id"])
        if frame_id in sampled_frames:
            a, b = sorted((int(row["track_id_a"]), int(row["track_id_b"])))
            close = row["is_close"].strip().lower()
            if row["distance_px"] and close in {"true", "false", "1", "0"}:
                pairs[(frame_id, int(row["event_id"]), a, b)] = close in {"true", "1"}
    return summary, frames, tracks, pairs


def proximity_checks(labels, matches, predictions):
    counts = dict(eligible=0, uncertain=0, expected_close=0, expected_far=0,
                  available=0, available_close=0, available_far=0, agreed=0,
                  agreed_close=0, agreed_far=0)
    for frame_id, rows in labels.items():
        for first, second in combinations(rows, 2):
            distance = math.dist(center(box(first)), center(box(second)))
            if abs(distance - PROXIMITY_PX) < LABEL_MARGIN_PX:
                counts["uncertain"] += 1
                continue
            kind = "close" if distance <= PROXIMITY_PX else "far"
            counts["eligible"] += 1
            counts[f"expected_{kind}"] += 1
            a = matches.get((frame_id, first["subject_id"]))
            b = matches.get((frame_id, second["subject_id"]))
            if a is None or b is None or a["event_id"] != b["event_id"]:
                continue
            low, high = sorted((a["track_id"], b["track_id"]))
            value = predictions.get((frame_id, a["event_id"], low, high))
            if value is not None:
                counts["available"] += 1
                counts[f"available_{kind}"] += 1
                if value == (kind == "close"):
                    counts["agreed"] += 1
                    counts[f"agreed_{kind}"] += 1
    return counts


def track_at(matches, index):
    row = matches[index]
    return row["event_id"], row["track_id"]


def evaluate(continuous, gated):
    labels = defaultdict(list)
    sequences = defaultdict(list)
    for row in read_csv(ANNOTATIONS):
        frame_id = int(row["frame_id"])
        label = (frame_id, row["subject_id"])
        labels[frame_id].append(row)
        sequences[(row["window"], row["subject_id"])].append(label)

    result = {"label_rows": sum(map(len, labels.values())), "sample_frames": len(labels), "runs": {}}
    matches = {}
    both_active = set(labels)
    occluded = {(int(row["frame_id"]), row["subject_id"])
                for rows in labels.values() for row in rows if row["occluded"] == "yes"}
    for mode, directory in zip(MODE_NAMES, (continuous, gated), strict=True):
        summary, frames, tracks, pairs = read_run(directory, mode, labels)
        found = {}
        misses = []
        for frame_id, rows in labels.items():
            assignments, best = match_frame(rows, tracks.get(frame_id, []))
            found.update({(frame_id, subject): assignment for subject, assignment in assignments.items()})
            for row in rows:
                subject = row["subject_id"]
                if subject not in assignments:
                    misses.append({
                        "frame_id": frame_id,
                        "time_s": frame_id / summary["fps"],
                        "subject_id": subject,
                        "occluded": row["occluded"] == "yes",
                        "active": bool(frames[frame_id]["event_id"]),
                        "best_iou": round(best.get(subject, 0.0), 4),
                    })

        matches[mode] = found
        both_active &= {frame_id for frame_id, row in frames.items() if row["event_id"]}
        result["runs"][mode] = {
            "summary": summary,
            "matched_labels": len(found),
            "matched_occluded_labels": len(occluded & found.keys()),
            "missed_labels": misses,
            "trigger_frames_for_review": [i for i, row in frames.items() if float(row["trigger_count"] or 0) > 0],
            "proximity": proximity_checks(labels, found, pairs),
            "matched_tracks": [
                {"frame_id": frame_id, "subject_id": subject, **assignment}
                for (frame_id, subject), assignment in sorted(found.items())
            ],
        }

    joint_matches = matches["continuous"].keys() & matches["gated"].keys()
    sampled_transitions = [transition for sequence in sequences.values()
                           for transition in pairwise(sorted(sequence))]
    joint_transitions = [(a, b) for a, b in sampled_transitions if a in joint_matches and b in joint_matches]
    result.update({
        "occluded_labels": len(occluded),
        "joint_active_frames": len(both_active),
        "joint_active_labels": sum(len(labels[frame_id]) for frame_id in both_active),
        "joint_matches": len(joint_matches),
        "sampled_transitions": len(sampled_transitions),
        "joint_matched_transitions": len(joint_transitions),
    })
    for mode in MODE_NAMES:
        found = matches[mode]
        run = result["runs"][mode]
        run["matched_joint_active_labels"] = sum(frame_id in both_active for frame_id, _ in found)
        run["sampled_id_changes"] = sum(track_at(found, a) != track_at(found, b) for a, b in joint_transitions)
        run["sampled_lost_transitions"] = sum(a in found and b not in found for a, b in sampled_transitions)
        run["sampled_recovered_transitions"] = sum(a not in found and b in found for a, b in sampled_transitions)
        run["id_change_examples"] = [
            {"subject_id": a[1], "from_frame": a[0], "to_frame": b[0],
             "from_track": track_at(found, a), "to_track": track_at(found, b)}
            for a, b in joint_transitions if track_at(found, a) != track_at(found, b)
        ]
    baseline_config, gated_config = (result["runs"][mode]["summary"]["config"] for mode in MODE_NAMES)
    if any(baseline_config[key] != gated_config[key] for key in (
        "device", "person_confidence", "track_activation_threshold", "track_min_consecutive_frames",
        "max_speed_px_s", "distance_change_px", "person_weights",
    )):
        raise ValueError("runs must use the same person model, device, tracker and movement settings")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", required=True, type=Path, help="Continuous run output directory")
    parser.add_argument("--gated", required=True, type=Path, help="Detector-gated run output directory")
    parser.add_argument("--output", type=Path, help="Write JSON report here, or print to stdout if omitted")
    args = parser.parse_args()
    try:
        payload = json.dumps(evaluate(args.continuous, args.gated), indent=2) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.write_text(payload, encoding="utf-8")
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
