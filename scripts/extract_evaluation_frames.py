"""Extract uniformly spaced frames before running or inspecting a detector.

The video is decoded twice: first to count real decoded frames, then to save
fixed midpoint samples. No model is loaded and no annotation files are created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import cv2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def midpoint_indices(frame_count: int, samples: int) -> list[int]:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise ValueError("video must contain at least one decoded frame")
    count = min(samples, frame_count)
    # Integer arithmetic implements floor((i + 0.5) * N / count).
    return [(2 * index + 1) * frame_count // (2 * count) for index in range(count)]


def extract_frames(
    video: Path, source_id: str, output_dir: Path, samples: int = 12
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", source_id):
        raise ValueError(
            "source-id must contain 1–64 letters, digits, underscores or hyphens, starting with a letter or digit"
        )
    midpoint_indices(1, samples)  # Validate the request before opening the video.
    video, output_dir = video.expanduser().resolve(), output_dir.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if output_dir.exists():
        raise ValueError(
            f"Output already exists; choose a fresh directory: {output_dir}"
        )
    original_hash = file_sha256(video)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video must report a positive finite FPS")
        reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded_count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded_count += 1
    finally:
        capture.release()
    indices = midpoint_indices(decoded_count, samples)
    if file_sha256(video) != original_hash:
        raise ValueError("Video changed during counting; no output was created")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)
    selected = set(indices)
    frames = []
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot reopen video for sampling")
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in selected:
                filename = f"{source_id}_{frame_index:08d}.png"
                image_path = image_dir / filename
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"Could not save {image_path}")
                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                frames.append(
                    {
                        "image": f"images/{filename}",
                        "image_sha256": file_sha256(image_path),
                        "frame_index_zero_based": frame_index,
                        "nominal_timestamp_seconds": frame_index / fps,
                        "decoder_reported_timestamp_ms": timestamp_ms
                        if math.isfinite(timestamp_ms) and timestamp_ms >= 0
                        else None,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
            frame_index += 1
    finally:
        capture.release()
    if frame_index != decoded_count or len(frames) != len(indices):
        raise ValueError(
            "Decode changed between passes; partial output must not be used for evaluation"
        )
    if file_sha256(video) != original_hash:
        raise ValueError(
            "Video changed while sampling; partial output must not be used for evaluation"
        )
    manifest = {
        "schema_version": 1,
        "source_id": source_id,
        "video_path": str(video),
        "video_sha256": original_hash,
        "nominal_fps": fps,
        "reported_frame_count": reported_count
        if math.isfinite(reported_count)
        else None,
        "decoded_frame_count": decoded_count,
        "requested_samples": samples,
        "saved_samples": len(frames),
        "sampling_rule": "Zero-based floor((i + 0.5) * decoded_frame_count / min(requested_samples, decoded_frame_count)); no prediction-dependent selection.",
        "timestamp_note": "Nominal times assume constant FPS. Decoder-reported media times may be approximate; neither field is a real-world capture timestamp.",
        "frames": frames,
        "annotation_status": "Unannotated. No label files were created; an absent label is not a background judgment.",
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    try:
        result = extract_frames(
            args.video, args.source_id, args.output_dir, args.samples
        )
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"Saved {result['saved_samples']} unannotated frames from {result['decoded_frame_count']} decoded frames."
    )
    print(f"Manifest: {args.output_dir.resolve() / 'source_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
