"""
Synthetic video dataset generator for offline verification, unit tests, and pipeline testing.
Generates controlled video sequences with distinct motion dynamics:
- Violent/Combat: High-velocity, chaotic, overlapping, high-entropy motion.
- Normal/Campus: Smooth laminar pedestrian motion, stationary figures.
- Sports/Activity: Fast rhythmic motion (e.g. bouncing sports ball, running).
- Camera Shake: Global frame jitter simulating wind or pole vibration.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("ViolenceClassifier.Synthetic")


def generate_synthetic_video(
    output_path: str,
    duration_seconds: float = 3.0,
    fps: int = 30,
    resolution: Tuple[int, int] = (224, 224),
    behavior_type: str = "normal",
    noise_level: float = 0.05,
) -> str:
    """
    Renders and encodes a synthetic video sequence simulating different campus behaviors.

    Args:
        output_path: Path where MP4 video will be written.
        duration_seconds: Video duration in seconds.
        fps: Frames per second.
        resolution: (width, height)
        behavior_type: 'normal', 'violence', 'sports', 'camera_shake'
        noise_level: Background sensor noise level.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = resolution
    total_frames = int(duration_seconds * fps)

    # Use cv2.VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))

    # Base background: simulated campus courtyard (gradient background)
    base_bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(80 + 60 * (y / h))
        base_bg[y, :, :] = [val, val, val + 10]

    # Motion agents
    stable_seed = int.from_bytes(
        hashlib.sha256(str(Path(output_path).resolve()).encode("utf-8")).digest()[:4],
        byteorder="big",
    )
    rng = np.random.RandomState(stable_seed)

    # Initial positions for 2 entities
    p1 = np.array([w * 0.35, h * 0.5])
    p2 = np.array([w * 0.65, h * 0.5])

    for frame_idx in range(total_frames):
        frame = base_bg.copy()

        # Add Gaussian sensor noise
        if noise_level > 0:
            noise = (rng.randn(h, w, 3) * (noise_level * 255)).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Global camera shake behavior
        dx_shake, dy_shake = 0, 0
        if behavior_type == "camera_shake":
            dx_shake = int(rng.uniform(-6, 6))
            dy_shake = int(rng.uniform(-6, 6))

        if behavior_type == "violence":
            # Chaotic, sudden convergent and divergent motions with rapid limb flailing
            v1 = rng.uniform(-18, 18, size=2)
            v2 = rng.uniform(-18, 18, size=2)
            # Pull towards each other (collision/altercation)
            center = np.array([w * 0.5, h * 0.5])
            p1 += 0.3 * (center - p1) + v1
            p2 += 0.3 * (center - p2) + v2

            p1 = np.clip(p1, [25, 25], [w - 25, h - 25])
            p2 = np.clip(p2, [25, 25], [w - 25, h - 25])

            # Draw entities with rapid irregular postures
            c1 = (int(p1[0]) + dx_shake, int(p1[1]) + dy_shake)
            c2 = (int(p2[0]) + dx_shake, int(p2[1]) + dy_shake)

            cv2.circle(frame, c1, 14, (0, 0, 200), -1)  # Entity 1 torso
            cv2.circle(frame, c2, 14, (200, 0, 0), -1)  # Entity 2 torso

            # Fast flailing limbs
            limb1_end = (
                int(c1[0] + rng.uniform(-25, 25)),
                int(c1[1] + rng.uniform(-25, 25)),
            )
            limb2_end = (
                int(c2[0] + rng.uniform(-25, 25)),
                int(c2[1] + rng.uniform(-25, 25)),
            )
            cv2.line(frame, c1, limb1_end, (0, 0, 255), 4)
            cv2.line(frame, c2, limb2_end, (255, 0, 0), 4)

        elif behavior_type == "sports":
            # Fast rhythmic trajectory with a bouncing ball
            t = frame_idx / fps
            ball_x = int(
                (w * 0.2 + (w * 0.6) * (0.5 + 0.5 * np.sin(2 * np.pi * t * 1.5)))
                + dx_shake
            )
            ball_y = int(
                (h * 0.3 + abs(np.sin(2 * np.pi * t * 3.0)) * (h * 0.4)) + dy_shake
            )
            cv2.circle(frame, (ball_x, ball_y), 9, (255, 200, 0), -1)

            # Players tracking smoothly
            p1[0] = w * 0.3 + np.sin(2 * np.pi * t * 0.8) * 40
            cv2.circle(
                frame,
                (int(p1[0]) + dx_shake, int(p1[1]) + dy_shake),
                14,
                (50, 180, 50),
                -1,
            )

        else:  # Normal pedestrian motion
            # Smooth laminar progression across frame
            t = frame_idx / float(total_frames)
            p1[0] = 30 + t * (w - 60)
            p2[0] = (w - 30) - t * (w - 60)

            c1 = (int(p1[0]) + dx_shake, int(p1[1]) + dy_shake)
            c2 = (int(p2[0]) + dx_shake, int(p2[1]) + dy_shake)

            cv2.circle(frame, c1, 14, (60, 180, 60), -1)
            cv2.circle(frame, c2, 14, (180, 180, 60), -1)

        out.write(frame)

    out.release()
    return output_path


def create_synthetic_benchmark_dataset(
    output_dir: str = "./data/behavior/synthetic",
    num_train_videos: int = 16,
    num_val_videos: int = 6,
    num_test_videos: int = 6,
) -> Dict[str, str]:
    """
    Creates a full self-contained synthetic dataset split into train, val, and test manifests.
    Guarantees strict video-level isolation.
    """
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path("./data/behavior/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    def generate_partition(name: str, count: int) -> List[Dict[str, Any]]:
        part_records = []
        for i in range(count):
            is_violence = i % 2 == 1
            behavior = (
                "violence" if is_violence else ("sports" if i % 4 == 0 else "normal")
            )
            vid_id = f"syn_{name}_{behavior}_{i:03d}"
            vid_path = str(base_dir / f"{vid_id}.mp4")

            # Generate video if not already exists
            if not os.path.exists(vid_path):
                generate_synthetic_video(
                    output_path=vid_path,
                    duration_seconds=2.5,
                    fps=25,
                    behavior_type=behavior,
                )

            label = 1 if is_violence else 0
            category = (
                "Violence_Brawl"
                if is_violence
                else (
                    "Sports_Activity" if behavior == "sports" else "Pedestrian_Walking"
                )
            )

            # Record video metadata
            part_records.append(
                {
                    "source_video_id": vid_id,
                    "video_path": vid_path,
                    "label": label,
                    "category": category,
                    "duration_seconds": 2.5,
                    "fps": 25,
                    "total_frames": int(2.5 * 25),
                }
            )
        return part_records

    train_recs = generate_partition("train", num_train_videos)
    val_recs = generate_partition("val", num_val_videos)
    test_recs = generate_partition("test", num_test_videos)

    train_path = splits_dir / "train_manifest.json"
    val_path = splits_dir / "val_manifest.json"
    test_path = splits_dir / "test_manifest.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_recs, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_recs, f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_recs, f, indent=2)

    logger.info(
        "Generated synthetic manifests: Train=%d, Val=%d, Test=%d",
        len(train_recs),
        len(val_recs),
        len(test_recs),
    )
    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    }


if __name__ == "__main__":
    create_synthetic_benchmark_dataset()
