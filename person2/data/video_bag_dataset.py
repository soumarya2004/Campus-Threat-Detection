"""Video-level weak-supervision dataset for long XD-Violence videos.

Each item is a bag of temporally distributed clips from one source video. The
video label is kept at bag level and is not copied onto every clip.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

KINETICS_MEAN = np.asarray([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.asarray([0.22803, 0.22145, 0.216989], dtype=np.float32)


class VideoBagDataset(Dataset):
    """Load fixed-size clip bags from video-level labels without caching videos."""

    def __init__(
        self,
        manifest_path_or_records: str | Path | list[dict[str, Any]],
        project_root: str | Path = ".",
        clip_length: int = 16,
        sampling_stride: int = 2,
        spatial_resolution: tuple[int, int] = (112, 112),
        clips_per_video: int = 4,
        horizontal_flip_prob: float = 0.5,
        is_training: bool = False,
    ):
        super().__init__()
        if isinstance(manifest_path_or_records, (str, Path)):
            with Path(manifest_path_or_records).open("r", encoding="utf-8") as stream:
                self.records = json.load(stream)
        else:
            self.records = list(manifest_path_or_records)
        self.project_root = Path(project_root).resolve()
        self.clip_length = int(clip_length)
        self.sampling_stride = int(sampling_stride)
        self.target_h, self.target_w = map(int, spatial_resolution)
        self.clips_per_video = int(clips_per_video)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.is_training = bool(is_training)
        if self.clip_length < 2 or self.sampling_stride < 1 or self.clips_per_video < 1:
            raise ValueError(
                "clip_length, sampling_stride and clips_per_video must be positive"
            )
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be between 0 and 1")
        if not self.records:
            raise ValueError("Video manifest is empty")

        missing = []
        for record in self.records:
            if "video_path" not in record or "label" not in record:
                raise ValueError(
                    "Each manifest record must contain video_path and label"
                )
            path = self.resolve_video_path(record["video_path"])
            if not path.is_file():
                missing.append(str(path))
            if int(record["label"]) not in (0, 1):
                raise ValueError(
                    f"Expected binary label 0/1 in manifest record: {record}"
                )
        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} manifest video(s) are missing. Examples:\n{preview}"
            )

        self.samples = self.records
        self.mean = KINETICS_MEAN.reshape(1, 1, 1, 3)
        self.std = KINETICS_STD.reshape(1, 1, 1, 3)

    def resolve_video_path(self, video_path: str) -> Path:
        path = Path(video_path)
        return (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )

    def __len__(self) -> int:
        return len(self.records)

    def _clip_starts(self, total_frames: int) -> list[int]:
        span = (self.clip_length - 1) * self.sampling_stride + 1
        max_start = max(0, total_frames - span)
        starts: list[int] = []
        for segment in range(self.clips_per_video):
            low = int(segment * (max_start + 1) / self.clips_per_video)
            high = max(
                low, int((segment + 1) * (max_start + 1) / self.clips_per_video) - 1
            )
            if self.is_training and high > low:
                starts.append(random.randint(low, high))
            else:
                starts.append((low + high) // 2)
        return starts

    def _read_clip(
        self, capture: cv2.VideoCapture, start: int, total_frames: int
    ) -> np.ndarray:
        span = (self.clip_length - 1) * self.sampling_stride + 1
        if total_frames < span:
            indices = np.linspace(
                0, max(0, total_frames - 1), self.clip_length, dtype=int
            ).tolist()
        else:
            indices = [
                start + i * self.sampling_stride for i in range(self.clip_length)
            ]

        frames = []
        if total_frames >= span:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            for clip_index in range(self.clip_length):
                frame_index = start + clip_index * self.sampling_stride
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise OSError(f"Could not read frame {frame_index} from video")
                frames.append(self._resize_rgb(frame))
                if clip_index < self.clip_length - 1:
                    for _ in range(self.sampling_stride - 1):
                        if not capture.grab():
                            raise OSError(
                                f"Could not advance to frame {frame_index + self.sampling_stride}"
                            )
        else:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise OSError(f"Could not read frame {frame_index} from video")
                frames.append(self._resize_rgb(frame))

        clip = np.stack(frames).astype(np.float32) / 255.0
        if self.is_training and random.random() < self.horizontal_flip_prob:
            clip = clip[:, :, ::-1, :].copy()
        return (clip - self.mean) / self.std

    def _resize_rgb(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(
            frame,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_LINEAR,
        )

    def __getitem__(self, index: int):
        record = self.records[index]
        path = self.resolve_video_path(record["video_path"])
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"OpenCV could not open video: {path}")
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError(f"Video has no readable frames: {path}")
            starts = self._clip_starts(total_frames)
            clips = [self._read_clip(capture, start, total_frames) for start in starts]
        finally:
            capture.release()

        # [K, T, H, W, C] -> [K, C, T, H, W]
        bag_np = np.stack(clips).transpose(0, 4, 1, 2, 3).copy()
        bag = torch.from_numpy(bag_np).float()
        label = torch.tensor(float(record["label"]), dtype=torch.float32)
        meta = {
            "source_video_id": record.get("source_video_id", path.stem),
            "category": record.get("category", "unknown"),
            "video_path": record["video_path"],
        }
        return bag, label, meta


def create_video_bag_dataloader(
    dataset: VideoBagDataset,
    batch_size: int,
    is_training: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> DataLoader:
    sampler = None
    shuffle = is_training
    if is_training:
        labels = np.asarray(
            [int(record["label"]) for record in dataset.records], dtype=np.int64
        )
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(
                f"Training manifest must contain both labels; got counts {counts.tolist()}"
            )
        weights = 1.0 / counts[labels]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
