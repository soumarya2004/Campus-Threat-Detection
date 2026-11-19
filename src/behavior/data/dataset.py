"""
PyTorch Dataset and DataLoader for temporal video clips.
Includes motion-preserving augmentations to prevent background memorization.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger("ViolenceClassifier.Dataset")


class VideoClipDataset(Dataset):
    """
    Dataset yielding spatiotemporal video clips of shape (C, T, H, W).
    """

    def __init__(
        self,
        manifest_path_or_records: Any,
        clip_length: int = 16,
        sampling_stride: int = 2,
        spatial_resolution: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        enable_augmentation: bool = True,
    ):
        super().__init__()
        if isinstance(manifest_path_or_records, (str, os.PathLike)):
            with open(manifest_path_or_records, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.records = list(manifest_path_or_records)

        self.clip_length = clip_length
        self.sampling_stride = sampling_stride
        self.target_w, self.target_h = spatial_resolution
        self.is_training = is_training
        self.enable_augmentation = enable_augmentation and is_training

        # Validate entries and precompute clip index
        # Cache for decoded frames to dramatically accelerate CPU / GPU dataloading
        self._video_cache: Dict[str, np.ndarray] = {}
        self.samples = self._build_samples_index()
        logger.info(
            "VideoClipDataset initialized: %d clips from %d source videos (is_training=%s)",
            len(self.samples),
            len(self.records),
            self.is_training,
        )

    def _get_all_frames(self, vpath: str) -> np.ndarray:
        if vpath in self._video_cache:
            return self._video_cache[vpath]

        cap = cv2.VideoCapture(vpath)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if not frames:
            # Fallback zero frame
            frames = [np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)]

        arr = np.stack(frames, axis=0)  # Shape: (total_frames, H, W, 3)
        # Limit cache size to 64 videos to prevent high memory usage
        if len(self._video_cache) > 64:
            self._video_cache.pop(next(iter(self._video_cache)))
        self._video_cache[vpath] = arr
        return arr

    def _build_samples_index(self) -> List[Dict[str, Any]]:
        samples = []
        span = (self.clip_length - 1) * self.sampling_stride + 1

        for rec in self.records:
            vpath = rec["video_path"]
            if not os.path.exists(vpath):
                continue
            total_frames = rec.get("total_frames", 0)
            fps = rec.get("fps", 30.0)

            if total_frames <= 0:
                cap = cv2.VideoCapture(vpath)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()

            if total_frames < span:
                samples.append(
                    {
                        "video_path": vpath,
                        "start_frame": 0,
                        "label": rec["label"],
                        "source_video_id": rec["source_video_id"],
                        "category": rec.get("category", "unknown"),
                        "fps": fps,
                        "total_frames": total_frames,
                    }
                )
            else:
                step = span if not self.is_training else max(1, span // 2)
                for start_f in range(0, total_frames - span + 1, step):
                    samples.append(
                        {
                            "video_path": vpath,
                            "start_frame": start_f,
                            "label": rec["label"],
                            "source_video_id": rec["source_video_id"],
                            "category": rec.get("category", "unknown"),
                            "fps": fps,
                            "total_frames": total_frames,
                        }
                    )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_clip_frames(
        self, sample: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, float]:
        vpath = sample["video_path"]
        start_f = sample["start_frame"]
        fps = sample["fps"]
        stride = self.sampling_stride

        # Temporal jitter during training: random offset of -1, 0, or +1 frame
        if self.enable_augmentation:
            start_f = max(0, start_f + np.random.randint(-1, 2))

        all_video_frames = self._get_all_frames(vpath)
        total_frames = len(all_video_frames)

        indices = [start_f + i * stride for i in range(self.clip_length)]
        indices = [min(idx, max(0, total_frames - 1)) for idx in indices]

        clip = all_video_frames[
            indices
        ]  # Fast direct indexing in memory! Shape: (T, H, W, C)
        start_sec = indices[0] / fps if fps > 0 else 0.0
        end_sec = indices[-1] / fps if fps > 0 else 0.0

        return clip, start_sec, end_sec

    def _apply_augmentations(self, clip: np.ndarray) -> np.ndarray:
        """
        Applies spatiotemporal data augmentations consistently across all frames in a clip.
        This preserves motion coherence while disrupting static scene background bias.
        """
        T, H, W, C = clip.shape

        # 1. Random Horizontal Flip (50% probability)
        if np.random.rand() > 0.5:
            clip = np.flip(clip, axis=2).copy()

        # 2. Random Spatial Crop / Scale (Crop 85-100% of spatial dimensions, then resize)
        if np.random.rand() > 0.3:
            scale = np.random.uniform(0.85, 1.0)
            crop_h = int(H * scale)
            crop_w = int(W * scale)
            top = np.random.randint(0, H - crop_h + 1)
            left = np.random.randint(0, W - crop_w + 1)
            clip = clip[:, top : top + crop_h, left : left + crop_w, :]

        # 3. Color Jitter: slight brightness and contrast variation applied to all frames
        if np.random.rand() > 0.4:
            alpha = np.random.uniform(0.85, 1.15)  # Contrast
            beta = np.random.uniform(-15, 15)  # Brightness
            clip = np.clip(clip.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )

        # Spatial Resize to target dimensions
        resized_frames = []
        for t in range(T):
            f = cv2.resize(
                clip[t], (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR
            )
            resized_frames.append(f)

        return np.stack(resized_frames, axis=0)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]
        clip_raw, start_sec, end_sec = self._read_clip_frames(sample)

        if self.enable_augmentation:
            clip_aug = self._apply_augmentations(clip_raw)
        else:
            # Resize without augmentation
            T, H, W, C = clip_raw.shape
            resized_frames = [
                cv2.resize(
                    clip_raw[t],
                    (self.target_w, self.target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                for t in range(T)
            ]
            clip_aug = np.stack(resized_frames, axis=0)

        # Normalize: (x/255.0 - mean) / std
        clip_float = clip_aug.astype(np.float32) / 255.0
        clip_normalized = (clip_float - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to PyTorch 3D CNN tensor shape: (C, T, H, W)
        # Transpose from (T, H, W, C) -> (C, T, H, W)
        clip_tensor = torch.from_numpy(clip_normalized).permute(3, 0, 1, 2).float()

        # Target label
        label_tensor = torch.tensor([sample["label"]], dtype=torch.float32)

        meta = {
            "source_video_id": sample["source_video_id"],
            "category": sample["category"],
            "start_sec": start_sec,
            "end_sec": end_sec,
        }

        return clip_tensor, label_tensor, meta


def create_dataloader(
    dataset: VideoClipDataset,
    batch_size: int = 8,
    is_training: bool = False,
    balance_classes: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Creates PyTorch DataLoader with optional class-balancing WeightedRandomSampler.
    """
    sampler = None
    shuffle = is_training

    if is_training and balance_classes:
        labels = [s["label"] for s in dataset.samples]
        class_counts = np.bincount(labels, minlength=2)
        total_samples = len(labels)

        # Inverse frequency weights
        class_weights = {
            cls: total_samples / (max(1, count) * 2.0)
            for cls, count in enumerate(class_counts)
        }
        sample_weights = [class_weights[lbl] for lbl in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training,
    )
