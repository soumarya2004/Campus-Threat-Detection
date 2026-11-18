"""
Video preprocessing, clip extraction, and video-level dataset splitting.
Strictly isolates source videos across splits to prevent identical background/frame leakage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .xd_violence import parse_xd_filename

logger = logging.getLogger("ViolenceClassifier.Preprocessing")

# Kinetics-400 normalization used by the pretrained torchvision Video ResNet.
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
# Backwards-compatible aliases for the original project modules.
IMAGENET_MEAN = KINETICS_MEAN
IMAGENET_STD = KINETICS_STD


def sample_clip_indices(
    total_frames: int,
    clip_length: int = 16,
    stride: int = 2,
    mode: str = "uniform",
) -> List[List[int]]:
    """
    Generate frame index sequences for fixed-length temporal clips.

    Args:
        total_frames: Total number of frames in the source video.
        clip_length: Number of frames per clip (e.g. 16).
        stride: Temporal stride between sampled frames (e.g. 2).
        mode: 'sliding' (consecutive overlapping clips) or 'uniform' (spaced throughout video).

    Returns:
        List of lists containing integer frame indices for each clip.
    """
    required_span = (clip_length - 1) * stride + 1

    if total_frames < required_span:
        # Loop / repeat frames if video is shorter than required temporal span
        base_indices = np.linspace(0, total_frames - 1, clip_length, dtype=int).tolist()
        return [base_indices]

    clips = []
    if mode == "sliding":
        # Step forward by half the span for overlapping temporal coverage
        step = max(1, required_span // 2)
        for start_idx in range(0, total_frames - required_span + 1, step):
            indices = [start_idx + i * stride for i in range(clip_length)]
            clips.append(indices)
    else:  # 'uniform'
        num_clips = max(1, total_frames // required_span)
        start_positions = np.linspace(
            0, total_frames - required_span, num_clips, dtype=int
        )
        for start_idx in start_positions:
            indices = [start_idx + i * stride for i in range(clip_length)]
            clips.append(indices)

    return clips


def extract_clips_from_video(
    video_path: str,
    clip_length: int = 16,
    stride: int = 2,
    target_size: Tuple[int, int] = (224, 224),
    max_clips: Optional[int] = None,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Extracts fixed-length clips from a video file.

    Returns:
        List of tuples: (clip_array_rgb_normalized [T, H, W, C], start_sec, end_sec)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_index_lists = sample_clip_indices(
        total_frames, clip_length=clip_length, stride=stride, mode="sliding"
    )
    if max_clips and len(clip_index_lists) > max_clips:
        # Uniformly subsample clips
        chosen_indices = np.linspace(0, len(clip_index_lists) - 1, max_clips, dtype=int)
        clip_index_lists = [clip_index_lists[i] for i in chosen_indices]

    extracted_clips = []

    for indices in clip_index_lists:
        clip_frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Pad with last valid frame or zeros if read failed
                if clip_frames:
                    clip_frames.append(clip_frames[-1].copy())
                else:
                    clip_frames.append(
                        np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    )
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame_rgb.shape[1], frame_rgb.shape[0]) != target_size:
                frame_rgb = cv2.resize(
                    frame_rgb, target_size, interpolation=cv2.INTER_LINEAR
                )
            clip_frames.append(frame_rgb)

        if len(clip_frames) == clip_length:
            clip_array = np.stack(clip_frames, axis=0).astype(np.float32) / 255.0
            # Apply ImageNet normalization: (clip - mean) / std
            clip_normalized = (clip_array - IMAGENET_MEAN) / IMAGENET_STD
            start_sec = indices[0] / fps
            end_sec = indices[-1] / fps
            extracted_clips.append((clip_normalized, start_sec, end_sec))

    cap.release()
    return extracted_clips


class VideoPreprocessor:
    """
    Preprocesses video datasets and generates split manifests.
    Enforces video-level splitting to prevent identical scene background leakage.
    """

    def __init__(self, mapping_config_path: str = "./configs/dataset_mapping.yaml"):
        with open(mapping_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.clip_len = self.config["clip_parameters"]["clip_length_frames"]
        self.stride = self.config["clip_parameters"]["sampling_stride"]
        self.target_size = tuple(self.config["clip_parameters"]["spatial_resolution"])

    def parse_ucf_crime_label(self, video_name: str) -> Tuple[int, str]:
        """
        Parses UCF-Crime video filename (e.g. 'Assault001_x264.mp4', 'Normal_Videos_001_x264.mp4').
        Returns (target_binary_label, category_name).
        """
        mapping = self.config["ucf_crime"]["label_mapping"]
        video_stem = Path(video_name).stem

        for category, info in mapping.items():
            if category.lower() in video_stem.lower():
                return info["target_class"], category

        raise ValueError(
            f"Could not map UCF-Crime filename to a known category: {video_name}"
        )

    def parse_xd_violence_label(self, annotation_str: str) -> Tuple[Optional[int], str]:
        """
        Parses an XD category name, filename, or official filename tag.

        Returns ``None`` for hazard-only videos excluded from the human-
        violence task. Unknown categories raise an error rather than becoming
        negative examples.
        """
        mapping = self.config["xd_violence"]["label_mapping"]
        clean_tag = annotation_str.strip()
        if clean_tag in mapping:
            return mapping[clean_tag].get("target_class"), clean_tag
        candidate = (
            clean_tag
            if "_label_" in clean_tag
            else f"sample_label_{clean_tag.removeprefix('label_')}.mp4"
        )
        label, categories, _disposition = parse_xd_filename(candidate)
        return label, "+".join(categories)


def create_video_level_splits(
    video_records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits records strictly by source video ID.
    Frames or clips from the same source video NEVER leak across splits.

    Stratified by class: Positive videos and negative videos are split independently.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, (
        "Ratios must sum to 1.0"
    )
    rng = np.random.RandomState(seed)

    # Group records by source_video_id and track video-level label
    video_groups: Dict[str, List[Dict[str, Any]]] = {}
    video_labels: Dict[str, int] = {}

    for rec in video_records:
        vid_id = rec["source_video_id"]
        if vid_id not in video_groups:
            video_groups[vid_id] = []
            video_labels[vid_id] = rec["label"]
        video_groups[vid_id].append(rec)

    # Separate into positive and negative unique video lists
    pos_videos = [vid for vid, lbl in video_labels.items() if lbl == 1]
    neg_videos = [vid for vid, lbl in video_labels.items() if lbl == 0]

    rng.shuffle(pos_videos)
    rng.shuffle(neg_videos)

    def split_list(vids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(vids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_v = vids[:n_train]
        val_v = vids[n_train : n_train + n_val]
        test_v = vids[n_train + n_val :]
        return train_v, val_v, test_v

    pos_tr, pos_va, pos_te = split_list(pos_videos)
    neg_tr, neg_va, neg_te = split_list(neg_videos)

    train_vids = set(pos_tr + neg_tr)
    val_vids = set(pos_va + neg_va)
    test_vids = set(pos_te + neg_te)

    # Verify zero overlap
    assert len(train_vids.intersection(val_vids)) == 0, (
        "Video leakage detected between train and val!"
    )
    assert len(train_vids.intersection(test_vids)) == 0, (
        "Video leakage detected between train and test!"
    )
    assert len(val_vids.intersection(test_vids)) == 0, (
        "Video leakage detected between val and test!"
    )

    train_records = [rec for vid in train_vids for rec in video_groups[vid]]
    val_records = [rec for vid in val_vids for rec in video_groups[vid]]
    test_records = [rec for vid in test_vids for rec in video_groups[vid]]

    logger.info(
        "Video-Level Split Completed (Zero frame leakage): "
        "Train: %d videos (%d clips) | Val: %d videos (%d clips) | Test: %d videos (%d clips)",
        len(train_vids),
        len(train_records),
        len(val_vids),
        len(val_records),
        len(test_vids),
        len(test_records),
    )

    return train_records, val_records, test_records
