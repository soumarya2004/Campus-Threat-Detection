"""
Unit tests for video preprocessing, clip sampling, and video-level dataset splitting.
"""

import numpy as np

from data.preprocessing import (
    VideoPreprocessor,
    create_video_level_splits,
    sample_clip_indices,
)


def test_sample_clip_indices_sliding():
    total_frames = 100
    clip_length = 16
    stride = 2
    clips = sample_clip_indices(
        total_frames, clip_length=clip_length, stride=stride, mode="sliding"
    )

    assert len(clips) > 0
    for c in clips:
        assert len(c) == clip_length
        # Check stride consistency
        diffs = np.diff(c)
        assert np.all(diffs == stride)
        assert c[-1] < total_frames


def test_sample_clip_indices_short_video():
    # Video shorter than required span
    total_frames = 10
    clip_length = 16
    stride = 2
    clips = sample_clip_indices(
        total_frames, clip_length=clip_length, stride=stride, mode="sliding"
    )

    assert len(clips) == 1
    assert len(clips[0]) == clip_length
    assert clips[0][-1] <= total_frames - 1


def test_video_level_splitting_zero_leakage():
    # Construct synthetic records across 10 distinct videos
    records = []
    for vid_idx in range(10):
        vid_id = f"video_{vid_idx:02d}"
        label = 1 if vid_idx % 2 == 0 else 0
        # 5 clips per video
        for clip_idx in range(5):
            records.append(
                {
                    "source_video_id": vid_id,
                    "clip_idx": clip_idx,
                    "label": label,
                    "path": f"/fake/{vid_id}_{clip_idx}.mp4",
                }
            )

    train_recs, val_recs, test_recs = create_video_level_splits(
        records,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    train_vids = {r["source_video_id"] for r in train_recs}
    val_vids = {r["source_video_id"] for r in val_recs}
    test_vids = {r["source_video_id"] for r in test_recs}

    # Verify complete independence - ZERO video or frame leakage!
    assert len(train_vids.intersection(val_vids)) == 0, "Train and Val share video IDs!"
    assert len(train_vids.intersection(test_vids)) == 0, (
        "Train and Test share video IDs!"
    )
    assert len(val_vids.intersection(test_vids)) == 0, "Val and Test share video IDs!"

    # Total clips match
    assert len(train_recs) + len(val_recs) + len(test_recs) == len(records)


def test_label_mapping_parser():
    preprocessor = VideoPreprocessor("configs/dataset_mapping.yaml")

    # UCF-Crime violence mapping
    lbl, cat = preprocessor.parse_ucf_crime_label("Assault001_x264.mp4")
    assert lbl == 1
    assert cat == "Assault"

    lbl, cat = preprocessor.parse_ucf_crime_label("Fighting012_x264.mp4")
    assert lbl == 1
    assert cat == "Fighting"

    # UCF-Crime property crime / normal mapping (must NOT be marked as violence)
    lbl, cat = preprocessor.parse_ucf_crime_label("Shoplifting005_x264.mp4")
    assert lbl == 0
    assert cat == "Shoplifting"

    lbl, cat = preprocessor.parse_ucf_crime_label("Normal_Videos_015_x264.mp4")
    assert lbl == 0
    assert cat == "Normal_Videos_event"

    # XD-Violence mapping
    lbl, cat = preprocessor.parse_xd_violence_label("Fighting")
    assert lbl == 1

    lbl, cat = preprocessor.parse_xd_violence_label("Car accident")
    assert lbl is None  # Excluded from human-violence training; not a normal negative.
