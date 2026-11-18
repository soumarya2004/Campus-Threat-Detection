"""
Unit tests for the real-time BehaviorClassifier inference module and rolling buffer.
"""

import numpy as np

from models.behavior_classifier import BehaviorClassifier
from models.temporal_model import TemporalViolenceModel


def test_behavior_classifier_buffering_and_schema():
    # Instantiate classifier with lightweight custom architecture for fast CPU unit testing
    model = TemporalViolenceModel(architecture="custom_3d_cnn", pretrained=False)
    classifier = BehaviorClassifier(
        model=model,
        clip_length=8,
        sampling_stride=2,
        spatial_resolution=(112, 112),
        threshold=0.50,
    )

    required_frames = classifier.required_buffer_size
    assert required_frames == (8 - 1) * 2 + 1  # 15 frames

    # Feed frames 1 by 1 until required_frames - 1
    dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)

    for i in range(required_frames - 1):
        res = classifier.process_frame(dummy_frame, timestamp=float(i * 0.1))
        assert res["model_status"] == "buffering"
        assert res["violence_probability"] == 0.0
        assert res["buffered_frames"] == i + 1

    # Ingest the final frame to fill buffer
    res_final = classifier.process_frame(
        dummy_frame, timestamp=float((required_frames - 1) * 0.1)
    )

    # Output verification against required format
    assert res_final["model_status"] == "valid"
    assert "timestamp" in res_final
    assert "window_start" in res_final
    assert "window_end" in res_final
    assert "violence_probability" in res_final
    assert isinstance(res_final["violence_probability"], float)
    assert 0.0 <= res_final["violence_probability"] <= 1.0
    assert res_final["window_end"] >= res_final["window_start"]

    # Verify reset functionality
    classifier.reset()
    assert len(classifier.frame_buffer) == 0
    res_after_reset = classifier.infer()
    assert res_after_reset["model_status"] == "buffering"
