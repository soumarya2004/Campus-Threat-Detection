"""Optional real-checkpoint test; unit tests use fakes to stay fast."""

import numpy as np
import pytest


@pytest.mark.integration
def test_bundled_checkpoint_runs_a_real_frame_on_cpu():
    from src.models.object_detector import YOLOObjectDetector

    detector = YOLOObjectDetector(device="cpu")
    try:
        result = detector.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result["model_status"] == "ok", result["error"]
        assert detector.get_model_info()["class_names"] == {
            0: "machete",
            1: "knife",
            2: "baseball bat",
            3: "rifle",
            4: "gun",
        }
    finally:
        detector.close()
