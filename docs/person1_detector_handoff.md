# Person 1 — Object Detector Integration Handoff

## 1. Purpose

This module provides weapon and dangerous-item object detection for the Campus Threat Detection System.

It uses the cleaned YOLO11n checkpoint through the existing `YOLOObjectDetector` wrapper. Other modules should call this wrapper instead of loading the model independently.

## 2. Supported Classes

| Class ID | Class name |
|---:|---|
| 0 | machete |
| 1 | knife |
| 2 | baseball bat |
| 3 | rifle |
| 4 | gun |

These are the five classes supported by the current model. Person detection, violence recognition, tracking, and threat classification are separate responsibilities.

## 3. Model and Environment

- Wrapper: `src/models/object_detector.py`
- Class: `YOLOObjectDetector`
- Checkpoint: `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt`
- Framework: PyTorch and Ultralytics YOLO
- Preferred inference device: CUDA GPU when available
- Default confidence threshold: `0.35`

Keep the checkpoint locally available. Large model weights are excluded from Git.

## 4. Basic Integration Example

Run from the repository root with the project's virtual environment activated.

```python
import cv2
from src.models.object_detector import YOLOObjectDetector

detector = YOLOObjectDetector(confidence_threshold=0.35)

capture = cv2.VideoCapture(0)

try:
    frame_id = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = detector.predict(
            frame,
            frame_id=frame_id,
            timestamp=frame_id / 30.0,
        )

        if result["model_status"] == "ok":
            for detection in result["detections"]:
                print(
                    detection["class_name"],
                    detection["confidence"],
                    detection["bbox"],
                )

        frame_id += 1

finally:
    capture.release()
    detector.close()
```

This example assumes a camera is available at index `0`. For the project pipeline, the frame can instead come from the smartphone camera feed supplied through DroidCam and OpenCV.

The timestamp calculation above assumes 30 FPS; replace it with the actual source timestamp or frame rate when integrating the real video pipeline.

## 5. Input Contract

Pass each frame as an OpenCV BGR image:

- Type: NumPy array
- Shape: height × width × 3
- Data type: `uint8`

The wrapper validates input frames. Do not pass a file path, RGB image, or raw video object directly to `predict()`.

## 6. Prediction Output

The wrapper returns a result dictionary containing:

- `frame_id`: frame identifier supplied by the caller
- `timestamp`: timestamp supplied by the caller
- `detections`: normalized detection results
- `inference_time_ms`: detector-reported inference latency
- `model_status`: model/inference status
- `error`: error information when applicable
- `dropped_detections`: count of rejected detections

Each accepted detection contains:

- `class_id`: numeric class ID
- `class_name`: mapped class name
- `confidence`: confidence score
- `bbox`: bounding box in `[x1, y1, x2, y2]` format

Check the actual returned schema in `src/models/object_detector.py` when implementing downstream interfaces. Do not assume fields that are not part of the wrapper's contract.

## 7. Downstream Integration

### Person 2 — Video pipeline

Supply decoded OpenCV frames to the detector and preserve frame IDs and timestamps. Keep camera capture and detector inference as separate responsibilities.

### Person 3 — Tracking and movement analysis

Consume object detections with their frame IDs and timestamps. Tracking requires consistent identifiers and temporal state; the detector itself does not assign persistent object-track IDs.

### Person 4 — Temporal violence/action recognition

Implement action recognition as a separate module using the agreed video-frame or temporal-clip interface. The object detector does not classify violence or human actions.

### Person 5 — Dashboard and threat assessment integration

Consume outputs from the integrated detection, tracking, and action-recognition modules. Do not treat an isolated weapon detection as a definitive threat decision without the project's agreed threat-assessment logic.

Confirm exact function signatures and message schemas with each module owner before integration.

## 8. Video Inference Utility

The video runner is located at:

`scripts/run_video_inference.py`

Example:

```powershell
python scripts\run_video_inference.py input.mp4 --max-frames 300
```

It writes an annotated MP4 and a JSON report. The report includes detection counts, inference latency percentiles, startup/warm-up duration, processing time, and processing throughput.

For a custom output path:

```powershell
python scripts\run_video_inference.py input.mp4 --output outputs\annotated.mp4
```

This utility processes a video file; it is not itself the complete multi-module campus monitoring application.

## 9. Testing

Run the automated test suite:

```powershell
python -m pytest tests -q
```

The current suite has passed 67 tests in the development environment.

The video runner can be smoke-tested using the synthetic video generated during development:

```powershell
python scripts\run_video_inference.py outputs\synthetic_test.mp4 --max-frames 30
```

Synthetic video verifies pipeline execution, not weapon-detection accuracy.

## 10. Evaluation and Limitations

Cleaned-model test-split metrics:

| Metric | Result |
|---|---:|
| Precision | 0.878 |
| Recall | 0.815 |
| mAP@50 | 0.882 |
| mAP@50–95 | 0.616 |

These metrics apply to the evaluated dataset and test split. They do not establish real-world campus detection accuracy. Source-aware grouping and near-duplicate leakage across splits have not been independently verified.

The detector does not independently perform person detection, persistent tracking, temporal action recognition, or threat classification. Real-world evaluation and end-to-end latency testing remain necessary.

## 11. Relevant Files

- `src/models/object_detector.py` — reusable object-detection wrapper
- `src/detector.py` — existing detector implementation
- `scripts/run_video_inference.py` — video inference and reporting utility
- `scripts/smoke_test_wrapper.py` — wrapper smoke test
- `tests/` — automated tests
- `reports/person1_model_evaluation.md` — original model evaluation
- `reports/person1_clean_model_evaluation.md` — cleaned model evaluation

Keep the detector interface stable and coordinate any schema changes with downstream module owners.
