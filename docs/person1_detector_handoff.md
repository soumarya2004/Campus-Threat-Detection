# Person 1 — Object Detector Integration Handoff

## 1. Purpose

This module provides five-class dangerous-item object detection for the Campus Threat Detection System. It uses the cleaned YOLO11n checkpoint through the reusable `YOLOObjectDetector` wrapper. Downstream modules should use the wrapper instead of loading the model independently.

## 2. Supported Classes

| Class ID | Class name |
|---:|---|
| 0 | machete |
| 1 | knife |
| 2 | baseball bat |
| 3 | rifle |
| 4 | gun |

Person detection, persistent tracking, violence/action recognition, and threat classification are separate responsibilities.

## 3. Model and Environment

- Wrapper: `src/models/object_detector.py`
- Class: `YOLOObjectDetector`
- Checkpoint: `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt`
- Framework: PyTorch and Ultralytics YOLO
- Preferred inference device: CUDA GPU when available
- Default confidence threshold: `0.35`
- Checkpoint status: **included in this Git repository** at the path above.

The dataset and most local training artifacts are not included. Check `data.yaml` and the evaluation report for the expected dataset layout and recorded results.

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

This example assumes a camera is available at index `0`. For the project pipeline, frames may instead come from the smartphone camera feed supplied through DroidCam and OpenCV. The example timestamp assumes 30 FPS; use the actual source timestamp or frame rate in the integrated pipeline.

## 5. Input Contract

Pass each frame as an OpenCV BGR image:

- Type: NumPy array
- Shape: height × width × 3
- Data type: `uint8`

Do not pass a file path, RGB image, or raw video object directly to `predict()`.

## 6. Prediction Output

The wrapper result includes the following fields (see `src/models/object_detector.py` for the authoritative schema):

- `frame_id`: frame identifier supplied by the caller
- `timestamp`: timestamp supplied by the caller
- `detections`: normalized detection results
- `inference_time_ms`: detector-reported inference latency
- `model_status`: model/inference status
- `error`: error information when applicable
- `dropped_detections`: count of rejected detections

Each accepted detection contains `class_id`, `class_name`, `confidence`, and `bbox` in `[x1, y1, x2, y2]` format.

## 7. Downstream Integration

- **Person 2 — Video pipeline:** supply decoded OpenCV frames and preserve frame IDs and timestamps.
- **Person 3 — Tracking/movement analysis:** consume detections with frame IDs and timestamps. This detector does not assign persistent track IDs.
- **Person 4 — Temporal violence/action recognition:** implement action recognition separately using the team's agreed frame/clip interface.
- **Person 5 — Dashboard/threat assessment:** consume the integrated modules' outputs. Do not treat one object detection as a definitive threat decision without the agreed threat-assessment logic.

Confirm exact function signatures, timestamp units, and message schemas with downstream module owners before integration.

## 8. Video Inference Utility

The video runner is `scripts/run_video_inference.py`.

```powershell
python scripts\run_video_inference.py input.mp4 --max-frames 300
```

For a custom output path:

```powershell
python scripts\run_video_inference.py input.mp4 --output outputs\annotated.mp4
```

It writes an annotated video and JSON report containing aggregate detection counts, latency percentiles, startup/warm-up duration, processing time, and processing throughput. This utility is not the complete multi-module monitoring application.

## 9. Testing and Preliminary Video Validation

Run the automated test suite:

```powershell
python -m pytest tests -q
```

The development run recorded **67 passed tests**. Re-run the suite in the current checkout to verify the current environment.

Recorded-video runs completed without inference errors for the following local clips:

| Clip | Frames processed | Total detections | Processing throughput | Notes |
|---|---:|---:|---:|---|
| `knife.mp4` | 385 | 141 | 38.76 FPS | Sampled frames showed knife predictions and at least one likely knife/baseball-bat class confusion. |
| `rifle.mp4` | 145 | 105 | 89.65 FPS | Rifle-class predictions were visible in sampled frames; detections have not all been manually verified. |

These are preliminary qualitative checks, not formal accuracy measurements. Detection totals are not counts of verified true positives. The synthetic clip, when used, checks pipeline execution only; it does not establish weapon-detection accuracy.

## 10. Evaluation and Limitations

The cleaned checkpoint's reported test-split metrics are:

| Metric | Result |
|---|---:|
| Precision | 0.878 |
| Recall | 0.815 |
| mAP@50 | 0.882 |
| mAP@50–95 | 0.616 |

See `reports/person1_clean_model_evaluation.md` for the dataset split, test counts, and interpretation. The older `reports/person1_model_evaluation.md` describes a different model/run and should not be treated as the cleaned checkpoint's metrics.

These metrics apply to the evaluated dataset and split; they do not establish real-world campus detection accuracy. Source-aware grouping and near-duplicate leakage across splits have not been independently verified. The confidence threshold `0.35` is configured but has not been established as an optimal operating point through validation-set threshold tuning.

The detector does not independently perform person detection, persistent tracking, temporal action recognition, or threat classification. Representative real-world video evaluation, false-positive/false-negative analysis, and end-to-end latency testing remain necessary.

## 11. Relevant Files

- `src/models/object_detector.py` — reusable object-detection wrapper
- `src/detector.py` — existing detector implementation
- `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt` — committed cleaned checkpoint
- `scripts/run_video_inference.py` — video inference and reporting utility
- `scripts/inspect_video_detections.py` — class-level video detection inspection utility
- `scripts/inspect_suspicious_frames.py` — targeted frame inspection utility; currently configured for `outputs/knife.mp4` and frames 150, 200, and 350
- `tests/` — automated tests
- `reports/person1_model_evaluation.md` — original model/run evaluation
- `reports/person1_clean_model_evaluation.md` — cleaned model evaluation

Keep the detector interface stable and coordinate schema changes with downstream module owners.
