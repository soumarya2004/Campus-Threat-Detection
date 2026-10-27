# Person 1 — Cleaned Dangerous-Item Detection Model Report

## 1. Model

- Architecture: Ultralytics YOLO11n
- Task: Object detection
- Number of classes: 5
- Checkpoint: `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt`
- Inference framework: Ultralytics
- Compute device used for inference: NVIDIA RTX 5060 Laptop GPU
- Inference confidence threshold used by the project wrapper: `0.35`

## 2. Dataset

- Dataset: Dangerous Items Dataset for 5-Class Object Detection
- Source: https://zenodo.org/records/16422779
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Processed dataset: `data/processed/dangerous_items_cleaned_v2/`
- Annotation format: YOLO bounding-box labels

### Classes

| Class ID | Class name |
|---:|---|
| 0 | machete |
| 1 | knife |
| 2 | baseball bat |
| 3 | rifle |
| 4 | gun |

### Dataset split

| Split | Images |
|---|---:|
| Train | 5,854 |
| Validation | 1,228 |
| Test | 1,220 |
| **Total** | **8,302** |

The dataset is not included in the Git repository.

## 3. Training

- Training run: `runs/detect/person1_clean/yolo11n_clean_v2/`
- Training duration: 50 epochs
- Image size: 640 × 640
- Hardware: NVIDIA RTX 5060 Laptop GPU
- Best checkpoint: `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt`

The run directory contains local training artifacts. Consult its `args.yaml` and `results.csv` for the exact recorded configuration and training history.

## 4. Test-Split Evaluation

The cleaned checkpoint was evaluated separately on the cleaned dataset's test split.

| Metric | Result |
|---|---:|
| Precision | 0.878 |
| Recall | 0.815 |
| mAP@50 | 0.882 |
| mAP@50–95 | 0.616 |
| Test images | 1,220 |
| Test instances | 1,273 |

### Metric interpretation

- **Precision (87.8%):** Measures the proportion of predicted detections that are correct under the evaluation matching criteria.
- **Recall (81.5%):** Measures the proportion of ground-truth objects successfully detected.
- **mAP@50 (88.2%):** Mean average precision at an intersection-over-union threshold of 0.50.
- **mAP@50–95 (61.6%):** Mean average precision averaged across IoU thresholds from 0.50 to 0.95.

These are dataset evaluation metrics, not estimates of real-world campus threat-detection accuracy.

## 5. Inference Pipeline Smoke Test

The `YOLOObjectDetector` wrapper was tested with the cleaned checkpoint.

The video inference runner was also tested on a synthetic 640 × 480 video containing 75 frames at 15 FPS. A 30-frame run produced the following results:

| Measurement | Result |
|---|---:|
| Frames processed | 30 |
| Detection count | 0 |
| Frames with inference errors | 0 |
| Startup and warm-up | 1.647 seconds |
| Processing time | 0.276 seconds |
| Processing throughput | 108.80 FPS |
| Mean detector-reported inference latency | 7.75 ms |
| 95th-percentile inference latency | 8.79 ms |

The synthetic video contained no weapons, so zero detections were expected. This test verifies that the video-processing pipeline runs end-to-end; it does not validate weapon detection accuracy or sustained performance on real footage.

Processing throughput includes video frame reading, inference, annotation, and output writing. It should not be interpreted as a benchmark of inference alone.

## 6. Reproducibility Artifacts

Local artifacts include:

- Cleaned model checkpoint: `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt`
- Cleaned training arguments: `runs/detect/person1_clean/yolo11n_clean_v2/args.yaml`
- Cleaned training metrics: `runs/detect/person1_clean/yolo11n_clean_v2/results.csv`
- Video inference script: `scripts/run_video_inference.py`
- Synthetic input video: `outputs/synthetic_test.mp4`
- Annotated test output: `outputs/synthetic_test_annotated.mp4`
- Video inference JSON report: `outputs/synthetic_test_annotated.json`

The cleaned checkpoint `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt` is committed to Git. The local dataset, other training artifacts (such as `args.yaml` and `results.csv`), and generated outputs may be excluded from Git according to the repository's ignore rules.

## 7. Limitations

1. Evaluation metrics apply to the selected dataset and its test split.
2. Source-aware grouping and near-duplicate leakage across dataset splits have not been independently verified; performance may therefore be optimistic.
3. The confidence threshold of `0.35` is a configured inference threshold, not a demonstrated optimal operating point.
4. The detector recognizes the five listed object classes. It does not independently perform person detection, action recognition, tracking, or threat classification.
5. Synthetic-video testing does not establish performance on real campus footage.
6. Before deployment, the system needs representative video evaluation, false-positive and false-negative analysis, and end-to-end latency testing.

## 8. Summary

The cleaned YOLO11n checkpoint achieved **0.882 mAP@50** and **0.616 mAP@50–95** on the reported test split. The detector wrapper and synthetic-video inference pipeline completed smoke tests without inference errors, and the existing automated test suite passed 67 tests.

These results establish a working object-detection component, not a validated campus threat-detection system.
