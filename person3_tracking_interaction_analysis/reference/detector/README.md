# Campus Threat Detection

A real-time campus threat detection project using YOLO object detection and computer vision.

## Person 1 — Object Detection Model

This module trains and evaluates a YOLO model for five dangerous-item classes:

- Machete
- Knife
- Baseball bat
- Rifle
- Gun

The current integration checkpoint is `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt` and is included in this repository.

## Environment

- Python 3.12.10
- Ultralytics 8.4.175
- PyTorch 2.11.0+cu128 (development environment)
- CUDA-enabled NVIDIA GPU for GPU inference; CPU use may be possible but will be slower

Exact compatibility depends on the local hardware and installed packages.

## Dataset

Dataset: [Dangerous Items Dataset for 5-Class Object Detection](https://doi.org/10.5281/zenodo.16422779)

License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The dataset is not included in this repository. Local training artifacts other than the committed checkpoint may also be absent.

## Current Cleaned Model Evaluation

The cleaned YOLO11n checkpoint achieved the following results on its reported test split:

| Metric | Result |
|---|---:|
| Precision | 0.878 |
| Recall | 0.815 |
| mAP@50 | 0.882 |
| mAP@50–95 | 0.616 |

See [`reports/person1_clean_model_evaluation.md`](reports/person1_clean_model_evaluation.md) for the dataset split, counts, and limitations. The older [`reports/person1_model_evaluation.md`](reports/person1_model_evaluation.md) documents a different run and its separate metrics; do not mix those values with the cleaned checkpoint results.

These are dataset test-split metrics, not estimates of real-world campus threat-detection accuracy. Source-aware grouping and near-duplicate leakage across splits have not been independently verified, and the configured confidence threshold of `0.35` has not been established as optimal through validation-set threshold tuning.

## Preliminary Video Checks

The video inference utility completed two local clip runs without inference errors:

- `knife.mp4`: 385 frames processed, 141 total detections, 38.76 FPS processing throughput.
- `rifle.mp4`: 145 frames processed, 105 total detections, 89.65 FPS processing throughput.

These are preliminary pipeline/generalization checks. Total detections are not verified true positives, and sampled frames showed at least one likely knife/baseball-bat class confusion. Do not present these video results as formal accuracy metrics.

## Local Setup

1. Install Python 3.12.
2. Create and activate a virtual environment.
3. Install PyTorch and Ultralytics with versions compatible with your hardware.
4. If running training or dataset evaluation, download the dataset and arrange it according to `data.yaml`.
5. Install the remaining dependencies required by the repository.
6. Run the automated tests:

   ```powershell
   python -m pytest tests -q
   ```

7. Run video inference:

   ```powershell
   python scripts\run_video_inference.py input.mp4 --max-frames 300
   ```

## Project Structure

- `src/` — model integration code
- `scripts/` — video inference and inspection utilities
- `docs/` — technical documentation and handoff
- `reports/` — evaluation summaries
- `data.yaml` — dataset configuration
- `runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt` — committed cleaned checkpoint

The detector handles only the five listed object classes. Person detection, tracking, action recognition, threat assessment, and the complete camera/dashboard integration are separate system components.
