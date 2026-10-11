# Campus Threat Detection

A real-time campus threat detection project using YOLO object detection and computer vision.

## Person 1 — Object Detection Model

This module focuses on training and evaluating a YOLO model for five dangerous-item classes:

- Machete
- Knife
- Baseball bat
- Rifle
- Gun

## Environment

- Python 3.12.10
- Ultralytics 8.4.175
- PyTorch 2.11.0+cu128
- CUDA-enabled NVIDIA GPU

## Dataset

Dataset: [Dangerous Items Dataset for 5-Class Object Detection](https://doi.org/10.5281/zenodo.16422779)

License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The dataset and local training artifacts are not included in this repository.

## Model Evaluation

The trained YOLO11n checkpoint achieved the following results on the configured test split:

| Metric | Result |
|---|---:|
| Precision | 0.894 |
| Recall | 0.795 |
| mAP@50 | 0.879 |
| mAP@50–95 | 0.622 |

These results are from the local test evaluation and should be interpreted in the context of the dataset and evaluation configuration.

## Local Setup

1. Install Python 3.12.
2. Create and activate a virtual environment.
3. Install PyTorch and Ultralytics with compatible GPU support.
4. Download the dataset and arrange it according to `data.yaml`.
5. Obtain the required YOLO checkpoint.
6. Run training and evaluation using the documented commands.

## Project Structure

- `src/` — model integration code
- `scripts/` — training and evaluation scripts
- `docs/` — technical documentation
- `reports/` — evaluation summaries
- `data.yaml` — dataset configuration
