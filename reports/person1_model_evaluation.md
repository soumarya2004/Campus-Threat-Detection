# Person 1 — Dangerous-Item Detection Model Report

## 1. Model

- Architecture: Ultralytics YOLO11n
- Pretrained checkpoint: `yolo11n.pt`
- Fine-tuned checkpoint: `best.pt`
- Task: Object detection
- Number of classes: 5

## 2. Dataset

- Name: Dangerous Items Dataset for 5-Class Object Detection (YOLO annotation)
- Source: https://doi.org/10.5281/zenodo.16422779
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Classes: machete, knife, baseball bat, rifle, gun
- Annotation format: YOLO bounding-box labels

The dataset itself is not included in this repository.

## 3. Training Configuration

- Maximum epochs: 50
- Completed epochs: 50
- Image size: 640 × 640
- Batch size: 8
- Device: CUDA GPU 0
- DataLoader workers: 0
- Optimizer: auto
- Initial learning rate: 0.01
- Final learning-rate factor: 0.01
- Random seed: 0
- Deterministic mode: enabled
- Automatic mixed precision: disabled
- Validation during training: enabled
- Dataset fraction: 1.0

The complete training configuration is recorded locally in `runs/detect/person1/yolo11n_50ep/args.yaml`.

## 4. Final-Epoch Metrics

Source: `runs/detect/person1/yolo11n_50ep/results.csv`, epoch 50.

| Metric | Value |
|---|---:|
| Precision | 0.89009 |
| Recall | 0.80376 |
| mAP@50 | 0.87955 |
| mAP@50–95 | 0.62476 |
| Training box loss | 0.78113 |
| Validation box loss | 0.99599 |

These are the final-epoch training-run metrics, not the separate test-split results.

## 5. Test-Split Evaluation

The separately executed test evaluation reported:

| Metric | Value |
|---|---:|
| Precision | 0.894 |
| Recall | 0.795 |
| mAP@50 | 0.879 |
| mAP@50–95 | 0.622 |

Test evaluation used `best.pt` and the configured test split. The test run evaluated 1,272 images and 1,328 instances.

## 6. Test Metrics by Class

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Machete | 0.865 | 0.756 | 0.845 | 0.601 |
| Knife | 0.827 | 0.709 | 0.812 | 0.575 |
| Baseball bat | 0.914 | 0.844 | 0.909 | 0.721 |
| Rifle | 0.938 | 0.872 | 0.935 | 0.682 |
| Gun | 0.928 | 0.793 | 0.895 | 0.532 |

## 7. Limitations

These metrics describe performance on the selected dataset and test split. They do not establish real-world campus threat-detection accuracy.

The model detects the five listed item classes; it does not provide a person-detection class. Real-time camera integration, latency testing, and validation on representative campus footage remain separate tasks.

## 8. Reproducibility Artifacts

Local artifacts include:

- Training arguments: `runs/detect/person1/yolo11n_50ep/args.yaml`
- Training metrics: `runs/detect/person1/yolo11n_50ep/results.csv`
- Training plots and confusion matrices in the same run directory
- Test evaluation plots in `runs/detect/person1/test_evaluation/`
- Best checkpoint: `runs/detect/person1/yolo11n_50ep/weights/best.pt`

Large checkpoints, local dataset files, and generated training outputs are excluded from Git by `.gitignore`.