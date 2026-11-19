# Temporal behavior module

The Person 2 work is integrated under `src/behavior/`. The module prepares
video-level datasets, trains and evaluates the R3D-18 classifier, and provides
rolling-window inference for a timestamped frame stream. It keeps video-level
weak-label learning separate from the UCF-Crime Fighting/Normal notebook
experiment; those datasets and scores are not interchangeable.

The shared camera source is `src.camera_source.CameraSource`. The behavior
module's `StreamFrameProvider` uses it in a background thread and passes frame
timestamps to `BehaviorClassifier`. Live input uses a bounded latest-frame queue
to keep predictions current. Recorded clips use a bounded FIFO queue and apply
backpressure so the run can process the whole file.

## Setup

Install the shared environment from the repository root after choosing the
matching CPU or CUDA PyTorch build:

```powershell
python -m pip install -r requirements.txt
```

The shared requirements preserve the detector versions used for the recorded
Person 1 results and add the behavior and tracking libraries. Training datasets
are not included. See [dataset mapping](dataset_analysis_and_mapping.md),
[architecture](architecture.md), and [model limits](model_card_and_limitations.md)
before interpreting behavior scores.

## Prepare XD-Violence manifests

Place the dataset files outside version control under
`data/behavior/raw/XD-Violence/`, then create video-disjoint manifests:

```powershell
python -m src.behavior.scripts.prepare_xd_violence `
  --dataset-root data/behavior/raw/XD-Violence `
  --train-dir data/behavior/raw/XD-Violence/Train `
  --test-dir data/behavior/raw/XD-Violence/Test `
  --test-annotations data/behavior/raw/XD-Violence/annotations.txt `
  --training-source "archive and source description"
```

The script keeps train and test videos separate and records the source label in
the manifest summary. It maps only the selected human-violence categories to
the positive class; hazard-only labels are not treated as normal footage.

## Train and evaluate

```powershell
python -m src.behavior.train --config configs/behavior/training_config.yaml --device cuda --epochs 8 --batch_size 2
python -m src.behavior.evaluate --config configs/behavior/training_config.yaml `
  --checkpoint models/behavior/best_model.pt `
  --manifest data/behavior/splits/xd_test_manifest.json
python -m src.behavior.error_analysis --report outputs/behavior/logs/xd_test_evaluation.json
```

The supplied source archive contains `models/behavior/best_model.pt` as a Git
LFS pointer, not the 382 MB checkpoint payload. If the Git LFS object is
available on the repository remote, fetch it before inference. Otherwise train
a model from locally acquired data. The real-video evaluation summary in
[experiment status](experiment_status.md) describes a prior run; it does not
make the pointer file loadable or establish campus-scene performance.

## Run temporal inference

With a real checkpoint available, run the rolling classifier on a recorded
clip, a camera index, or the generated synthetic clip:

```powershell
python -m src.behavior.realtime_demo --source path/to/clip.mp4 `
  --checkpoint models/behavior/best_model.pt `
  --config configs/behavior/training_config.yaml `
  --save_output outputs/behavior/demo_output.mp4
```

Use `--source 0` for camera index 0. The demo's violence probability is a model
prediction for review. It is not a confirmed threat decision.

## Code map

| Path | Responsibility |
|---|---|
| `src/behavior/data/` | Clip datasets, video bags, preprocessing, and dataset manifests |
| `src/behavior/models/` | Temporal network, rolling classifier, and feature fusion |
| `src/behavior/interfaces/` | Shared frame provider and alert-event adapter |
| `src/behavior/train.py` | Training and validation threshold selection |
| `src/behavior/evaluate.py` | Held-out evaluation |
| `tests/behavior/` | Behavior-model and preprocessing tests |
| `configs/behavior/` | Training, environment, and label-mapping settings |
