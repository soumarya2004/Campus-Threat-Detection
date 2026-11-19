# Campus Threat Detection

This shared repository contains the computer-vision components for a university
major project: object detection, temporal behavior recognition, tracking and
interaction analysis, and a common camera/video source. The modules now live in
one package tree so the integration layer can pass frames and timestamps between
them through shared packages and interfaces.

The object detector recognizes
**machete, knife, baseball bat, rifle, and gun**. A separate pretrained person
detector supplies boxes to tracking. The temporal model provides a rolling
violence-related score, while tracking exports movement and proximity evidence.
These are separate evidence streams; the repository does not treat a detection,
movement pattern, or model score as a confirmed threat.

`src.camera_source.CameraSource` is the shared input interface for live cameras
and recorded video. `src.models.object_detector.YOLOObjectDetector` takes an OpenCV frame and
returns boxes, class names, confidence, timing, and an explicit inference status.
See [the handoff](docs/person1_detector_handoff.md) for the exact schema.
See [the camera integration guide](docs/camera_pipeline.md) for input settings,
capture metadata, recorded-video mode, and the shared module layout. The
[Person 5 dashboard and external evaluation guide](docs/person5_dashboard_evaluation.md)
explains the operator view, UCF-Crime evaluation workflow, and current integration
limits. The [standalone detector guide](docs/standalone_demo.md) documents the
detector-only demonstration and its limits.

## Quick demonstration

With the environment installed, run:

```powershell
python scripts/check_model_release.py
python scripts/test_detector.py path/to/image.jpg --output outputs/image.jpg
python scripts/run_video_inference.py path/to/video.mp4 --max-frames 300
python scripts/run_camera_demo.py --camera 0 --with-people
python -m src.tracking_analysis run --video path/to/video.mp4 --mode gated --output outputs/tracking/run-01
```

The camera demo uses the shared source in `configs/camera.yaml`, including a
local webcam or DroidCam virtual camera. Use `--video path/to/video.mp4` for a
recorded source.
Press **Q** or **Escape** to stop. Green boxes are items; blue boxes are people.
It displays object predictions, not threat classifications. The `--with-people`
option requires the local `yolo11n.pt` pretrained checkpoint described below.
Use `--device cpu` when CUDA is unavailable.

The standalone detector consumer runs both models and validates their output after JSON
serialization. It keeps item and person class IDs in separate namespaces:

```powershell
python scripts/check_standalone_pipeline.py --video path/to/video.mp4 --max-frames 60 --device cpu --output outputs/solo/check.json
python scripts/run_camera_demo.py --camera 0 --with-people --device cpu --max-frames 120 --report outputs/solo/camera.json
python scripts/run_camera_demo.py --video path/to/video.mp4 --device cpu --max-frames 120 --no-display --report outputs/solo/video-camera.json
```

A recorded rehearsal passed all 60 frames with 25 item boxes and 60 person boxes.
A visible integrated-camera rehearsal processed 120 frames without inference
errors at 640 x 480. The live scene returned no boxes. These establish execution
and interface behavior, not accuracy. Live capture/inference/display measured
5.55 FPS on CPU; the separate recorded consumer measured 17.73 FPS.

A later [GPU consumer run](reports/experiments/standalone_gpu_20261117.json)
also passed all 60 frames with 25 item boxes and 60 person boxes. Combined
prediction median/p95 were 19.89/23.65 ms. Ultralytics emitted one NMS time-limit
warning during this run; passing the output checks does not make it a
warning-free reliability test.

## Standalone submission package

```powershell
python scripts/package_person1.py
```

This creates `outputs/deliverables/person1_standalone.zip` with both checkpoints,
the shared capture source, detector source, tests, configurations, documentation and recorded results. Each file has
a SHA-256 entry in the included manifest. Extract it to a new folder and follow
the setup instructions. Python dependencies still need installation; training
datasets and local recordings are acquired separately. The bundle includes the
capture source required by its live camera demo.

## Selected model and evidence

`configs/detector.yaml` selects the checkpoint, image size, and confidence.
`models/model_card.json` records its hashes, training conditions, validation
selection, test results, and limitations. The current experiment report is
[the grouped-model evaluation](reports/person1_grouped_model_evaluation.md).

| Held-out grouped test | Result |
|---|---:|
| mAP@50 / mAP@50-95 | 85.79% / 65.89% |
| Precision / recall at confidence 0.45 | 88.03% / 77.94% |
| Knife recall | 68.24% |

The [external-video pilot](reports/person1_demo_and_external_pilot.md) was much
weaker: 1 of 12 provisional knife annotations matched. Its labels need human
review. Present that limitation alongside the dataset score.
The [dated label review](reports/person1_external_label_review.md) checked all 36
frames and confirmed that the poor result is not confined to ambiguous boxes:
none of six originally unflagged knife frames matched. Frozen labels and scores
remain unchanged.

The historical cleaned checkpoint scored 0.882 mAP@50 on its old test split.
The audit reproduced that number **and confirmed augmented train/test overlap**.
Those reports remain as experiment history; use the new grouped-run report for
the selected model. Scores from the old and new splits are not a controlled
before/after comparison.

## Local setup

Tested on Windows with Python 3.12.10, PyTorch 2.11.0, torchvision 0.26.0,
Ultralytics 8.4.175, and an NVIDIA RTX 5060 Laptop GPU. The detector also accepts
`device="cpu"`; CPU performance must be measured on the target machine.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Choose the matching PyTorch build first. For the CUDA 12.8 environment used here:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

For a CPU-only environment, use `--index-url https://download.pytorch.org/whl/cpu`
in that command instead. Then install the remaining packages and check them:

```powershell
python -m pip install -r requirements.txt
python src/utils/check_environment.py
python scripts/check_model_release.py
python -m pytest
```

Both detector checkpoints are included in the repository and submission ZIP.
If the general pretrained checkpoint is missing, restore it once from the
repository root. The runtime wrappers require local files and do not download
models during a demo:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python -m pytest -m integration
```

The environment check executes a native NMS operation, not just package imports.
The default tests use fakes and small temporary datasets. The separate
`integration` tests load both real checkpoints and predict a frame on CPU;
they check runtime compatibility, not detection accuracy.
Use `--require-cuda` when checking a machine intended for GPU training.

The initial November 17 checks used CPU while the NVIDIA GPU was unavailable.
A later [hardware recheck](reports/experiments/hardware_recheck_20261117.json)
passed CUDA tensor/NMS checks on the RTX 5060 Laptop GPU, and a fresh
[GPU benchmark](reports/experiments/gpu_benchmark_20261117.json) completed.
The earlier CPU reports and historical GPU timings remain unchanged.
The [standalone verification record](reports/experiments/solo_completion_20261117.json)
records the current tests, live-camera run and standalone consumer results.
The [person model card](models/person_model_card.json) identifies the
separate pretrained weights and their unvalidated campus operating threshold.

## Run the detector

```powershell
python scripts/smoke_test_wrapper.py path/to/image.jpg
python scripts/run_video_inference.py path/to/video.mp4 --max-frames 300
python scripts/run_video_inference.py path/to/video.mp4 --device cpu --output outputs/demo.mp4
```

The video tool saves an annotated MP4 and JSON report. Reports include processed
frames, detections, startup/warm-up, latency percentiles, processing FPS, stop
reason, decoding warnings, and sampled inference error messages. Detection
counts are not counts of verified true positives.
Media timestamps come from the decoder. Missing or stalled values remain
`null`; a fallback playback FPS is never treated as a capture timestamp.

`--model`, `--confidence`, and `--device` override the shared defaults for an
experiment. Without overrides, every inference utility uses the same selected
configuration. An image can be annotated with:

```powershell
python scripts/test_detector.py path/to/image.jpg --output outputs/image.jpg
python scripts/inspect_video_detections.py path/to/video.mp4
python scripts/inspect_suspicious_frames.py --video path/to/video.mp4 --frames 40 80 120
```

For a repeatable runtime check on a real image:

```powershell
python scripts/benchmark_detector.py --image path/to/image.jpg --device cpu --output outputs/benchmarks/cpu.json
python scripts/benchmark_detector.py --image path/to/image.jpg --device 0 --output outputs/benchmarks/gpu.json
```

The benchmark reports p50/p95 latency, setup time, input/model hashes, and peak
PyTorch CUDA allocation/reservation when available. It repeats one decoded
image; it does not estimate dataset accuracy or complete video throughput.

The [17 November GPU run](reports/experiments/gpu_benchmark_20261117.json)
measured 100 synchronized predictions after warm-up: median **15.06 ms**, p95
**16.53 ms**, peak allocated **68.3 MiB**, and peak reserved **86.0 MiB**.
These allocator peaks include model tensors but exclude CUDA driver/context
memory and other processes. They describe this item-only inference workload.

The [model/batch comparison](reports/person1_model_comparison.md) uses 64 fixed
validation images, three repetitions, YOLO11n/YOLO11s and batches 1/4/8. It
measures compute cost rather than larger-model weapon accuracy. General YOLO11s
was 2.16 times slower per image at batch 1; batch 8 offered little improvement
over batch 4 for recorded batch processing. Live detection stays at batch 1.

A separate [common-class baseline](reports/experiments/common_class_baseline_20261117.json)
compares the general pretrained and fine-tuned YOLO11n on all 700 validation
images, restricted to knife and baseball bat. At the same confidence of 0.25,
micro F1 was 39.84% versus 79.33%. This is descriptive validation evidence for
the two shared classes, not five-class mAP or an untouched-test estimate.

The [historical training-memory analysis](reports/experiments/grouped_training_memory.json)
found a largest logged reserved allocation of 3.30 GiB during the recorded
640-pixel, batch-16 GPU training run. That is a rounded training-log snapshot,
separate from the fresh inference-memory measurement above. The
[earlier hardware record](reports/experiments/hardware_access_20261117.json)
preserves the unavailable-CUDA state before the successful recheck.

## Reproduce the data preparation

Source: [Dangerous Items Dataset for 5-Class Object Detection](https://zenodo.org/records/16422779),
by Zbigniew Omiotek. The source dataset is attributed under its CC BY 4.0 license.
See also the [associated research paper](https://doi.org/10.3390/electronics14173540).
The images are not included in this repository.

Extract the downloaded images and YOLO labels into:

```text
data/raw/dangerous_items/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

Run from the repository root:

```powershell
python scripts/build_deduplicated_dataset.py --data data.yaml
python scripts/fix_rounding_boxes.py
python scripts/build_grouped_dataset.py
python scripts/verify_clean_splits.py --manifest reports/dataset/grouped_v1_manifest.csv
```

Each step creates a new dataset and preserves its source. Existing dataset
outputs are refused. When rebuilding from a fresh checkout, the builder accepts
existing manifest/summary files only if the regenerated bytes match. See
[the data methodology](docs/data_methodology.md) for alternate output directories
and cross-platform line-ending differences.

The final split contains **3,267 training, 700 validation, and 700 test images**.
Originals and known offline augmentations are grouped together; byte-identical
images also join the same group. One representative per group is selected, and
online augmentation is applied only during training. Groups with conflicting
annotations on identical images are excluded and listed in the manifest.

The verifier checks hashes, filename families, labels, and manifest contents.
This establishes separation of known image families. Different frames from an
unknown source video can still be related; source-video independence has not
been established.

## Train and evaluate

Download the general pretrained initialization once from the repository root:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python scripts/train_detector.py
```

The training script requires that local `yolo11n.pt`. It deliberately starts
from general pretrained weights rather than the historical fine-tune, which
already saw images related to the regrouped holdout. Settings are in
`configs/train.yaml`: 50 epochs, early-stopping patience 12, batch 16, two data
workers, AMP, image size 640, seed 0. If the machine needs conservative settings:

```powershell
python scripts/train_detector.py --batch 8 --workers 0 --no-amp --name conservative_run
```

The selection plan in `configs/experiment_plan.yaml` is fixed before test
inspection. Compare 640 and 960 inference sizes on **validation**:

```powershell
$model = "runs/detect/grouped/yolo11n_grouped_v1/weights/best.pt"
python scripts/evaluate_detector.py --model $model --split val --threshold 0.35 --imgsz 640 --output outputs/reproduction/val640.json
python scripts/evaluate_detector.py --model $model --split val --threshold 0.35 --imgsz 960 --output outputs/reproduction/val960.json
```

Select the larger size only if its validation mAP50-95 gain is at least 0.02 and its warmed
model-stage p95 stays below 33.33 ms. These timings exclude file decoding and
display; they are a component budget, not an end-to-end FPS claim.

Then tune confidence on validation and evaluate the frozen settings on test.
Use the selected size for both commands; examples below use 640:

```powershell
$model = "runs/detect/grouped/yolo11n_grouped_v1/weights/best.pt"
python scripts/tune_threshold.py --model $model --imgsz 640 --thresholds 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 --output outputs/reproduction/validation_threshold.json
python scripts/evaluate_detector.py --model $model --split test --imgsz 640 --threshold-file outputs/reproduction/validation_threshold.json --output outputs/reproduction/test.json --plots
python scripts/review_model_errors.py outputs/reproduction/test.json --output-dir outputs/reproduction/error_review
```

Replace `$model` with the actual run's checkpoint if Ultralytics created a
suffixed run directory. The tuner maximizes validation macro F1; ties prefer
recall, then lower confidence. The evaluator rejects threshold files tied to
a different model, validation dataset, image size, or inference settings.
Evaluation, tuning, and benchmark reports refuse existing output paths by
default. Use a new experiment name, or explicitly pass `--overwrite` to replace
a report. The training provenance records actual train/validation content hashes.

Reports separate standard mAP from **fixed-threshold** precision/recall/F1.
They include per-class results, background-image false-positive rate, confusion
counts, timing, and individual mistakes. Background-image false positives are
not a video false-alert rate. The review sheets show both errors and successes;
selected examples are diagnostic, not another accuracy estimate.

## Project map

| Location | Purpose |
|---|---|
| `src/detector.py` | Small Ultralytics adapter |
| `src/models/object_detector.py` | Validated frame interface used by the team |
| `src/models/person_detector.py` | Separate pretrained person boxes for tracking |
| `src/camera_source.py`, `configs/camera.yaml` | Shared live-camera and recorded-video capture with timestamps and reconnect handling |
| `src/behavior/` | Temporal behavior model, data preparation, training, and rolling inference |
| `src/tracking_analysis/` | Event-gated tracking, movement/proximity features, and video exports |
| `assets/tracking/` | Tracking example inputs and local model weights |
| `tests/behavior/`, `tests/tracking/` | Module-specific tests under one project test suite |
| `src/model_config.py`, `configs/detector.yaml` | Shared inference defaults |
| `src/evaluation.py` | Explicit matching, metrics, and experiment fingerprints |
| `scripts/` | Preparation, training, evaluation, and inspection commands |
| `tests/` | Interface, dataset, scoring, and video failure tests |
| `scripts/check_model_release.py` | Verify defaults, weights, model card and evaluation agree |
| `scripts/run_camera_demo.py` | Camera/video demonstration with optional dashboard feed |
| `dashboard/app.py` | Local operator view for live session state and evaluation reports |
| `src/external_evaluation/` | UCF-Crime manifest parsing and held-out event metrics |
| `docs/integration_audit.md` | Current integration risks, scope and validation gaps |
| `reports/dataset/` | Frozen split manifest and verification evidence |
| `reports/experiments/` | Recorded training, selection, and evaluation results |

The next external validation step is a larger human-reviewed footage study
with ordinary look-alike objects, distance, low light, and occlusion. The small
provisional pilot and dataset test scores do not establish campus performance.
