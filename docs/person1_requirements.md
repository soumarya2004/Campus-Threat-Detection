# Person 1 requirements and evidence

This maps the Member 1 work described in `ctd.pdf`, pages 3-5, to the repository
as of 17 November 2026. The longer `ctd_p1.pdf` is a record of previous work;
statements in that conversation are not treated as evidence that an experiment
or integration check passed. Paths below identify the code or recorded results
that can be inspected. The deliverable is now scoped to Person 1 running alone:
the local consumer verifies the handoff contract without requiring P3/P4 code.
Building the team's tracker, camera service, or dashboard is outside this scope.

| Requirement | Repository evidence | Status and remaining work |
|---|---|---|
| Set up CPU/CUDA inference and check the environment | [`check_environment.py`](../src/utils/check_environment.py), [`requirements.txt`](../requirements.txt), [hardware recheck](../reports/experiments/hardware_recheck_20261117.json), real-checkpoint integration tests | CPU execution checked for both detector streams. The later 17 November recheck passed `--require-cuda` on the RTX 5060 Laptop GPU; fresh GPU inference completed. |
| Audit the dataset and labels; prevent related images crossing splits | [`data_methodology.md`](data_methodology.md), dataset preparation scripts, [`grouped_v1_manifest.csv`](../reports/dataset/grouped_v1_manifest.csv), [`grouped_v1_verification.json`](../reports/dataset/grouped_v1_verification.json) | Known augmentation families and exact duplicates are separated. Independence between differently named frames from unknown source videos remains unproven. |
| Establish a pretrained baseline | [`compare_pretrained_baseline.py`](../scripts/compare_pretrained_baseline.py), [`common_class_baseline_20261117.json`](../reports/experiments/common_class_baseline_20261117.json) | Completed a common-class baseline for knife and baseball bat on all 700 validation images. Shared confidence 0.25 gives micro F1 39.84% for pretrained versus 79.33% for the fine-tune. COCO lacks the full project label set; no five-class baseline or untouched-test claim is made. |
| Fine-tune on the selected item classes | [`train_detector.py`](../scripts/train_detector.py), [`training_args.yaml`](../reports/experiments/training_args.yaml), [`training_results.csv`](../reports/experiments/training_results.csv) | Recorded 50-epoch YOLO11n grouped-data run; selected weights are shipped in `models/`. |
| Compare model sizes, input resolutions, and batch settings | [Model/batch report](../reports/person1_model_comparison.md), [`validation_640.json`](../reports/experiments/validation_640.json), [`validation_960.json`](../reports/experiments/validation_960.json) | Completed validation resolution study plus controlled CPU compute comparison of YOLO11n/YOLO11s and batches 1/4/8 on 64 fixed validation images across three repeats. Larger-model five-class accuracy and training-batch convergence were not measured; inference batching is identified separately. |
| Report precision, recall, mAP, per-class behavior, and errors | [`grouped_test.json`](../reports/experiments/grouped_test.json), [`review_model_errors.py`](../scripts/review_model_errors.py), [external-label review](../reports/person1_external_label_review.md) | Frozen 700-image item test includes mAP and fixed-threshold metrics. All 36 external images/labels were checked and visually reviewed by an assistant. Six knife boundaries remain ambiguous; even the six unflagged knife frames had no matches. This is not human approval or a campus benchmark. |
| Provide a frame-level API | [`object_detector.py`](../src/models/object_detector.py), [`test_output_schema.py`](../tests/test_output_schema.py), [handoff](person1_detector_handoff.md) | Implemented: BGR frames in; pixel boxes, labels, confidence, frame metadata, timing, and explicit status out. Invalid frames/metadata and inference failures have separate handling. |
| Supply consumable person boxes | [`person_detector.py`](../src/models/person_detector.py), [`check_standalone_pipeline.py`](../scripts/check_standalone_pipeline.py), [CPU rehearsal](../reports/experiments/standalone_handoff_20261117.json), [GPU rehearsal](../reports/experiments/standalone_gpu_20261117.json) | Separate namespaces, matching metadata and JSON consumption verified on 60 recorded frames on each device: 25 item boxes and 60 person boxes per run. The GPU run emitted one NMS time-limit warning; all 60 output checks still passed. No teammate code is needed for this P1 check. Campus-person accuracy remains unmeasured. |
| Demonstrate recorded and live input | [`run_video_inference.py`](../scripts/run_video_inference.py), [`run_camera_demo.py`](../scripts/run_camera_demo.py), [`live_camera_20261117.json`](../reports/experiments/live_camera_20261117.json) | Completed a visible integrated-camera run of 120 frames at 640 x 480 on CPU with both streams and zero inference errors. Live loop throughput was 5.55 FPS; combined prediction median/p95 were 65.66/73.67 ms. The scene produced no retained boxes, so this is runtime evidence, not live detection accuracy. |
| Measure latency and GPU memory | [`benchmark_detector.py`](../scripts/benchmark_detector.py), [fresh GPU benchmark](../reports/experiments/gpu_benchmark_20261117.json), [historical memory analysis](../reports/experiments/grouped_training_memory.json) | Completed 100 synchronized item-only GPU predictions: median/p95 15.06/16.53 ms, peak allocated/reserved 68.3/86.0 MiB after warm-up. PyTorch allocator memory excludes CUDA driver/context and other processes. CPU item-only, recorded two-stream and live-camera latency are also measured. The historical 3.30 GiB training snapshot remains a separate measurement. |
| Package weights, settings, and a usable handoff | [`package_person1.py`](../scripts/package_person1.py), [`check_model_release.py`](../scripts/check_model_release.py), [standalone guide](standalone_demo.md) | ZIP exporter verifies both checkpoint hashes and packages weights, source, tests, configs and evidence with a per-file manifest. Datasets, local recordings, environments and temporary files are excluded. |

The [standalone demonstration guide](standalone_demo.md) runs both detectors on
the same decoded frame, frame ID, and timestamp. It keeps item and person labels
separate and validates the serialized output. The measured recorded pipeline ran
at 17.73 FPS on CPU. Its throughput and the live-camera rate measure different
input/display workloads; neither includes another team's tracking or dashboard.
The later GPU rehearsal measured combined prediction median/p95 of 19.89/23.65
ms; its NMS warning is retained in the verification record.

The remaining research work is specific: review the external labels, collect
permitted footage with independent source videos, and measure person and item
detection on it before making campus-performance claims. A successful demo is
useful evidence of runtime behavior, but it does not replace those measurements.
