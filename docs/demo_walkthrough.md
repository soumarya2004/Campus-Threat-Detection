# A five-minute Person 1 demo

This demonstration covers the five-class item detector and its handoff. It does
not demonstrate person tracking, violence recognition, or a completed campus
alert system. The existing clips are runtime examples. An external-video pilot
exists, but its labels remain provisional and its knife results are weak; see
[the pilot report](../reports/person1_demo_and_external_pilot.md).

## Prepare before presenting

Run PowerShell from the repository root with the project environment installed:

```powershell
.\.venv\Scripts\Activate.ps1
python src/utils/check_environment.py
python scripts/check_model_release.py
python -m pytest
python -m pytest -m integration
Get-Content configs/detector.yaml
```

The selected settings should be `models/dangerous_items_yolo11n.pt`, image size
640, and confidence 0.45. Keep those settings fixed while showing mistakes.

These commands use files already present in this workspace. The dataset and
local clips are not included in Git, so copy the permitted demo inputs to the
presentation laptop in advance or follow the README's dataset setup.

```powershell
python scripts/test_detector.py data/processed/dangerous_items_grouped_v1/images/test/00a649fb-2917.jpg --output outputs/demo/knife_match.jpg
python scripts/test_detector.py data/processed/dangerous_items_grouped_v1/images/test/0017cc01-0481.jpg --output outputs/demo/knife_miss.jpg
python scripts/run_video_inference.py outputs/rifle.mp4 --output outputs/demo/rifle.mp4
python scripts/run_video_inference.py outputs/knife.mp4 --output outputs/demo/knife.mp4
python scripts/review_model_errors.py outputs/experiments/grouped_test_full.json --output-dir outputs/demo/error_review --limit 3
```

The first image had one correctly matched knife in the saved test evaluation
(confidence about 0.806). The second had one annotated knife and no retained
prediction. These are examples from the frozen test report, not a new accuracy
measurement. The plain image tool draws predictions only; the review sheets
also draw missed ground-truth boxes in red.

Keep these ready to open:

- [Successful image](../outputs/demo/knife_match.jpg) and
  [missed-knife image](../outputs/demo/knife_miss.jpg).
- [Rifle video](../outputs/demo/rifle.mp4) and
  [knife video](../outputs/demo/knife.mp4), with their adjacent JSON reports.
- [Missed objects](../outputs/demo/error_review/missed.jpg) and
  [false positives](../outputs/demo/error_review/false_positive.jpg).
- [Final report](../reports/person1_grouped_model_evaluation.md),
  [model card](../models/model_card.json), and
  [team handoff](person1_detector_handoff.md).

The error-sheet command requires the full local report and its source images.
If that report is absent, recreate it with the README's evaluation commands;
the compact committed report deliberately omits per-image predictions.

## The five minutes

| Time | Show | Explain |
|---|---|---|
| 0:00–0:35 | The five classes and `configs/detector.yaml` | A BGR frame enters; item boxes, classes, confidence and status come out. An item box does not determine a person's intent. |
| 0:35–1:20 | Rerun the successful-image command above, then open its output | The current checkpoint can execute through the same wrapper used by teammates. Explain one box and its confidence; do not call confidence a threat probability. |
| 1:20–2:05 | Short sections of the prepared rifle and knife outputs | This is recorded-video processing. In each JSON, check `processed_frames`, `frames_with_inference_errors` and `stop_reason`. The recorded complete checks processed 145 and 385 frames with zero inference errors. |
| 2:05–3:05 | The result table below, then the missed-object and false-positive sheets | Show both a successful prediction and a failure. A labeled knife can be missed. Selected examples explain behavior; the full 700-image test supplies the metrics. |
| 3:05–4:05 | The dataset section and validation decisions in the final report | Known augmentation families were grouped before splitting, a fresh model was trained, and 640/0.45 were chosen on validation before the final test. The old and new test scores are not a controlled before/after comparison. |
| 4:05–5:00 | The output dictionary in the handoff | Point out pixel boxes, frame ID, timestamp and `model_status`. Explain why `inference_error` must not be treated as a safe empty frame. Finish with the remaining need for independently sourced, labeled footage. |

Use these exact results from the selected checkpoint's grouped test:

| Metric | Result |
|---|---:|
| mAP@50 | 85.79% |
| mAP@50–95 | 65.89% |
| Precision at confidence 0.45 | 88.03% |
| Recall at confidence 0.45 | 77.94% |
| Knife recall at confidence 0.45 | 68.24% |
| TP / FP / FN | 544 / 74 / 154 |

There were no detections on 74 background test images, but there were 74
false-positive boxes elsewhere in the test set. This does not establish a zero
false-alarm rate. Source-video independence also remains unverified.

## If something fails during the demo

For a local live-camera demonstration, run:

```powershell
python scripts/run_camera_demo.py --camera 0 --with-people
```

Use the camera index assigned to DroidCam if different. The person detector
requires the local general pretrained `yolo11n.pt`; the item model remains the
selected five-class checkpoint. Green boxes are items and blue boxes are people.
Press Q or Escape to stop. The [saved camera rehearsal](../reports/experiments/live_camera_20261117.json)
processed 120 frames on the integrated camera with both streams and no inference
errors. It retained no boxes in that scene. Repeat the rehearsal on your
presentation laptop; the recorded CPU live loop ran at 5.55 FPS.

Read the error instead of changing the model or threshold to hide it. A CPU
fallback for the successful image is:

```powershell
python scripts/test_detector.py data/processed/dangerous_items_grouped_v1/images/test/00a649fb-2917.jpg --device cpu --output outputs/demo/knife_match_cpu.jpg
```

If the laptop cannot run the model, show the prepared outputs and identify them
as saved runs. Keep the final report available as the evidence. Video playback
speed and the model's component timing are not live camera-to-dashboard latency.

The [standalone guide](standalone_demo.md) checks BGR frames, timestamp
units/origin, frame IDs, class namespaces and error handling with a local consumer.
This completes the P1 interface rehearsal without requiring the other modules.
