# Person 1: object detector handoff

Person 1 supplies two detector streams: five item classes (machete, knife,
baseball bat, rifle, gun) and person boxes for Member 3's tracker. Other team
members supply the camera pipeline and implement tracking, actions, alerts, and
the dashboard. A bounding box does not establish a person's intent or ownership
of a nearby item.

## Start here

Both wrappers load their checkpoint once and return the same frame dictionary.
Use these interfaces rather than the lower-level adapter in `src/detector.py`:

| Stream | Interface | Checkpoint and defaults |
|---|---|---|
| Items | `YOLOObjectDetector` from `src.models.object_detector` | `configs/detector.yaml`: selected fine-tuned model, confidence 0.45, input size 640 |
| People | `YOLOPersonDetector` from `src.models.person_detector` | Local `yolo11n.pt`: general pretrained model, confidence 0.35, input size 640; inference filters COCO class 0 |

The item model card in `models/model_card.json` records its validation selection.
Keep those defaults unless running a named experiment. Person settings are
independent and can be passed to its constructor. The person checkpoint must
already exist locally; loading never triggers an implicit download. Person
detection has a real-checkpoint runtime check, but campus person-detection
accuracy has not been measured.

```python
import time
import cv2
from src.models.object_detector import YOLOObjectDetector
from src.models.person_detector import YOLOPersonDetector

capture = cv2.VideoCapture(0)
try:
    if not capture.isOpened():
        raise RuntimeError("Could not open camera 0")
    with YOLOObjectDetector() as items, YOLOPersonDetector() as people:
        items.warmup()
        people.warmup()
        started = time.monotonic()
        for frame_id in range(60):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            timestamp = time.monotonic() - started
            streams = {
                "items": items.predict(frame, frame_id, timestamp),
                "people": people.predict(frame, frame_id, timestamp),
            }
            for name, result in streams.items():
                if result["model_status"] != "ok":
                    print(f"{name} failed: {result['error']}")
                else:
                    print(name, result["detections"])
            # Member 3 consumes streams["people"] and assigns track IDs.
finally:
    capture.release()
```

This camera example is an integration sketch. Its timestamp is elapsed host
time in seconds after frame acquisition. If the camera pipeline supplies
capture timestamps, pass those instead; a recording can use its media time.
Do not invent timestamps from a presumed 30 FPS camera rate.
The two models run sequentially in this example. The item model's timing alone
does not represent combined person-and-item throughput.

## Input and output

Input is an OpenCV BGR image: NumPy `uint8`, shape `(height, width, 3)`.
Invalid image types and shapes raise `FrameValidationError` before inference.
Optional `frame_id` must be a nonnegative integer. Optional `timestamp` must be
a finite number in seconds, with its origin agreed with the caller. Invalid
metadata raises `ValueError` before inference. NumPy numeric metadata is converted
to ordinary Python numbers so the result can be serialized as JSON.

```python
{
    "frame_id": 42,
    "timestamp": 1.68,
    "detections": [
        {
            "bbox": [100.0, 80.0, 150.0, 200.0],
            "class_id": 1,
            "class_name": "knife",
            "confidence": 0.82,
        }
    ],
    "inference_time_ms": 12.3,
    "model_status": "ok",
    "error": None,
    "dropped_detections": 0,
}
```

The numbers above illustrate the schema; they are not an evaluation result.
Boxes use pixel coordinates `[x1, y1, x2, y2]` in the original frame. IDs are
0 machete, 1 knife, 2 baseball bat, 3 rifle, and 4 gun in the **item stream**.
The **person stream** uses class 0 for `person`. Never concatenate the two lists
and interpret their integer IDs without keeping the stream name. Neither stream
returns tracking IDs or associates a person with an item.

An empty detection list with status `ok` means no item passed the threshold.
An empty list with status `inference_error` means the model failed. Downstream
code must distinguish these cases. Malformed detections are dropped and
counted. Missing caller timestamps remain `None`.

Model loading errors raise `ModelLoadError`. A checkpoint with a different
class mapping is rejected: the item wrapper requires the five project classes,
and the person wrapper requires class 0 to be `person`. `close()` is safe to call more than once;
`load_model()` reloads a closed detector. Use one instance per
processing worker; do not call the same model concurrently from several threads.
Both wrappers support `with` blocks, which release the model even when consumer
code raises an exception.

## Recorded-video check

```powershell
python scripts/run_video_inference.py input.mp4 --max-frames 300
python scripts/run_video_inference.py input.mp4 --device cpu --output outputs/demo.mp4
```

The command writes an annotated video and a JSON summary. Check inference
errors, decode warnings, and processed-frame count before using its throughput.
Warm-up is measured separately. Processing FPS includes decoding, inference,
drawing, and writing; it is not camera-to-dashboard latency.

For the standalone live demo, run:

```powershell
python scripts/run_camera_demo.py --camera 0 --with-people
python scripts/run_camera_demo.py --camera 0 --device cpu --max-frames 30 --no-display
```

Press Q or Escape in the preview to stop. The standalone integrated-camera
rehearsal processed 120 frames with both streams and no inference errors; see
[the saved session](../reports/experiments/live_camera_20261117.json).
It retained no boxes in that scene and is an execution check, not an accuracy
test. The [standalone consumer](standalone_demo.md) verifies the P1 output
contract without requiring other team modules.

## Checks before handing this module over

```powershell
python -m pytest
python -m pytest -m integration
python src/utils/check_environment.py
python scripts/check_model_release.py
python scripts/smoke_test_wrapper.py path/to/a/known_image.jpg
```

Unit tests check the interface and failure paths. The integration tests load both
real checkpoints on CPU; the image smoke test checks the selected item model.
Neither replaces labeled test-set evaluation. The release check compares the
item checkpoint hash, defaults, model card, and frozen evaluation evidence.
See the README for
the reproducible dataset, training, threshold-selection, and evaluation commands.

## What the other modules need to agree on

- Video pipeline: BGR frames, sequential IDs, and timestamp units/origin.
- Tracking: consume person boxes from `YOLOPersonDetector`; keep track IDs outside
  the detector and retain the frame ID and timestamp when associating results.
- Action recognition: consume a separate frame/clip interface; object labels do
  not establish violence.
- Dashboard and assessment: show inference failures explicitly and keep a
  human review step before treating a detection as a confirmed incident.

The grouped split prevents known augmentation families and exact image hashes
from crossing splits. It cannot prove independence between different frames
from an unidentified source video. Independent footage remains the next
evaluation step. The existing external item pilot has provisional annotations
that require human review. It supplies no person accuracy estimate.

## Rehearse the consumer contract

```powershell
python scripts/check_detector_handoff.py --video outputs/knife.mp4
python scripts/check_detector_handoff.py --video teammate_clip.mp4 --max-frames 10 --device cpu
```

This uses the selected detector on the first five frames by default. It checks
original-frame pixel boxes, class IDs/names, confidence, preserved frame IDs and
media timestamps, JSON serialization, and inference status. OpenCV media
timestamps must be readable and advance between frames; no camera or network
connection is used. The command returns a nonzero exit code on a failed check.

`outputs/handoff_check.json` contains check counts and at most five actual output
payloads, including raw detection boxes when present. Override the path with
`--output`. This is a runtime/schema rehearsal for a downstream consumer; it
does not measure detection accuracy or verify another teammate's full module.
