# Run Person 1 on its own

Person 1 supplies two detectors and a small consumer that verifies their output.
You can demonstrate both streams without any other team member's code. The item
model detects five classes; the general pretrained model supplies person boxes.
Tracking IDs, action recognition, and threat decisions are outside this demo.

## Prepare the environment

Open PowerShell in the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python src/utils/check_environment.py
python scripts/check_model_release.py
python -m pytest
python -m pytest -m integration
```

Follow the README setup on a new machine. The item weights are in
`models/dangerous_items_yolo11n.pt`; person detection requires the separately
downloaded root `yolo11n.pt`. Both wrappers require local files so a demonstration
does not depend on a network download. Use CPU when the CUDA check is unavailable.

## Demonstrate a recording

Run both detectors on the same original frame, then pass their results through
the standalone JSON consumer:

```powershell
python scripts/check_standalone_pipeline.py --video outputs/rifle.mp4 --max-frames 60 --device cpu --output outputs/solo/rehearsal.json --payloads outputs/solo/frames_new.jsonl
```

Replace the video path with any permitted local recording. The repository's
existing demo videos are local files and are not included in Git. Use new output
filenames for another run, or pass `--overwrite` when intentionally replacing
your report. Inputs and model checkpoints remain protected from output collisions.

The consumer checks every frame packet after JSON serialization:

- Both streams use the same frame ID and decoder timestamp.
- Boxes use the original frame's pixel coordinates and lie within its dimensions.
- Class IDs and names match the correct stream; confidence and timing are finite.
- Both inference statuses are successful. A failed detector makes the rehearsal
  fail even if the other stream succeeds.

Class `0` means **machete** in `items` and **person** in `people`. Keep the
namespaces separate. The JSONL file contains one validated packet per line:

```python
import json
from pathlib import Path

for line in Path("outputs/solo/frames_new.jsonl").read_text().splitlines():
    packet = json.loads(line)
    items = packet["items"]
    people = packet["people"]
    assert items["frame_id"] == people["frame_id"]
    print(items["frame_id"], len(items["detections"]), len(people["detections"]))
```

The command exits with code 0 only when the rehearsal passes. Missing or stalled
media timestamps fail this strict contract check; an assumed FPS is not used to
invent them. The JSON report records errors, model and source hashes, box counts,
sample packets, and measured combined processing time. Optional JSONL output
contains validated packets only; consult the report before treating it as a
complete run.

For an annotated item video, run the existing visual demo separately:

```powershell
python scripts/run_video_inference.py outputs/rifle.mp4 --max-frames 60 --device cpu --output outputs/solo/rifle_annotated.mp4
```

## Demonstrate the live camera

```powershell
python scripts/run_camera_demo.py --camera 0 --with-people --device cpu --report outputs/solo/camera_session.json
```

Use the index assigned to your integrated webcam or DroidCam. Green boxes are
items and blue boxes are people. Press Q or Escape to close the window and
release the camera. For a bounded check without a display window:

```powershell
python scripts/run_camera_demo.py --camera 0 --with-people --device cpu --max-frames 60 --no-display --report outputs/solo/camera_check.json
```

Live timestamps are elapsed host time after capture. They are not the camera's
hardware timestamps. The [saved live rehearsal](../reports/experiments/live_camera_20261117.json)
processed 120 frames with both streams and a visible window at 640 x 480.
There were no inference errors and no retained detections in that scene. The
combined prediction median/p95 were 65.66/73.67 ms; capture, inference and display
together ran at 5.55 FPS on CPU. This is not a real-time GPU performance claim.
Repeat the check on the presentation machine when its setup changes.

## Recorded standalone result

The [17 November rehearsal](../reports/experiments/standalone_handoff_20261117.json)
processed **60 frames on CPU**. All 60 passed the consumer checks, with **25 item
boxes and 60 person boxes** validated. Neither stream dropped detections, and no
inference or schema failures were recorded. The full local packet log is
`outputs/solo/frames.jsonl`.

These counts establish that the standalone output can be consumed correctly.
They are not true-positive counts: the demo clip has unverified source
provenance and no independent person annotations. Use the
[grouped item evaluation](../reports/person1_grouped_model_evaluation.md) and
[external pilot](../reports/person1_demo_and_external_pilot.md) when discussing
detection accuracy. Present a missed detection alongside a success, and keep
the frozen model settings visible during the demonstration.
