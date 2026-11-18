# Event-Gated Person Tracking and Interaction Analysis

Seventh-semester, five-credit computer vision project. The experiment asks how much an object-prediction gate reduces person-detector calls and how it changes observed track coverage. It processes recorded video and exports an annotated MP4, per-frame gate state, movement measurements, pair measurements, and proximity episodes.

## Run the bundled example in VS Code

Open this build folder in VS Code. Install Python 3.13, then run the following from a PowerShell terminal at the folder root.

```powershell
py -3.13 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Select `.venv\Scripts\python.exe` with **Python: Select Interpreter**. With the Python Debugger extension installed, choose **Run offline tracking example** in **Run and Debug**. Give the run a new directory name such as `vscode-smoke`.

The same example runs from the terminal. Choose a new output directory for each run.

```powershell
& '.\.venv\Scripts\python.exe' -m src.tracking_analysis run `
  --video 'assets/demo/approach-and-separate.mp4' `
  --replay 'assets/demo/triggers.json' `
  --mode gated `
  --output 'outputs/terminal-smoke'
```

The bundled 20 FPS clip and JSON replay are synthetic. ByteTrack receives known input boxes and the gate receives scripted predictions. The example verifies timing, track scope and exports without loading either checkpoint. It should process 54 frames and activate on frames 4–20 and 30–46 inclusive. A scripted inference error on frame 9 leaves the existing grace window in effect.

## Run the models on a recorded clip

The build includes the original pretrained person checkpoint and Member 1's cleaned object-detection checkpoint for local use. Point both commands at the same clip so the comparison uses the same frames. CPU is sufficient for a small demonstration.

```powershell
$clip = 'C:\path containing spaces\input.mp4'
& '.\.venv\Scripts\python.exe' -m src.tracking_analysis run `
  --video $clip --mode continuous --device cpu --output 'outputs/clip-continuous'
& '.\.venv\Scripts\python.exe' -m src.tracking_analysis run `
  --video $clip --mode gated --device cpu --output 'outputs/clip-gated'
```

Use `--person-weights` and `--item-weights` to select other local checkpoints. Use `--help` on `run` for supported settings. The selected recorded video should be readable by OpenCV with a positive source FPS. Timing assumes a constant frame rate.

Continuous mode invokes the person stage for every frame. Gated mode invokes the item detector on every frame and opens a track window after a candidate object prediction. The grace period keeps the window open for 0.8 seconds after the last positive prediction by default. A detector error is recorded and cannot extend the window. A new window has a new `event_id` and a fresh tracker instance.

## Read the outputs

| File | Contents |
| --- | --- |
| `annotated.mp4` | Decodable video at the source frame rate with frame state and tracked boxes. |
| `frames.csv` | One row per decoded frame with zero-based frame number, timestamp, prediction status, trigger, event ID and track count. An empty event ID means tracking is inactive. |
| `movement.csv` | One observation per scoped track and frame. Contains source box corners, center, raw and accepted speed, velocity components, direction, and the movement decision. |
| `pairs.csv` | One row per observed canonical pair and event. Distance is image-plane pixels. Approach and separation use change between adjacent valid observations. |
| `proximity.csv` | Continuous, observed close runs for a pair within one event, with inclusive duration in seconds. |
| `summary.json` | Input geometry, processed and active frames, model calls, errors, run settings and wall times. Replay still calls ByteTrack. Its person/item **model** call counts are zero. |

Use `event_id` plus `track_id` to identify a track within a run. A track is an association hypothesis. The same person may have several identifiers after occlusion or a gate reset. Report distances in px and speeds in px/s. Perspective and detector jitter affect both. Candidate object detections require review before assigning any incident label.

## Submission materials

- Open `notebooks/demo.ipynb` in VS Code with the same interpreter for a compact walkthrough.
- Review `evaluation/protocol.md` and `evaluation/results.md` for the comparison, annotations and interpretation.
- Use `docs/submission.md` for the architecture, limitations and viva outline.
- Use `docs/build_review.md` for the judges' completed-build scores and remaining verification.
- Use `reference/PROJECT_AUDIT.md` for the original findings and the two judges' proposals. The original notebook and conversation PDF remain with the source materials.
- Use `assets/README.md` for the model and media sources. The original pedestrian footage and its annotated outputs are outside this build because the source and redistribution terms could not be verified.
