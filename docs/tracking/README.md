# Tracking and interaction analysis

The Person 3 module is integrated under `src/tracking_analysis/`. It compares
continuous person tracking with event-gated tracking and exports annotated
video, per-frame state, movement features, pair measurements, and proximity
episodes. It uses the shared `src.camera_source.CameraSource` for recorded video
so frame IDs and timestamps follow the same contract as camera demos.

## Run the bundled replay example

Run commands from the repository root. The bundled replay uses scripted person
boxes and candidate-object triggers, so it does not load a model checkpoint:

```powershell
python -m src.tracking_analysis run `
  --video assets/tracking/demo/approach-and-separate.mp4 `
  --replay assets/tracking/demo/triggers.json `
  --mode gated `
  --output outputs/tracking/replay-smoke
```

The MP4 is generated locally and ignored by Git. Create it together with the
JSON inputs using `python assets/tracking/make_demo.py`.

## Run model inference

For a real clip, use the included person and candidate-object checkpoints:

```powershell
python -m src.tracking_analysis run --video path/to/clip.mp4 `
  --mode continuous --device cpu --output outputs/tracking/clip-continuous
python -m src.tracking_analysis run --video path/to/clip.mp4 `
  --mode gated --device cpu --output outputs/tracking/clip-gated
```

The candidate-object detector uses the shared validated detector API. Gated
mode opens a tracking window after candidate detections and keeps it open for
the configured grace period. This reduces person-model calls; it also means
people outside active windows are not tracked in gated mode.

## Read the outputs

| File | Contents |
|---|---|
| `annotated.mp4` | Annotated video at source dimensions and frame rate |
| `frames.csv` | Frame ID, capture timestamp, event ID, gate state, and track counts |
| `movement.csv` | Track boxes, centers, velocities, and movement decisions |
| `pairs.csv` | Observed pair distances and approach/separation measures |
| `proximity.csv` | Continuous close-proximity runs within an event |
| `summary.json` | Model hashes, run settings, call counts, errors, and timing |

Distances and speeds are image-plane measurements. Track IDs are temporary
association labels, not identities. A detected object, close proximity, or
movement pattern is supporting evidence and should not be presented as a
confirmed incident.

## Source layout

| Path | Responsibility |
|---|---|
| `src/tracking_analysis/` | Gate, tracker adapter, movement analysis, and recorded-video runner |
| `src/tracking_detector.py` | YOLO wrapper used by the tracking module's detector adapter |
| `src/tracking_models/` | Legacy tracking-module frame API retained for its handoff tests |
| `assets/tracking/models/` | Local YOLO weights |
| `evaluation/tracking/` | Protocol, sample annotations, and recorded results |
| `tests/tracking/` | Gate, analytics, pipeline, and detector-schema tests |

See [the submission notes](submission.md), [evaluation protocol](../../evaluation/tracking/protocol.md),
and [recorded results](../../evaluation/tracking/results.md) for experiment
scope and limitations.
