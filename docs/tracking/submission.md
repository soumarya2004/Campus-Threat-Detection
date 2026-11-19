# Submission and viva

## Problem and contribution

An offline recorded-video experiment measures the trade-off between calling a person detector on every frame and calling it during detector-triggered windows. The contribution combines the supplied object detector with pretrained person detection, ByteTrack association, an explicit gate, and interpretable geometry over tracked boxes. A synthetic case verifies the algorithms and a pedestrian clip exercises the model path.

| Contribution | Evidence |
| --- | --- |
| Shared object detector | Cleaned YOLO11n checkpoint, frame-level wrapper, interface tests, and handoff under `reference/detector/`. |
| Tracking and interaction module | Continuous and gated modes, event-scoped ByteTrack identifiers, movement/pair/proximity calculations, CLI, notebook, and evaluation. |
| Behavior and camera modules | The temporal behavior package and shared live/recorded capture source are integrated in the repository. The dashboard and final threat-assessment policy are outside this tracking module's evaluated scope. |

## Method

The shared camera source supplies timestamped BGR frames. The person model uses the person class and sends boxes to ByteTrack. In gated mode the separate five-class item detector supplies candidate predictions. A valid positive opens or extends a finite tracking window. An error is recorded on that frame and consumes elapsed time without extending the window. Scope all analytics by event, then by raw tracker ID or canonical pair.

Movement is the change in a bounding-box center over time. Track starts, missing observations, and implausible speeds require explicit validity handling. Pair distance uses the Euclidean separation of two centers. The defaults of 5 px for change, 150 px for closeness and 600 px/s for a plausible speed are experiment settings tied to image resolution and scene geometry. Duration counts observed close frames using the source FPS.

## Viva outline

1. **0:00–0:40.** State the research question and describe the two separate detectors and the gate.
2. **0:40–1:45.** Play the synthetic gated example. Pause at frames 20 and 21 to show the inclusive grace boundary. Pause again at frame 30 and show the new event ID. The same raw ByteTrack IDs can occur in both events.
3. **1:45–2:30.** Pick one tracked frame during an event. Trace its event ID from `frames.csv` into `movement.csv` and the matching `pairs.csv` row. Explain px, px/s, and one proximity episode in the CSVs.
4. **2:30–3:30.** Show the continuous versus gated results. Explain that fewer person-model calls can coexist with lower coverage and higher wall time when the item detector runs on all frames.
5. **3:30–4:00.** Explain one recorded ambiguity or fragmented track from the evaluation, and describe the limitations of a small monocular sample.

## Limitations

A single recorded scene establishes a small reproducibility and coverage check. Synthetic detections establish known-answer behavior and supply no detector accuracy estimate. Perspective and occlusion affect box centers and track identity. The system reports candidate object detections and image-plane observations for review. It makes no determination about a person's intent or the presence of a real incident.
