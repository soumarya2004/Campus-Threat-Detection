# Evaluation protocol

Does gating the person stage reduce its inference calls on the same recorded clip, and how much sampled person-track coverage and ID continuity does that cost? The comparison runs the two modes with the same person model, settings, input and CPU. Continuous runs the person stage on every frame. Detector-gated runs the candidate-object model on every frame and the person model while the gate is active. Evaluate processing time for the whole pipeline, including the extra model.

## Inputs and fixed settings

| Input | Role | Provenance |
| --- | --- | --- |
| `44860-440261070_medium.mp4` | Real inference and sampled person-track evaluation | User-supplied local copy, 250 frames at 25 FPS and 1280 × 720. SHA-256 `1a5d9ba0917355803a6e7f662daf5f7a372abfe7f56b015c846159b8eea63098`. Source and redistribution terms are unverified. The clip and annotated copies stay outside the submission. |
| [pedestrian_sample.csv](pedestrian_sample.csv) | Manual reference boxes on the real clip | 30 fixed frames, 95 observations, seven pseudonymous subject IDs. Two to four visible targets sampled per frame. The comparison labels were frozen from the raw clip before inspecting the corrected build's model outputs. |
| `assets/tracking/demo/approach-and-separate.mp4`, `triggers.json`, `no-trigger.json` | Deterministic playback and negative control | Self-generated footage and known injection inputs. Replay exercises gate, tracker, movement, pair and export logic using supplied boxes. |
| `assets/tracking/models/yolov8n.pt`, `assets/tracking/models/best.pt` | Real person and candidate-object inference | Ultralytics person model and teammate detector from the supplied archive. The run summary records checkpoint hashes. |

The real clip was used for the earlier prototype, which limits claims of generalization to new footage. The evaluated operating point was kept fixed across the real runs. It uses person model confidence 0.25, ByteTrack activation threshold 0.25, minimum consecutive frames 1, candidate-object confidence 0.35, and gate grace 0.8 s. Pair rules use 150 px for closeness and a 5 px change between adjacent observations for approach/separation. The movement validity limit is 600 px/s. These values express the lab setup and are sensitive to camera perspective, frame rate and resolution.

Review units are zero-based frame IDs `[0, 5, 10, 15, 20]`, `[55, 60, 65, 70, 75]`, `[105, 110, 115, 120, 125]`, `[155, 160, 165, 170, 175]`, `[205, 210, 215, 220, 225]` and `[230, 235, 240, 245, 249]`. Each window represents a different portion of the clip. Labels use visible body extents clipped to the original image, omit carried bags where distinguishable, and mark person-on-person occlusion. The reference covers selected visible tracks, so unannotated objects and unannotated people never count as false positives. The raw clip was reviewed using a pixel grid. The annotation represents one visual pass with approximate boxes.

## Metrics and interpretation

- **Processing.** Record source frames, active frames, candidate-object and person inference calls, detector inference errors, startup time and processing FPS. `elapsed_s` includes input frame reads, inference, rendering/writing and analytics/CSV generation, with model startup and post-write video readback outside the measured interval. Report hardware, library versions and number of runs alongside throughput.
- **Coverage.** For each sampled frame, assign at most one predicted track to a labeled subject and at most one subject to a predicted track, greedily from highest intersection over union (IoU) down to the fixed minimum of 0.50. Report matching labels over all 95 observations, over those in frames processed by both modes, and visibly occluded observations as a separate subset. Missed labels on inactive frames count toward end-to-end coverage loss.
- **Continuity.** A transition is a pair of successive sampled observations of the same subject in the same window. Use both modes' common matched transitions as the denominator for changes in `(event_id, track_id)`. For each mode, also count matched→unmatched and unmatched→matched transitions among all sampled transitions. Changes between sample points may go unseen. The result is a sampled continuity check rather than a full MOT benchmark.
- **Proximity.** Recompute center-to-center distance from each pair of manual boxes in the same sample frame. Distances within 20 px of 150 px are ambiguous and removed from the reference. Count predictions available and their agreement separately for expected close and far pairs. Compare frame-level geometry only. The sparse human labels cannot time consecutive episodes or validate per-frame approach. Use deterministic cases with known geometry and times for these checks.
- **Failure review.** Inspect examples using frame ID and time, distinguishing gate inactivity, below-threshold IoU, missing or occluded tracks, and visually ambiguous candidate boxes. The selected person labels provide no true object annotations, so detector precision, event recall and campus incident accuracy are outside the reported metrics.

## Reproduce the real comparison locally

Run from the build folder after installing its requirements. Set `$clip` to your supplied local copy and `$runs` to a writable local directory whose outputs can stay private. Each mode requires a new output folder. The checked-in report records the settings actually used.

```powershell
$clip = 'C:\path\to\44860-440261070_medium.mp4'
$runs = Join-Path $env:LOCALAPPDATA 'Tracking Analysis Evaluation'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
Get-FileHash -Algorithm SHA256 -LiteralPath $clip
python -m src.tracking_analysis run --video "$clip" --mode continuous --device cpu --output "$runs\continuous"
python -m src.tracking_analysis run --video "$clip" --mode gated --device cpu --output "$runs\gated"
python evaluation/evaluate.py --continuous "$runs\continuous" --gated "$runs\gated" --output "$runs\comparison.json"
```

The evaluator validates the file hash, frame indices and shared settings before comparing modes. Review `matched_tracks` and `missed_labels` in the JSON alongside your private annotated MP4. Synthetic demonstrations are described in the main README and produce their own artifacts without this real clip.
