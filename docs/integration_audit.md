# Integration audit

## Scope reviewed

This review covers the merged source layout, shared camera interface, dashboard file contract, UCF-Crime manifest/evaluation workflow, and repository packaging. It is a source and artifact review; the test suite and live model runs were not executed.

## Findings

| Priority | Finding | Effect and action |
|---|---|---|
| High | The supplied behavior checkpoint is a Git LFS pointer, and the UCF-Crime videos/annotations are not in the archive. | The external-evaluation workflow is implemented, but no result can be computed from this checkout. Fetch a real checkpoint and authorized dataset before reporting metrics. |
| High | The existing camera demo runs the object/person detectors only; there is no shared runtime policy combining object, behavior and tracking evidence. | The dashboard shows the connected model states and `UNASSESSED`. It does not convert a low score, object detection or proximity measurement into `NORMAL` or a confirmed threat. The integrated upstream pipeline still needs to publish the agreed policy output. |
| Medium | The Streamlit dashboard has no authentication and the current-frame image may contain identifiable people. | Keep it on a trusted local machine, do not expose the server to a network, and remove the session output after review. |
| Medium | UCF-Crime labels are not object boxes or MOT trajectories. The external report scores only the configured behavior mapping. | Do not use this report to claim Person 1 detector precision/recall or Person 3 tracking accuracy. Those require separate annotated datasets. |
| Medium | Live camera and dashboard timing depends on the connected device, backend and model load. | The dashboard displays runtime counters from the producer, but a sustained end-to-end benchmark still needs to be recorded on the target laptop. |

## Static review checks

- Dashboard paths are restricted to the repository `outputs/` tree; frame paths in status files must resolve inside the selected session directory.
- Dashboard reads are bounded to the latest 1,000 event rows and a 2 MB event-log tail.
- Session status writes use atomic replacement; event rows are appended as JSON Lines.
- Missing logs, stale sessions, missing model outputs and valid empty event logs are displayed as different states.
- UCF-Crime manifest generation accepts only relative split paths under the selected videos root, records source-file hashes, and does not copy dataset footage into the repository.
- Evaluation uses the checkpoint validation threshold and does not perform training or threshold selection on the official test list.
- Failed videos do not leave partial window predictions in the score file; unscored videos are excluded from event-rate denominators and shown separately.
- Temporal event scoring uses one-to-one interval matching, maximizing the number of valid IoU matches before counting misses and false alarms.
- The report marks partial runs and unscored videos explicitly.
- Ruff passed on the six changed Python files, all six passed Python syntax parsing, and the annotation parser read all 290 rows from the official UCF-Crime temporal annotation file.

## Not verified in this checkout

Streamlit is not installed in the current Python environment, the detector runtime dependency is unavailable, the behavior checkpoint payload is missing, and the UCF-Crime video data is absent. No automated tests or live camera evaluation were run. Install the declared requirements and supply the missing model/data before accepting runtime or accuracy results.

