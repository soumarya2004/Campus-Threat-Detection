# Person 5: Dashboard and external evaluation

Person 5 owns the operator view, event review, external test protocol, and consolidated performance report. The dashboard is a consumer of the inference process: it does not load models or start duplicate inference.

## Run the local dashboard

Install the shared requirements. For a USB-connected Android phone, start DroidCam on
the phone and its Windows client, then scan for the virtual camera index. Replace `N`
below with the index that returns a frame:

```powershell
python scripts/find_cameras.py --backend msmf
python scripts/run_camera_demo.py --camera N --backend msmf --with-people --device cpu --no-display --dashboard-session outputs/dashboard/phone-usb
streamlit run dashboard/app.py
```

Run the last command in a second terminal. Open the local dashboard and set **Output
folder** to `outputs/dashboard/phone-usb`. Use a new, empty directory for each camera
run. `--no-display` keeps the continuous capture session from opening an OpenCV window;
with `--dashboard-session`, the local dashboard receives the latest annotated frame,
capture state, model state, and performance data. It does not run the temporal classifier
or tracker, so those modules appear as `not connected` and the assessment stays
`UNASSESSED`.

The current-frame evidence table lists the detector outputs supplied by the producer. A valid empty detection list is shown separately from a producer that has not supplied detection details.

An integrated pipeline can publish to the same file contract through `src.monitoring.session.DashboardSessionWriter`. It should supply its own assessment level and event log. The writer deliberately does not infer a threat level from detections or proximity.

The dashboard polls its selected session folder and distinguishes a missing model output from a valid negative result. A stale running session is marked stale. Event rows show their supplied timestamps, evidence categories, durations and assessment; event review does not relabel them as confirmed incidents.

Run Streamlit on the same trusted computer as the camera process. The app has no authentication and can display identifiable footage. The latest processed frame is stored as `latest.jpg` in the session folder; delete the output folder after review.

## Prepare the UCF-Crime external test manifest

The official UCF-Crime release provides an anomaly-detection test split and temporal annotations for testing videos. Keep its video files and annotation downloads outside Git. The official dataset page describes its categories, split files and temporal annotation download: [UCF-Crime at UCF CRCV](https://www.crcv.ucf.edu/projects/real-world/).

After placing the authorized UCF-Crime files under `data/behavior/raw/UCF-Crime/`, build a local manifest. Paths in the command may need to match the directory names in the official archive:

```powershell
python scripts/prepare_ucf_crime_manifest.py `
  --videos-root data/behavior/raw/UCF-Crime/Videos `
  --test-split data/behavior/raw/UCF-Crime/UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Test.txt `
  --temporal-annotations data/behavior/raw/UCF-Crime/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt `
  --output outputs/behavior/external_ucf/manifest.csv
```

The manifest records only official test-list entries, maps labels using `configs/behavior/dataset_mapping.yaml`, marks missing local videos, and converts annotated frame ranges to seconds using each decoded video's FPS. The behavior task is violence versus non-violence: Abuse, Assault, Fighting and Shooting are positive; the configured non-violence and normal categories remain negative. Other classes are not silently re-labelled as violence.

## Run the frozen external evaluation

The evaluation command loads the selected behavior checkpoint, processes test videos, segments threshold-positive windows into predicted events, and writes `predictions.jsonl`, `report.json`, `run_metadata.json` and dashboard status into a new output directory:

```powershell
python scripts/run_ucf_crime_evaluation.py `
  --manifest outputs/behavior/external_ucf/manifest.csv `
  --checkpoint models/behavior/best_model.pt `
  --output-dir outputs/behavior/external_ucf/run-01 `
  --device cuda
```

The threshold is read from the checkpoint's validation threshold, with the training config used only when the checkpoint has no recorded validation threshold. This command never trains or tunes on the test split. `--max-videos` is for a partial smoke review; such a report is marked partial and must not be presented as a full official test result.

The report contains video-level accuracy/precision/recall/F1/AUROC, event-level temporal-IoU precision/recall/F1, false alarms per normal-video hour, detection delay, model latency, capture-to-score latency, and processing FPS. It counts missing model outputs as unscored rather than as valid negatives. Failed videos do not contribute partial predictions or event-rate hours. The dashboard can display the report by pointing it at the evaluation run directory.

## Interpretation limits

UCF-Crime annotates video-level anomaly classes and temporal event ranges. It does not provide the bounding boxes needed to score the Person 1 weapon detector or MOT ground truth needed to score Person 3 track IDs. This evaluation therefore measures the behavior model on the configured violence-versus-non-violence mapping and does not claim object-detection or tracking accuracy.

The archive's `models/behavior/best_model.pt` is currently an LFS pointer, not a loadable checkpoint payload. Fetch the LFS object or train and select a checkpoint before running the external evaluation. The UCF-Crime videos and annotation files were not included in the supplied project archive, so no external result is reported until those inputs are available.

The dashboard reports upstream levels only. The existing camera demo publishes detector evidence but has no multi-model assessment policy; the UI consequently shows `UNASSESSED`. A future assessment policy must be agreed, validation-tuned and documented before the dashboard can show `NORMAL`, `SUSPICIOUS` or `HIGH_RISK` decisions.

## Dashboard session files

A session folder keeps these files together:

| File | Producer writes | Dashboard shows |
|---|---|---|
| `status.json` | Camera state, model readiness, current frame metadata, FPS, latency, assessment level and heartbeat time | Current source, component status, stale-session warning and performance counters |
| `latest.jpg` | The most recent annotated frame only | Live preview |
| `events.jsonl` | One upstream-assessed event per JSON line | Filterable event history and evidence details |
| `report.json` | Optional external-evaluation summary | Video/event metrics, false alarms, misses and timing |

Event records use `event_id`, `recorded_at_utc`, `level`, `evidence_categories`, optional `start_s`/`end_s`, and producer-supplied evidence fields. The allowed levels are `NORMAL`, `SUSPICIOUS`, `HIGH_RISK`, and `UNASSESSED`. Producers should set `UNASSESSED` when no assessment policy ran. The camera demo writes detections into the current-frame snapshot; it does not promote them into threat events.

`DashboardSessionWriter.publish_frame(...)` atomically refreshes status and the latest frame. `DashboardSessionWriter.record_event(...)` appends an event after the upstream policy has assessed it. The app reads only session files under `outputs/` and never constructs or loads a detector.
