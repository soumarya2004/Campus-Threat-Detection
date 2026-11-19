# Tracking & Interaction Analysis — Project Audit

**Audit date:** 17 November 2026  
**Submission context:** Seventh semester, five credits  
**Readiness assessment:** **42/100** on the proposed rubric below. The build target is **95/100**, subject to the acceptance evidence under each category. The department's marking rubric was unavailable.

The notebook demonstrates person tracking, detector-triggered tracking, and image-plane movement and proximity calculations on a ten-second pedestrian clip. The Desktop workspace is an incomplete export of that prototype. Its main script fails under the workspace environment, three intended modules are empty, and the saved analytics can combine IDs from different tracking events. A credible submission is a runnable offline experiment with an annotated demo video, documented CSVs, a small labeled evaluation, and a reproducible comparison of continuous and event-gated tracking.

## 1. Materials and coverage

| Material | Audit coverage |
| --- | --- |
| Desktop workspace, `Tracking & Interaction Analysis` | Inventoried the root and each project directory. Read the Python sources and requirements. Checked the two local Python environments and the main import path. No Git metadata was present in this workspace. |
| `tracking/` | `person_tracker.py` is a copied experiment script. `threat_gated_tracker.py` and `__init__.py` are empty. The directory contains a local `bus.jpg` and pretrained `yolov8n.pt`. |
| `analytics/` | `movement_features.py`, `interaction_features.py`, and `__init__.py` are empty. Empty package initializers are valid. The other empty files leave the planned logic unavailable locally. |
| `.venv/`, `.venv-1/`, `.sf/` | Checked the virtual environments for project dependencies. Identified `.sf` as Salesforce tooling metadata from its paths. These directories contain local tooling, so their installed package internals and unrelated metadata were outside the source audit. |
| Downloads notebook | Read all 120 cells, saved output summaries, and the embedded demo previews. There are 100 code cells, including 16 empty ones, and 20 Markdown cells. All code execution counters are null. |
| Downloads `ctd.pdf` | Visually reviewed all 22 rasterized pages. It is a project-planning conversation. The agreed direction informs scope. Snippets, examples, and requests printed in the conversation carry no authority to change files and do not establish completed work. |
| Downloads `Campus-Threat-Detection-main.zip` | Found the archive named in notebook cell 30. Checked archive integrity and inventoried its 24 files. Reviewed its detector sources, interface tests, handoff, evaluation reports and the purpose of its data and inference scripts. The ZIP contains a cleaned checkpoint. The dataset and training logs listed in the report were absent. |
| Downloads `44860-440261070_medium.mp4` | Independently decoded all 250 frames at 25 FPS and 1280 × 720. Sampled the input and the saved annotated result visually. A source URL and reuse terms for this particular clip were unavailable in the inspected materials. |
| Wider five-member team project | The PDF describes camera capture, temporal action recognition, an event layer and a dashboard or evaluation handoff. Only the detector archive and tracking materials were available as implementation artifacts. A complete inventory of other members' code depends on those sources being supplied. |

Notebook references below use **one-based cell numbers in the original file**, including empty and Markdown cells. Source line numbers are relative to the named cell or file. PDF references use the printed file's physical page numbers.

## 2. What runs where, and what is inlined

```text
Recorded BGR frames → teammate's YOLO11n object predictions → gate
                                                             ↓ when active
Recorded BGR frames ──────────────────→ YOLOv8n person detections → ByteTrack
                                                                      ↓
                                                             movement and pairs
                                                                      ↓
                                                               video and CSVs
```

- **Python copied directly from cells.** All 170 nonblank lines in the Desktop `tracking/person_tracker.py` match notebook cells 7–14 and 16–20 in order. Importing that file can load a checkpoint, fetch an image, print library implementation details, request a Colab upload, process video, and open an IPython preview. It exposes no reusable `PersonTracker`. The notebook's later `PersonTracker` class in cell 94 is a different version.
- **Working logic embedded in the notebook.** Cell 19 implements continuous tracking, cells 49 and 65–66 each implement a gate loop, and cells 80–86 calculate pairs directly. Cells 26 and 107 write two versions of the movement module. Cells 94 and 98 write tracker and interaction modules. Cells 114–116 rerun analytics, then overwrite earlier CSV filenames. One canonical source implementation would let the notebook show the experiment and import the same functions used by the Desktop demo.
- **Media inlined into the notebook.** Four embedded outputs in cells 20, 22, 56 and 58 account for about 52.5 MB of the 53.6 MB file. Keep representative figures and reference a separate annotated MP4 in the submission. The source notebook can serve as the original experimental record.
- **Unrendered content printed in the PDF.** Page 16 prints schedule template code such as `{#each … as item}` and quoted data fields. Pages 17–18 print interactive checklist code. Rebuild these as a static week/deliverable table and Markdown checklist when drafting the formal submission. Example payloads on pages 7 and 9 need fenced code and explicit definitions of units and fields.

## 3. Findings and corrections

**P0** items affect whether the examiner can run the project or whether results are valid. **P1** items affect the quality of the evidence. **P2** items improve submission clarity.

| ID | Finding and evidence | Correction and acceptance check |
| --- | --- | --- |
| P0-1 | **Desktop startup fails.** `.venv` raises `ModuleNotFoundError: trackers` at `tracking/person_tracker.py:3` while `check_environment.py:1–14` prints success. `.venv-1` lacks OpenCV. With dependencies present, the script still uses a working-directory weight path (line 8), a remote sample image (27), Colab upload (116–123), and notebook display (252–254). | Add one import-safe local entry point and explicit input and weight paths. A documented run from a fresh environment should work from the project root and with a path containing spaces. The environment check should exercise the project dependencies and clarify failures. |
| P0-2 | **Dependency and execution provenance differ.** Workspace requirements pin Ultralytics 8.4.202 and Supervision 0.31.4. The archive pins Ultralytics 8.4.175. The notebook asks for Supervision 0.31.4 in cell 5 but prints 0.31.5 in cell 6. The Desktop `.venv` has 8.4.210 and 0.31.6. Cell 2 imports before the full install in cell 5. Colab upload and `/content` paths in cells 16 and 30–34 also prevent an unchanged Windows run. | Verify and record one combined environment, including the tracker version. Move installs out of the notebook. Use explicit local asset paths and one reproducible example command. Record model and dependency versions with each run. |
| P0-3 | **Track IDs collide across events.** Cells 49 and 66 instantiate a fresh ByteTrack at each rising edge. Cell 66 logs bare IDs. Cells 78, 80–82 and 114 group or reprocess using those IDs. I confirmed that two fresh tracker instances both assign ID `0`. Cell 73 records ID 5 at frames 17 and 81, on opposite sides of the first reset. | Key tracks and pairs by `event_id` plus tracker ID. Keep raw IDs for traceability. Group both analytics and summaries inside that scope. Test two events that intentionally start with the same raw IDs. |
| P0-4 | **Pair ordering and gaps distort the interaction labels.** Cells 80 and 98 take the order of input rows for A and B. The approach classifier in cells 82/98 can compare nonconsecutive observations. Isolated checks reproduced a reversed pair recorded separately, a six-frame gap labeled as movement, and a self-pair from duplicate IDs. The saved 181 first-pair rows exceed the 136 possible unordered distinct pairs among 17 raw IDs. | Canonicalize pair IDs, enforce one observation per scoped track per frame, and classify change only for adjacent valid observations. Export `insufficient_data` at starts and gaps, and document the distance-change unit. Verify reversed ordering, duplicates, and a missing observation. |
| P0-5 | **Time and I/O can silently invalidate outputs.** `person_tracker.py:159–199,242–249` reports “Output saved” after an unopened input in an isolated stub test. A readable input returning zero FPS raises before release. The writer is unchecked. Cells 86, 103 and 115 assume 25 FPS. At the 0.8 s gate boundary, the saved trigger replay excludes frame 77 because subtraction gives 0.8000000000000003 s. | Validate capture, FPS and writer before processing, release resources on errors, and verify the resulting video can be decoded. Use source timing consistently. Define the inclusive grace boundary with a frame-based rule for constant-FPS inputs or a clearly tested timestamp comparison. Test a missing file, zero FPS, expiry at the exact boundary, a non-25-FPS clip, and an empty but valid run. |
| P0-6 | **Output schemas drift.** Cell 76 calls a numeric filtered speed `speed_valid`. Cell 107 uses the same field as a boolean and renames `speed_raw` to `raw_speed`. Cells 90 and 116 write the same filenames. A no-schema empty table raises in cell 98 code, and a one-track run can return a columnless pair table. | Define stable columns and units for each CSV, with headers even when empty. Keep `speed_px_s`, `raw_speed_px_s`, a boolean validity field, and an invalid reason distinct. Export into a per-run directory and check row counts, keys and types on readback. |
| P1-1 | **The person-tracker wrapper does not reproduce the prototype settings.** Cell 94 starts with `self.model = None`, predicts at 0.35, and initializes ByteTrack with package defaults. Cells 49 and 66 use person confidence 0.25, tracker activation threshold 0.25, and one minimum consecutive frame. The wrapper in cell 95 was only imported. ByteTrack 2.7.0 accepts the wrapper's `frame` argument. | Initialize the model explicitly and share one documented person/tracker configuration. Exercise a real `detect_and_track` call in an integration run. Feed ByteTrack detection scores deliberately and explain the effect of an upstream confidence cutoff on its low-score candidates. The method's association design is described in the original [ByteTrack paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf). |
| P1-2 | **Movement filtering flags outliers but leaves contaminated geometry and history.** Cells 72–73 include a 4,512.9 px/s jump. Cell 107 records new history even when a step exceeds 600 px/s. In a synthetic jump-and-recovery sequence, this invalidated the recovery and produced a later 180° direction change. Cell 114 reconstructs boxes from stored centers and reuses bare IDs. | Preserve raw observations, mark initial/gap/outlier states, and define how a rejected point affects velocity, direction and consecutive pair observations. Use bounding-box center displacement as an image-plane feature, in px and px/s. Show one failure example and how the corrected pipeline handles it. |
| P1-3 | **The gate measures model outputs, with errors and ambiguous positives.** Cells 49/66 trigger on any class from Member 1. A single error can interrupt an active event and cause another tracker reset once inference resumes. Sampled red boxes in the saved overlay are ambiguous on the pedestrian clip. There are no event labels in the supplied results. | Log trigger predictions, inference errors and state transitions separately. Document the gate's error behavior and test it. Treat all predictions as candidate object detections. Validate event behavior with prerecorded or synthetic cases, including ordinary footage, short missed-detection gaps and controlled positive trigger inputs. |
| P1-4 | **Evidence of correctness and performance is limited.** One unlabelled ten-second clip supports a smoke demonstration. Its media FPS is 25. The local gated rerun took about 71 seconds on CPU. Gating reduces the number of person inference calls while the trigger model still runs on every frame. There is no paired runtime comparison, identity ground truth, interaction labels, or application-level accuracy study. | Compare continuous and gated tracking over the same hand-labeled short segments and hardware. Measure end-to-end wall-clock throughput with startup recorded separately, track fragmentation/ID switches on sampled spans, and agreement with labeled proximity/approach windows. Keep any accuracy claim specific to the labeled sample. [Tracking metrics and ground truth](https://link.springer.com/article/10.1007/s11263-020-01375-2) |
| P1-5 | **The detector's dataset result has limited traceability.** The ZIP report names a cleaned split and checkpoint, but the dataset, split manifest and training logs are missing from the archive. Its scripts check exact SHA-256 duplicates. Exact hashes cannot establish whether near-duplicates or source-related samples cross splits. `data.yaml` points to raw data, while the report names a cleaned path and the verification script expects a deduplicated path. | Use the already trained checkpoint for the demo and credit Member 1. Include its checksum, dataset provenance, and reported evaluation as component evidence. Obtain the exact processed split manifest and logs only if defending training or attempting reproduction. Make the split path explicit and keep the original and cleaned numbers separate. [Original dataset record](https://zenodo.org/records/16422779) |
| P2-1 | **Preview and documentation are inconsistent.** Cell 50 previews a file before cell 55 creates it, resulting in malformed saved HTML. Cell 119 opens a Markdown fence at line 19 and never closes it. Steps are duplicated or out of sequence, and notebook strings such as “Friend's detector” are informal. Python quote/syntax checks did not identify a quote defect in the parsed Python cell bodies. | Close the fence, use simple headings and consistent names, and show the demo after generating the video. Replace placeholders with member attributions and a real interface contract. Keep code string literals intact where Python requires them. |

## 4. What the results support

| Evidence | What can be reported |
| --- | --- |
| Notebook cells 17, 19 | The clip contains 250 frames at 25 FPS. The continuous pass logged 2,375 per-frame person detections and 23 distinct raw tracker ID values. Those ID values can include fragments of the same person. |
| Cells 44, 49 | The saved trigger pass logged 29 predicted object boxes across 25 frames. The gate called the person stage on 209 of 250 frames, or 83.6%, across three windows. That is 41 fewer person-stage calls than continuous tracking on this clip. Full pipeline compute savings require measurement. |
| Cell 44 log, replayed against cell 49 | Reconstructed active ranges of frames 2–76, 80–147, and 154–219 inclusive, matching all 209 saved active frames. Applying an exact inclusive boundary to the same frame data gives 210. These are replays of saved predictions. |
| Independent rerun of cell 49 | Ran the two supplied checkpoints on CPU with Python 3.13, Ultralytics 8.4.202, Supervision 0.31.4 and Trackers 2.7.0. Only the input/model bindings and temporary output path were adapted. The rerun processed 250 frames, produced a decodable 250-frame MP4, and recorded 24 trigger-positive and 207 active frames across five windows, including one still active at clip end. Cell execution took 71.04 s, excluding model initialization and output readback, or about 3.5 processed frames/s. The saved run had 25 trigger-positive and 209 active frames. The cause of the difference was unverified. |
| Cells 66, 69–76, 114 | Saved output includes 1,622 track observations. In the first pass 39 had no speed and 69 further values exceeded 600 px/s, leaving 1,514 accepted values. Later reanalysis reports 1,519 accepted and 103 invalid observations under a different calculation and schema. Neither count validates the threshold. |
| Cells 80–87, 115 | The notebook reports 5,678 pair observations, including 2,777 stable, 1,646 approaching, 1,074 moving apart, and 181 insufficient. There are 153 proximity episodes at 150 px. The longest saved episode spans 55 frame observations, or 2.20 s using the notebook's inclusive convention. Treat these as baseline output to regenerate after fixing event IDs and pair continuity. |
| Detector ZIP, `reports/person1_clean_model_evaluation.md:52–72` | Member 1 reports precision 0.878, recall 0.815, mAP@50 0.882, and mAP@50–95 0.616 on 1,220 test images and 1,273 instances. These are reported results for the cleaned object detector on its image split. The original dataset record gives 8,478 images. The archive's cleaned split counts total 8,302. [Original dataset record](https://zenodo.org/records/16422779) |

The source video shows an uncalibrated pedestrian scene with perspective and occlusion. All proposed movement outputs describe box geometry in the image plane. An approach label means decreasing projected distance between two tracks. A proximity episode means observed closeness under a documented pixel threshold. A confirmed incident or physical distance would require different evidence.

## 5. Report and wording revisions

- Use **Event-Gated Person Tracking and Interaction Analysis** as the tracking contribution title. Retain the team project's title separately and state where this contribution plugs in.
- Change **“Unique people detected”** to **“Distinct tracker IDs observed”** after implementing event-scoped IDs. Report true headcount only if it has been independently annotated.
- Change **“Weapons detected”** to **“Candidate object detections”** in the experiment results. Keep the class name and confidence as model outputs so ambiguous examples can be reviewed.
- Change **“Real-time performance”** to **“Processing throughput on the recorded clip”** until an end-to-end timing run measures the required live input rate on stated hardware. Distinguish the clip's FPS from measured throughput.
- Replace **“Social interaction”** and physical speed claims with **“Image-plane approach, separation and proximity”** and pixel units. State the 5 px distance-change threshold, 150 px proximity threshold, 600 px/s validity cutoff, and 0.8 s grace duration as configured heuristics wherever they are relevant.
- Reconcile dataset assignments in PDF pages 5–6 and 18–22. Record the original Zenodo dataset and cleaned Member 1 split. Cite external tracking or action datasets as planned work unless the team can provide runs. MOT17 is relevant to identity tracking and cannot serve as ground truth for pairwise interaction labels. Attribute ByteTrack, YOLOv8n, YOLO11n, the detector checkpoint, the dataset, and all shared footage. [YOLOv8](https://docs.ultralytics.com/models/yolov8/) · [YOLO11](https://docs.ultralytics.com/models/yolo11/) · [Dataset and requested paper citation](https://zenodo.org/records/16422779)
- Remove chat headers, avatars, URLs and clipped template fragments from the formal hand-in. Convert the estimated team percentages on PDF pages 1–2 to a contribution table with artifacts and owners. Add captions stating the clip, method, frame or timestamp, and units.

## 6. Proposed complete Desktop build

Use one VS Code folder named `Tracking & Interaction Analysis - Build`. The tree shows the proposed deliverable. The archive's detector and tests supply implementation code. The tracking package supplies the missing pipeline. Keep the original notebook as supporting material and make a compact walkthrough that calls the new package.

```text
Tracking & Interaction Analysis - Build/
├── README.md
├── requirements.txt
├── .gitignore
├── .vscode/
│   └── launch.json
├── src/
│   ├── detector.py
│   ├── models/object_detector.py
│   └── tracking_analysis/
│       ├── __main__.py
│       ├── pipeline.py
│       ├── person_tracker.py
│       ├── gate.py
│       ├── movement.py
│       └── interactions.py
├── assets/
│   ├── README.md
│   └── models/
├── notebooks/
│   └── demo.ipynb
├── tests/
├── evaluation/
│   ├── protocol.md
│   └── results.md
└── outputs/
    └── example/
```

The demo should process a recorded video with one command and offer continuous tracking and detector-gated tracking. A small, explicitly sourced replay of trigger inputs can demonstrate correctness without relying on detector behavior. The output for a run should include one playable MP4, a lightweight run summary with configuration and errors, and movement, pair and proximity CSVs with stable schemas. Document how to supply the required weights and footage. Include redistributable example assets where provenance supports it, and provide a synthetic fallback fixture for the demonstration and tests.

The five-credit core is local recorded-video analysis, correctness checks, a baseline comparison and an explainable demo. A live camera, learned action recognition, multi-camera tracking or a dashboard would need their own evidence. Fold in an available teammate module when it has a tested contract and helps the submission. A static timeline and results table are sufficient to present the tracking experiment.

## 7. Work order and acceptance criteria

| When | Work | Evidence to finish the stage |
| --- | --- | --- |
| 1. Establish the baseline | Copy reusable sources into the new build, choose and record one verified environment, define paths, and run continuous tracking on a small clip. | VS Code launch and one documented command produce a playable video and a run summary from a fresh environment. |
| 2. Correct integration and math | Put the gate, tracker and analytics into importable modules. Introduce event-scoped IDs, canonical pairs, timestamp rules, stable CSV headers and invalid reasons. | Tests exercise event resets, exact grace expiry, duplicate and reversed pairs, timing gaps, motion outliers, non-25-FPS input, empty outputs, and an invalid input. |
| 3. Evaluate | Use three to five short, permission-cleared or synthetic cases covering ordinary footage, scripted triggers, occlusion/crossing and known pair motion. Split examples used for choosing thresholds from examples reserved for reporting. Compare the two tracking modes on the same material. | Include clip IDs and annotations, pass/fail for gate transitions, checked ID continuity and interaction agreement, detector errors, active frame counts, hardware, runtime, and two or three failure examples. A small selective benchmark is appropriate for the submission. |
| 4. Package and rehearse | Finalize the reference list, attribution, report language, notebook preview, example outputs and a short viva script. Have another person follow the README without assistance. | Three-to-five-minute demonstration of one successful trace and one known limitation. The examiner can trace an observed frame to its scoped track, pair row and video overlay. |

## 8. Readiness score and path above 90

These are evidence-based readiness points under a proposed academic rubric. The target column is conditional on completing the checks and receiving credit for them. A review of the corrected build can replace the estimates with a measured score.

| Criterion | Available | Current | Target | What earns the target |
| --- | ---: | ---: | ---: | --- |
| Academic problem and scope | 10 | 7 | 10 | State a research question, define the tracking contribution and make the limits explicit. |
| Reproducibility and VS Code structure | 15 | 4 | 15 | A clean machine follows the README and runs the local demo with documented versions and assets. |
| Detection, tracking and gate integration | 20 | 12 | 19 | Versioned model/configuration provenance, event-scoped IDs, documented error behavior and a verified video trace. |
| Movement and interaction correctness | 20 | 9 | 19 | Units, tested gap semantics, stable schemas, canonical pair IDs, and examples with known expected outputs. |
| Evaluation and interpretation | 20 | 4 | 18 | Labeled clips, continuous/gated comparison, actual throughput measurements, failure examples and candid limitations. |
| Tests and error handling | 7 | 2 | 7 | Passing component and integration tests, including missing/invalid inputs, empty outputs and boundary cases. |
| Report quality and attribution | 5 | 3 | 4 | Formal report, accurate language, consistent evidence and complete citations for included assets. |
| Demo and viva | 3 | 1 | 3 | Repeatable walkthrough tracing a trigger to video and CSV evidence and explaining a failure case. |
| **Total** | **100** | **42** | **95** | **A working build plus defensible evaluation.** |

## 9. Verification record

- Parsed the workspace Python files and executable notebook code bodies after excluding IPython shell, write and Markdown directives. Checked archive integrity and parsed its Python files. These are syntax checks.
- Ran the workspace environment check and import through both local environments. Used isolated stubs to exercise the workspace video loop and ran focused tests of the notebook's movement and interaction classes. Verified tracker ID reuse against ByteTrack 2.7.0.
- Checked the saved detector log against gate arithmetic, decoded the supplied input, and reviewed sampled frames from both the input and the notebook's saved annotated video. No ground-truth labels were available for the pedestrian clip. A detector training reproduction also requires the prepared dataset and training records.
- Ran the archive's detector-interface tests against a fake detector in an isolated scratch directory. **67 passed**. This verifies the wrapper contract in the audit environment. I also ran the original cell 49 gate loop end to end with the real checkpoints and verified the output frame count. The isolated gate run and wrapper tests leave the source files untouched.
- The original notebook requires interactive Colab uploads and hard-coded paths. Its full top-to-bottom run was not verified. The correction phase should exercise the local entry point independently, then run the cleaned notebook in order with explicit inputs and verify the generated files.

## 10. Independent judges' review

| Review | Current readiness | Build plan | Focus |
| --- | ---: | ---: | --- |
| Engineering judge | 40/100 | 91/100 | Scored cross-event data corruption and the lack of proof from an isolated environment more strictly. Requires known-answer timing, identity and I/O tests and a paired runtime measurement. |
| Academic judge | 43/100 | 88/100 | Credited the real checkpoint rerun and wrapper tests. Asked for a narrower research question, a predeclared set of annotated frames, explicit denominators, and sample failures. |

The two scores assess the current evidence and the proposed work. A score for the completed build requires its own review. The engineering judge estimated two to three focused workdays. The academic judge estimated 16–24 student-hours, including annotation, evaluation, and viva preparation.

The build will address the judges' highest-priority requirements in this order.

1. Verify the entry point and error behavior, including a path with spaces and a model-free example.
2. Run known-answer cases at a different FPS and verify the exact gate boundary, event-scoped identities, pair continuity, outlier recovery, empty exports and playable annotated video.
3. Compare both modes on the same inputs and hardware. Count person detector calls and record end-to-end wall time. Report coverage over all annotated frames and association on comparable tracked frames. Keep known-answer cases and real inference results distinct.
4. Document one ambiguous detector trigger and observed tracking failure or fragmentation. Provide a short viva trace from a specific frame through its event, track and pair rows.
# Archived pre-merge audit

This document reviews the tracking component before it was moved into the
shared repository. Its historical paths refer to the standalone build; current
tracking code lives in `src/tracking_analysis/` and its assets in
`assets/tracking/`.

