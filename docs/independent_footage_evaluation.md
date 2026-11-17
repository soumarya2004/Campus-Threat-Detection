# Evaluate Person 1 on new footage

This checks the frozen item detector on new, manually labelled images sampled
from videos. It does not measure threat recognition, action recognition or
video-event accuracy. The existing `knife.mp4` and `rifle.mp4` have unverified
source provenance and were already inspected during development; keep their
results as qualitative demonstrations.

## 1. Record sources and freeze the plan

Choose permitted footage from recordings not used for training, threshold
selection or earlier model improvement. Prefer several different recordings,
locations and camera conditions, including ordinary objects that resemble the
five classes. New downloads are not automatically independent of the upstream
training dataset: record any uncertainty about overlap.

Create a small source ledger, for example `sources.csv`, with:

```text
source_id,recording_group,original_file,source_url_or_recorded_date,author,permission_or_license,prior_project_use,conditions
```

Use a stable ID such as `source_001`. Different clips from one original recording
must share a `recording_group`. Keep the licence/permission evidence with the
ledger. Record whether footage shows a real object or a prop, and report such
conditions honestly. Do not collect dangerous demonstrations just for testing.

Before inspecting predictions, record the promoted checkpoint hash and the
sampling rule. The small pilot rule is **12 midpoint-uniform frames across each
complete decoded clip**, or all frames when a clip has fewer than 12. Use the
same rule for every included source. Decide which sources to include before
running the model; never select only frames where it detects something.

```powershell
Get-FileHash models/dangerous_items_yolo11n.pt -Algorithm SHA256
```

Keep the deployed settings fixed: **640 pixels, confidence 0.45, NMS IoU 0.7**.
The matching IoU used to score a correct box is **0.5**. The original validation
decision is in `reports/experiments/validation_threshold.json`; the new footage
must not be used to tune these settings.

## 2. Extract unannotated samples

Replace the input path below with an actual new recording. Use a fresh output
directory and a distinct source ID for each clip:

```powershell
python scripts/extract_evaluation_frames.py path/to/new_recording.mp4 --source-id source_001 --output-dir outputs/independent_samples/source_001 --samples 12
```

The extractor loads no model. It first counts decoded frames, then selects
zero-based indices `floor((i + 0.5) * N / min(12, N))`. It saves original-size
PNG images and `source_manifest.json` containing video/image hashes, indices
and timestamps. It creates **no labels**. Nominal times assume constant FPS;
decoder times may be approximate. Inspect a large difference between reported
and decoded frame counts before accepting a source.

An extraction is complete only when its success manifest exists. An error can
leave partial images; use a fresh directory for a corrected rerun rather than
silently mixing outputs. Save the manifests and ledger with the evaluation.

## 3. Label before looking at predictions

Annotate the extracted source images, without detection overlays. Use the same
class IDs as the trained model:

| ID | Class |
|---:|---|
| 0 | machete |
| 1 | knife |
| 2 | baseball bat |
| 3 | rifle |
| 4 | gun |

For every included image, create a same-stem `.txt` file. Each visible target
gets one tight bounding box around its visible extent, clipped to the image
boundary. Include all visible instances. A baseball bat is still a target
object in a harmless sports scene: this is an object test, not an intent test.

Each YOLO label row is:

```text
class_id x_center/image_width y_center/image_height box_width/image_width box_height/image_height
```

All four coordinates must be finite and normalized to `[0, 1]`; box width and
height must be positive. **An empty label file means a reviewer explicitly
checked the image and found none of the five classes.** An absent file means unlabelled,
not background. Scissors, phones and umbrellas are backgrounds unless another
supported object is also visible.

If the class or box cannot reasonably be determined, log the image and reason
in `exclusions.csv` and exclude the entire frame before model inference.
Do not label an ambiguous object as background. Ask a second person to review
ambiguous cases and, for a small pilot, preferably all labels. Record reviewer
and label version. If an assistant drafts the labels, disclose that and keep
them provisional until human review; do not describe them as human-verified
ground truth. Preserve excluded samples in the extraction folder and
report how many were excluded; do not replace them with hand-picked successes.

Copy only the reviewed, included images and their labels into:

```text
data/external/independent_v1/
  data.yaml
  sources.csv
  exclusions.csv
  images/test/source_001_00000123.png
  labels/test/source_001_00000123.txt
```

Keep source IDs in filenames so results can be separated by recording.

## 4. Run the frozen detector

Save this exact configuration as `data/external/independent_v1/data.yaml`:

```yaml
path: .
train: null
val: null
test: images/test
names:
  0: machete
  1: knife
  2: baseball bat
  3: rifle
  4: gun
```

`train` and `val` are null because this is an evaluation-only dataset; the
installed Ultralytics parser requires those keys, but no training/validation
images are needed for this command. The evaluator resolves `path` relative to
the YAML. Use an explicit frozen threshold, not `--threshold-file`: the old
threshold file is correctly tied to the original validation dataset.

```powershell
python scripts/evaluate_detector.py --model models/dangerous_items_yolo11n.pt --data data/external/independent_v1/data.yaml --split test --threshold 0.45 --imgsz 640 --iou 0.5 --batch 1 --workers 0 --output reports/experiments/independent_v1.json
python scripts/review_model_errors.py reports/experiments/independent_v1.json --output-dir outputs/independent_v1_error_review
```

Add `--device cpu` if needed. Preserve the checkpoint hash and a note linking
the frozen 0.45 value to the existing validation decision, since explicit CLI
thresholds do not automatically copy that decision into the report.

The generic dataset audit and grouped-split verifier assume training,
validation and test folders, so they are not the right preflight for this
evaluation-only YAML. Check every image/label pair and the exclusion ledger
before running. The evaluator rejects missing or malformed labels; do not
convert missing files into empty files just to pass that check.

## 5. Report the pilot honestly

Report source/recording counts, sampled/included/excluded frames, class-instance
counts and background counts first. Then give fixed-threshold precision,
recall and F1, per-class misses/false positives, and the number of background
images with false positives **with its denominator**. Separate results by
source or recording group as well as providing the aggregate.

Report unsupported classes as **not measured**. For example, knife footage
plus normal scissors and arrival scenes can test knife detection and those
background conditions; it cannot establish five-class external accuracy.
If a subset has no target objects, object recall and mAP do not describe its
performance: use background false-positive counts instead.

Frames from one clip are correlated. Thirty-six frames from three clips are
three source videos, not 36 independent scenes. Do not interpret a sampled-frame
false-positive rate as false alerts per minute, or infer event recall or time
to detection from this sparse sample. Keep these results separate from the
grouped dataset benchmark and do not retune after seeing them. Any subsequent
improvements need a new held-out recording set for an independent final check.

## Reproduce this local pilot

The recorded pilot used the three sources in
`reports/experiments/external_pilot_sources.json`: steak cutting, scissors
cutting cardstock, and an arrival walkway. That ledger includes download URLs,
authors, licences and exact video hashes. The annotation ledger is
`reports/experiments/external_pilot_annotations.json`. All 36 frames were kept;
six have boundary-ambiguity flags. **All annotations remain provisional and
need human review.** The frozen run returned 1 true positive, 4 false positives
and 11 missed objects against those labels. Only knife positives were measured;
the other four classes have no positive coverage. Older-source overlap remains
unproven despite the recorded heuristic overlap screen.

These commands replay the saved inputs and labels; they do not produce another
independent test. Run from the repository root with the project's Python
environment active. Keep the source ledger's attribution/licence information
with any shared frames. The first block downloads a video only if missing,
verifies its hash, then extracts into a fresh sampling directory:

```powershell
if (Test-Path -LiteralPath outputs/external_pilot/reproduced_samples) { throw "Use a fresh sampling directory." }
$sources = (Get-Content reports/experiments/external_pilot_sources.json -Raw | ConvertFrom-Json).sources
foreach ($source in $sources) {
    $video = $source.local_file
    if (-not (Test-Path -LiteralPath $video)) {
        New-Item -ItemType Directory -Path (Split-Path $video) -Force | Out-Null
        Invoke-WebRequest -Uri $source.download_url -OutFile $video -UserAgent "CampusThreatDetectionStudentProject/1.0" -ErrorAction Stop
    }
    if ((Get-FileHash -LiteralPath $video -Algorithm SHA256).Hash.ToLowerInvariant() -ne $source.sha256) { throw "Video hash mismatch: $video" }
    python scripts/extract_evaluation_frames.py $video --source-id $source.source_id --output-dir "outputs/external_pilot/reproduced_samples/$($source.source_id)" --samples 12
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $video" }
}
```

Assemble a **fresh** `data/external/public_video_pilot/reproduced` dataset from
the frozen annotations. This checks the promoted model, every image and every
generated label against the recorded hashes before creating the dataset.
Labels use eight decimal places and explicit UTF-8/LF bytes; reviewed background
frames receive zero-byte files. The original dataset and reports are preserved.

```powershell
@'
import hashlib, json, shutil
from pathlib import Path
import cv2, yaml
digest = lambda data: hashlib.sha256(data).hexdigest()
source_file = Path("reports/experiments/external_pilot_sources.json")
label_file = Path("reports/experiments/external_pilot_annotations.json")
sources = json.loads(source_file.read_text(encoding="utf-8"))
records = json.loads(label_file.read_text(encoding="utf-8"))["records"]
output = Path("data/external/public_video_pilot/reproduced")
if output.exists():
    raise SystemExit("Use a fresh dataset directory; existing output is preserved.")
assert digest(Path(sources["model"]).read_bytes()) == sources["model_sha256"], "Model changed"
prepared = []
for record in records:
    image = Path("outputs/external_pilot/reproduced_samples") / record["source_id"] / "images" / record["image"]
    assert digest(image.read_bytes()) == record["image_sha256"], f"Image changed: {image}"
    height, width = cv2.imread(str(image)).shape[:2]
    assert (width, height) == (record["width"], record["height"])
    assert record["annotations"] or record["explicit_background_review"], "Unreviewed empty label"
    rows = []
    for item in record["annotations"]:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        values = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
        rows.append(str(item["class_id"]) + " " + " ".join(f"{value:.8f}" for value in values))
    label = ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")
    assert digest(label) == record["label_sha256"], f"Label changed: {image}"
    prepared.append((image, label))
assert len(prepared) == 36
(output / "images/test").mkdir(parents=True)
(output / "labels/test").mkdir(parents=True)
for image, label in prepared:
    shutil.copy2(image, output / "images/test" / image.name)
    (output / "labels/test" / image.with_suffix(".txt").name).write_bytes(label)
config = {"path": ".", "train": None, "val": None, "test": "images/test", "names": {0: "machete", 1: "knife", 2: "baseball bat", 3: "rifle", 4: "gun"}}
(output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
shutil.copy2(source_file, output / source_file.name)
shutil.copy2(label_file, output / label_file.name)
print("Prepared 36 hash-checked frames with unchanged provisional labels.")
'@ | python -
```

If decoding with a different library build changes an image hash, stop and
investigate; do not update the ledger to make the check pass. Reuse the recorded
environment or review and version the changed inputs separately. Do not use
Python's `-O` flag here, because it disables the assertions.

Finally, evaluate once at the same frozen settings and write to a new report:

```powershell
python scripts/evaluate_detector.py --model models/dangerous_items_yolo11n.pt --data data/external/public_video_pilot/reproduced/data.yaml --split test --threshold 0.45 --imgsz 640 --iou 0.5 --batch 1 --workers 0 --output outputs/external_pilot/reproduced_evaluation.json
```

The snippets were syntax-checked and their label formatting was checked against
all 36 frozen labels. Replaying downloads, extraction and inference is left to
the person reproducing the experiment; it must not be described as a second
completed evaluation until actually run.
