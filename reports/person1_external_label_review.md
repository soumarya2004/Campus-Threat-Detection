# External pilot label review: 17 November 2026

All **36 images and label files match the frozen annotation manifest**. The
boxes, dimensions, class IDs and YOLO coordinates are internally consistent.
Visual review supports the 24 background labels and the presence of knives in
the 12 positive frames. Six knife boxes still have uncertain boundaries.

This is a dated **assistant review**, not human approval. It checks existing
pilot material after the original evaluation. No labels, model settings or
original scores were changed. The
[review record](experiments/external_label_review_20261117.json) contains the
hashes and individual observations for every frame.

## What was inspected

The review inspected all 36 raw frames in source contact sheets, then the
12 knife-label overlays. Five ambiguous knife frames were also opened
individually at native size. Original annotation notes and aggregate results
were already known, so this is not a blinded review. Predicted boxes were not
used to revise labels.

| Source | Frames | Finding |
|---|---:|---|
| Steak slicing | 12 | Knife presence is plausible in every frame. Six boxes are plausible visible-extent labels; six retain their original ambiguity flags. |
| Scissors cutting card | 12 | No object from the five target classes is visible. Scissors are outside this model's label set. The first two sampled frames show almost blank tabletop. |
| Arrival scene | 12 | Paving, parked vehicles, plants and street furniture; no visible target object in the sampled frames. |

The flag on `steak_00001411.png` needs particular attention: its right box edge
at x=800 appears to include substantial occluding-hand area rather than visible
knife. A reviewer should resolve the visible-extent convention before treating
that box as a precise reference. No replacement coordinates are proposed here.
Other flagged frames contain motion blur, an edge-on blade, or uncertain
endpoints behind the hand/meat.

## Does boundary ambiguity explain the poor score?

The original `boundary_ambiguity` flags were recorded before inference. Grouping
the saved prediction matches by those flags gives this **post-hoc diagnostic**:

| Original label group | Frames | TP | FP | FN |
|---|---:|---:|---:|---:|
| Knife frames without a boundary flag | 6 | 0 | 1 | 6 |
| Knife frames with a boundary flag | 6 | 1 | 2 | 5 |
| Background frames | 24 | 0 | 1 | 0 |

These counts reuse the original confidence 0.45 and matching IoU 0.5. They are
not a new accuracy estimate, a new test set, or a corrected evaluation. None
of the six unflagged knife frames matched, so the generalization problem is
not confined to the uncertain boundaries. The original **1 TP / 4 FP / 11 FN**
result remains the pilot result.

## Reproduce the checks

From the repository root:

```powershell
python scripts/review_external_annotations.py --output outputs/external_review_reproduction
python scripts/review_external_annotations.py --evaluation outputs/external_pilot/evaluation_full.json --output outputs/external_review_diagnostic_reproduction
```

The first command verifies image/label hashes and renders raw and label-only
sheets without loading a detector. The second additionally groups matches from
the full saved evaluation. Choose fresh output directories; existing review
files are preserved. The second command requires the local full evaluation,
whose hash is recorded in the review JSON.

Prepared sheets: [knife frames 1-6](../outputs/external_label_review_20261117/steak_1_labels.jpg),
[knife frames 7-12](../outputs/external_label_review_20261117/steak_2_labels.jpg).
The corresponding `_raw.jpg` files contain no boxes. These outputs are generated
locally; the script recreates them from the source dataset.

Source authors, licenses and video hashes remain in the
[source ledger](experiments/external_pilot_sources.json). Twelve frames from one
video are correlated, and four positive classes are unmeasured. Human label
adjudication and broader source coverage would improve the evidence; this
assistant review does not establish campus performance.
