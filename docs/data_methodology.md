# Person 1 dataset preparation and evaluation boundary

The detector is trained to locate five item classes: machete, knife, baseball
bat, rifle and gun. Detecting an item does not classify a person's intent.

## Why the split changed

The old `dangerous_items_cleaned_v2` dataset removed byte-identical images
crossing train/validation/test boundaries. However, it still placed originals
and their offline augmentations in different splits. For example,
`train/0141184a-0419.jpg` and `test/A16_0141184a-0419.jpg` show the same frame;
the latter is blurred. An exact file hash cannot find this relationship.

Removing the `A<number>_` filename prefix identifies 434 inferred source-image
families shared by training and test, covering 580 of the 1,220 test images.
These are filename-derived family counts, with sampled pairs visually checked;
they are not a claim that every pair was manually inspected. The old 0.882
mAP@50 is an old-split result, not an independent benchmark for the new model.

## Grouped v1 procedure

`scripts/build_grouped_dataset.py` reads the cleaned_v2 dataset without changing
it. It validates image decoding, matching label files, class IDs and box bounds.
It then:

1. Joins images with the same base filename after removing `A<number>_`
   prefixes. It also joins byte-identical images by SHA-256. Connections are
   transitive: duplicate images with different names bring both augmentation
   families into the same group.
2. Excludes a whole group if byte-identical images have conflicting annotations.
   Comparison ignores row order and rounds numeric fields to six decimals, so
   insignificant text formatting differences do not become label conflicts.
3. Selects one representative per group. An original without an augmentation
   prefix is preferred; otherwise a deterministic filename order selects a
   fallback, which is counted in the summary. This removes repeated offline
   augmentation copies from all three splits.
4. Assigns whole groups using seed 42 and 70%/15%/15% image quotas. A simple
   greedy cost balances class-instance counts and background images across the
   splits. This is approximate stratification, not a guarantee of exact counts
   for every class. The summary records the achieved counts.
5. Copies the selected image/label pairs into a fresh dataset and writes a
   portable manifest. YOLO's configured online augmentation is applied only to
   training images; validation and test are evaluated without augmentation.

The CSV manifest records **every input image**, not just selected images. Its
paths are relative to the source dataset root. It contains image and label
hashes, source family, connected group, assigned split, selection status and
exclusion reason. The JSON summary records the seed, counts, method, limitations
and manifest SHA-256. Keep these small reports in version control; the image
dataset can remain local.

The recorded v1 build selected 4,667 images: 3,267 training, 700 validation and
700 test. It excluded 86 connected groups containing 229 source images because
their exact-copy annotations differed. Many differences were slightly different
box coordinates, so exclusion does not imply that all these labels had wrong
classes. There were 128 groups without an unprefixed original. Background image
counts were 347/74/74 respectively. The full counts and manifest checksum are in
`reports/dataset/grouped_v1_summary.json`.

## Reproduce from the downloaded dataset

Arrange the original dataset using root `data.yaml`, then run from the
repository root in the project virtual environment:

```powershell
python scripts/build_deduplicated_dataset.py --data data.yaml
python scripts/fix_rounding_boxes.py
python scripts/build_grouped_dataset.py
python scripts/verify_clean_splits.py --manifest reports/dataset/grouped_v1_manifest.csv
```

The first command creates `dangerous_items_deduplicated`; the second copies it
to `dangerous_items_cleaned_v2` and corrects out-of-bounds rounding errors; the
third creates `dangerous_items_grouped_v1`. If cleaned_v2 already exists and
has been audited, start at the third command. The scripts refuse to overwrite
existing datasets. The grouped builder reuses checked-in reports only when its
newly calculated manifest and summary match them byte for byte; it does not
rewrite those reports. A mismatch stops before copying any dataset files.
Use a fresh checkout/output location when reproducing an existing build; do
not delete the only copy of an experiment to rerun a command.

Source label newlines can differ across operating systems because the earlier
rounding correction script writes native newlines. The selected groups can be
the same while label-byte hashes differ. Record such a separate build with
`python scripts/build_grouped_dataset.py --report-dir outputs/reproduced_dataset`
and verify it with `--manifest outputs/reproduced_dataset/grouped_v1_manifest.csv`.
Keep its reports with the resulting training run and compare assignments and
numeric labels before attributing hash changes solely to newlines. The existing
experiment's manifest must remain unchanged.

The training YAML is
`data/processed/dangerous_items_grouped_v1/data.yaml`. Its dataset path is
absolute for compatibility with Ultralytics on the machine that generated it.
Regenerating the dataset produces the correct path on another machine. The
committed `configs/dataset_grouped.yaml` is a portable config for repository
tools, which resolve its path relative to the config file.

Verification checks that all three splits exist and contain valid image/label
pairs, and rejects exact-hash or filename-family overlaps. Supplying the
manifest additionally verifies the recorded group assignments and the exact
image/label contents copied into the output. It exits with a nonzero status on
failure, including a missing or empty dataset.

## What this establishes, and what it cannot establish

Grouped v1 separates the known image families and exact duplicates. **It does
not prove source-video or scene independence.** Different frames from the same
recording can have unrelated filenames. Source/video IDs are unavailable in
the current metadata, and this script does not infer them from appearance.
Related scenes with different names may remain across splits.

The regrouping also reuses images that participated in earlier experiments.
The new model must start from the general pretrained `yolo11n.pt`, never the
old fine-tuned checkpoint. Freeze the new split before training, select the
checkpoint and confidence threshold using validation only, and use test only
for the final report. The regrouped test is a better controlled holdout for
this training run, but is not a new externally collected test set.

For a college project, the next independent check is a small, consented set of
new camera scenes, including normal objects that resemble weapons, differing
distances, lighting and occlusion. Keep all frames from one recording in the
same split and report false positives and missed objects. Use safe props or
appropriate existing footage. Do not call detector confidence a probability
that a campus threat is present.
