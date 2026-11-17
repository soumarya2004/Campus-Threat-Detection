# Recorded demos and external-video pilot

The recorded demos establish that the detector can process video and produce
annotated output. They are not accuracy benchmarks. The separate external pilot
exposes a real weakness: only **1 of 12 provisional knife annotations** matched
at the frozen confidence of 0.45 and matching IoU of 0.5.

This report summarizes existing saved artifacts, not a newly collected test.

## Demo evidence

The [saved runtime verification](experiments/runtime_verification.json) records
complete processing of the 145-frame rifle clip and 385-frame knife clip without
inference errors. The original clips' source provenance is unverified. They can
demonstrate the interface and failure inspection, but should not be described as
independent test footage. The [demo walkthrough](../docs/demo_walkthrough.md)
shows both a successful example and a missed knife.

## External pilot

Three public videos supplied 12 uniformly spaced frames each. Source selection
and provisional annotations were recorded before prediction. The checkpoint,
image size and confidence were unchanged from the grouped-model selection.

| Source | Frames | Target annotations | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Steak slicing | 12 | 12 knives | 1 | 3 | 11 |
| Scissors cutting card | 12 | 0 | 0 | 1 | 0 |
| Arrival scene | 12 | 0 | 0 | 0 | 0 |
| Total | 36 | 12 knives | 1 | 4 | 11 |

Source: [pilot evaluation](experiments/external_pilot_evaluation.json).
The [source ledger](experiments/external_pilot_sources.json) preserves origins,
authors, licenses and file hashes. The
[annotation manifest](experiments/external_pilot_annotations.json) records the
provisional labels, including six boxes with uncertain boundaries.

The annotations were created with assistant assistance and still require human
review. Frames from each recording are correlated: 36 frames are not 36
independent scenes. Only knife-positive footage was tested; the other four
positive classes were not measured. Overlap screening found no exact matches
with the training images, but cannot establish full source or pretraining
independence.

The pilot suggests difficulty with changed object appearance and ordinary
look-alikes. It cannot support a precise campus-wide accuracy estimate. Preserve
it as an exploratory result and follow the
[independent evaluation protocol](../docs/independent_footage_evaluation.md) for
a larger human-reviewed study. Do not combine these counts with the grouped
dataset score or retune the model on these examples and call them an untouched test.
