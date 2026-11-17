# Explaining Person 1's work in a viva

Use the actual experiment files when answering. These notes describe the
implementation; they are not a script to memorize.

## What did this module contribute?

A five-class object detector, dataset integrity checks, a reproducible training
and evaluation workflow, and the frame interface used by the rest of the team.
YOLO and its pretrained weights are existing work. Our project contribution is
the preparation, experiments, measured limitations, and integration around it.

## Why YOLO11n?

It is a compact pretrained detector that runs on the available laptop. The
choice balances inference cost and a manageable training experiment. Do not
claim it is the best architecture unless a controlled comparison establishes
that. A newer model name alone is not an improvement.

## Why was the original split changed?

Exact SHA-256 hashing catches files with identical bytes. It does not catch a
blurred or rotated copy. The audit found such copies on opposite sides of the
training/test boundary. The new builder joins filename-derived augmentation
families and exact duplicates before assigning groups to splits.

One representative per group is used. YOLO applies training augmentation only
after the split. This reduces known image-level leakage, but different frames
from an unknown source video can still be related. Source provenance is a
remaining limitation, not something the filename rule can prove away.

## Why start training again from pretrained weights?

The earlier fine-tuned model already saw images related to the new held-out
set. Reusing it would carry that information into the experiment. The new run
starts from the original pretrained `yolo11n.pt`, not the old fine-tune.

## What does the confidence threshold mean?

It decides which predicted boxes to retain. It is not the probability that a
person is dangerous. Lowering it usually retains more objects and more false
positives. The threshold is selected on validation data using the documented
criterion, then held fixed for test evaluation.

For this run, the validation grid was 0.10 to 0.80 in steps of 0.05. Confidence
0.45 gave the highest macro F1, 0.8073; 0.50 was close at 0.8071. This is a
recorded operating choice, not proof that 0.45 is universally optimal.

## What are IoU and NMS, and why are their thresholds different?

Intersection over union (IoU) measures the overlap of two boxes: intersection
area divided by union area. If the overlap is 50 square pixels and the union
is 100, IoU is 0.5. It measures geometry, not prediction confidence.

Non-maximum suppression (NMS) removes redundant, overlapping predictions. The
configured class-aware NMS uses IoU 0.7 and keeps the higher-confidence box when
same-class predictions overlap beyond that threshold. This happens during
inference, before evaluation.

The fixed-threshold evaluator separately uses IoU 0.5 to match a prediction to
a ground-truth box of the same class. NMS compares predictions with predictions;
evaluation compares predictions with annotations. Their thresholds serve
different purposes and should not be interchanged.

## How does one-to-one matching count mistakes?

Predictions are considered in descending confidence order. Each can match at
most one ground-truth object of the same class, and an annotation can be used
only once. A second prediction around an already matched object is a false
positive, not another true positive. A knife predicted as a machete counts as
a false positive for machete and a missed knife.

The reported fixed operating point uses this class-aware matching. The
diagnostic confusion matrix separately permits geometric matches across
classes so wrong-class predictions appear off the diagonal. Its diagonal is
not a substitute for the fixed-threshold TP total.

## Why report more than mAP@50?

mAP@50 evaluates detection over confidence thresholds with an IoU match of 0.5.
mAP@50-95 is stricter about localization. Fixed-threshold precision and recall
show the operating point actually used by the wrapper. Per-class results show
whether an average conceals a weak class. Ordinary background images help
measure false positives, although image results are not video false-alert rates.

On the 700-image grouped test, mAP@50 is 85.79% and mAP@50-95 is 65.89%.
At confidence 0.45, there are 544 TP, 74 FP and 154 FN. Micro precision is
544 / (544 + 74) = 88.03%; micro recall is 544 / (544 + 154) = 77.94%.
These are object-detection metrics, not the percentage of safe or unsafe people
classified correctly. Knife recall is only 68.24%, which the overall average
can conceal.

Micro metrics combine the class counts first. Macro F1 averages the per-class
F1 scores for classes represented in validation. All five classes are present
here; macro F1 was the declared threshold-selection criterion.

## Why did 640 pixels win over 960?

Both inference sizes were evaluated on validation using the same checkpoint.
Validation mAP@50-95 was 0.6299 at 640 and 0.5355 at 960. The predeclared rule
required an improvement of at least 0.02 within the timing budget to select
960, so 640 was retained. A larger input did not improve this particular run;
this is not a general claim that larger images never help.

## Does zero detection on the background images mean zero false alarms?

No. None of the 74 background test images produced retained detections, but
there were 74 false-positive boxes on other test images. Those backgrounds
are a small dataset sample, not hours of normal campus footage. A video
false-alert rate also needs an agreed event definition and labeled recordings.

## Is the demonstrated video performance real time?

The complete recorded clips processed at about 81.48 FPS for the rifle clip
and 30.67 FPS for the 1920 x 1080 knife clip on the tested laptop. Those figures
include the processing loop, drawing and writing, but exclude startup. The
11.82 ms mean reported for the image test measures warmed model stages only.
Neither measurement includes the complete camera-to-dashboard path. Describe
the measured setup, not a guaranteed live-system frame rate.

## What do the tests prove?

Unit tests cover frame validation, output schema, malformed boxes, errors,
model lifecycle, split integrity, matching logic, and command-line failure
paths. A smoke test executes a real checkpoint. Neither test count nor FPS is
a measure of detection accuracy; that comes from labeled evaluation.

## Why two detector classes?

`WeaponDetector` translates Ultralytics output into plain Python detections.
`YOLOObjectDetector` is the team's public interface: it checks input/output,
preserves frame IDs and timestamps, and distinguishes a failed prediction from
an empty successful one. Tracking, actions, and final threat assessment belong
to the other modules.

For integration, the caller supplies BGR `uint8` frames and preserves frame IDs
and timestamp meaning. Boxes are `[x1, y1, x2, y2]` in original-image pixels.
An empty result with `model_status="ok"` means no item passed the threshold;
`model_status="inference_error"` means inference failed. Neither message
establishes that a scene is safe. The detector does not assign track IDs or
decide whether someone is dangerous.

## Can the test mistakes now be used to tune the model?

They can motivate a future experiment, but then this test is no longer an
untouched source of evidence for the new choices. Preserve the current report,
use training/validation data to choose changes, and obtain fresh evaluation
footage for new generalization claims. Do not adjust thresholds or change
test labels just to improve the published score.

## What would you improve next?

Collect or obtain permitted, independently sourced footage. Include ordinary
objects that resemble the target classes, small/distant items, occlusion, and
different lighting. Annotate it before evaluating. Report missed objects and
false positives, not just successful screenshots. This is the next evidence
needed before making campus-performance claims.

The standalone submission now includes a pretrained baseline for knife and
baseball bat, a measured model-size/inference-batch comparison, a visible
120-frame camera rehearsal and a two-stream JSON consumer. These are P1's own
experiments and do not require another team member's implementation. The
common-class validation F1 improved from 39.84% to 79.33% at the shared 0.25
threshold; this does not change the frozen five-class test result at 0.45.

The [external-video pilot](../reports/person1_demo_and_external_pilot.md) matched
only 1 of 12 provisional knife annotations across three video sources. Its
labels need human review and its correlated frames do not constitute a campus
benchmark. The result is a reason to improve data coverage, not a result to hide.
Use [the five-minute walkthrough](demo_walkthrough.md) to rehearse a balanced
presentation with a successful detection, a missed object, and the team
interface.

## How are people detected if the dataset has no person labels?

`YOLOPersonDetector` uses a separate general pretrained YOLO11n checkpoint and
requests only COCO person class 0. The five-class model keeps its own IDs,
including 0 for machete. The integration payload keeps `people` and `items`
separate. Person detections can feed the tracking teammate; they do not assign
identities, track IDs, or ownership of an item. The person stream has a tested
runtime interface but still needs labeled campus evaluation and tracker tuning.
