# Model Card

## Model

**Purpose:** Research prototype for video-level recognition of behavior associated with interpersonal violence.

**XD-Violence baseline:** Torchvision R3D-18 initialized with Kinetics-400 weights, followed by a binary classification head. Four RGB clips are sampled across each source video. Top-1 multiple-instance pooling produces one video score. The selected checkpoint was trained for six completed epochs on the available 996-video training subset and is tracked with Git LFS.

**UCF-Crime baseline:** The supplied notebook uses R3D-18 and classifies short Fighting or Normal clips. It is a separate experiment with a separate checkpoint.

**Input:** 16 RGB frames at a stride of two, resized to 112 × 112 and normalized with Kinetics-400 channel statistics. The XD training pipeline uses clip-level logits pooled at video level because its training labels are weak video labels.

**Output:** A probability-like model score in `[0, 1]`, a thresholded label, timestamps, and inference timing. The score is not a calibrated probability unless calibration is measured on validation data.

## Intended use

- Coursework and controlled research on video behavior recognition.
- Offline experiments on authorized datasets.
- A software prototype for testing a rolling frame interface and event-message integration.

## Out of scope

- Identifying a person, intent, guilt, or individual dangerousness.
- Replacing trained human responders or emergency procedures.
- Production campus surveillance or automated disciplinary action.
- Claims about reliable generalization to live campus footage.

## Known limitations

- Raw datasets and generated evaluation logs are local-only and excluded from source control. The selected checkpoint is tracked with Git LFS; it uses a partial training shard, so this is a subset experiment rather than a full benchmark training run.
- XD-Violence includes diverse media sources and scenes; it is not a campus-only dataset.
- This baseline uses RGB only, even though XD-Violence includes audio.
- Video-level labels are weak. Four sampled clips can miss short events, and top-1 pooling can amplify a single false positive clip.
- On the locally available XD-Violence test subset, recall was 0.617 at the validation-selected threshold, so 123 of 321 target videos were missed. This result is not adequate for operational use.
- The training run used 996 available files from the `1005-2004` archive, not the intended `0001-1004` shard or the full training collection. Four archive videos were unavailable locally, and training stopped during epoch 7 after decoder delays; the best checkpoint was saved at epoch 6.
- Car-accident-only and explosion-only videos are excluded from the human-violence task. Mixed-label videos that also contain a target category remain positive; event-specific labels within those videos are not separated by the current video-level evaluator.
- The UCF notebook uses a small Fighting-versus-Normal subset and its results are not interchangeable with XD-Violence results.
- Performance may change with viewpoint, distance, lighting, occlusion, compression, sports, playful interactions, crowds, camera movement, and different camera hardware.
- A single score threshold does not account for the base rate of incidents or the cost of missed and false alerts.

## Evaluation needed before any operational use

The subset run reported accuracy 0.773, precision 0.917, recall 0.617, F1 0.737, AUROC 0.890, AUPRC 0.905, 18 false alarms among 300 normal videos, and 123 missed detections among 321 target videos. Continue to report these measures, latency, and performance by camera or scene. Tune the operating threshold on validation videos only, freeze it, then evaluate once on the held-out test set. Include a separate consent-based campus-style evaluation set before making generalization claims.

All model outputs are review cues. They are not evidence that a real incident occurred.
