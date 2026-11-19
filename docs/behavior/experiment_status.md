# Experiment Status

## XD-Violence subset run

The main P2 pipeline was trained and evaluated on the files present in the workspace. The available training archive was `1005-2004.zip`, not the `0001-1004` shard originally expected. The archive lists 1,000 videos; 996 were available locally. Four files were unavailable, and one attempted archive member failed its CRC check. The selected checkpoint is configured for Git LFS; detailed evaluation logs and generated manifests remain local and are excluded from Git. The exact split counts and source label are recorded in `data/behavior/splits/xd_manifest_summary.json` when the local manifests are generated.

| Split | Videos used | Normal | Violence |
|---|---:|---:|---:|
| Train | 720 | 439 | 281 |
| Validation | 127 | 77 | 50 |
| Test | 621 | 300 | 321 |

From the 996 local training files, 149 hazard-only videos were excluded. The test folder contained 800 files; 179 hazard-only videos were excluded from the binary violence-versus-normal evaluation. Videos with a target category plus a hazard remain positive. This keeps the target focused on interpersonal violence rather than accidents or explosions.

The model is a Kinetics-initialized R3D-18 with four sampled clips per video, top-1 multiple-instance pooling, and a validation-selected threshold. The best checkpoint was saved after epoch 6. Training had been configured for eight epochs, but the run was stopped during epoch 7 after decoder timeouts and malformed-frame warnings caused a long epoch; the epoch 6 checkpoint remains intact. Best validation F1 was 0.8085 at threshold 0.4901; validation AUROC was 0.9031.

Run settings: seed 42, batch size 2 videos, 16 frames per clip, two-frame sampling stride, and 112 × 112 input frames. Training ran on an NVIDIA GeForce RTX 5060 Laptop GPU (7.93 GB reported VRAM) with PyTorch 2.11.0+cu128. The best local checkpoint is `models/behavior/best_model.pt`.

### Held-out video-level results

| Metric | Result |
|---|---:|
| Accuracy | 0.7729 |
| Precision | 0.9167 |
| Recall | 0.6168 |
| F1 | 0.7374 |
| AUROC | 0.8900 |
| AUPRC | 0.9050 |
| False alarm rate on normal videos | 0.0600 (18/300) |
| Miss rate on violence videos | 0.3832 (123/321) |
| Mean inference time per video | 34.9 ms |

Confusion matrix, with rows as actual class (`normal`, `violence`) and columns as predicted class (`normal`, `violence`):

```text
[[282,  18],
 [123, 198]]
```

The threshold was selected from validation data and saved in the checkpoint. The test split was used once for the report and was not used for threshold selection. The test evaluation covered all 621 videos eligible under this project's label mapping, rather than all 800 downloaded files. The full per-video predictions and false-positive/false-negative examples are in the ignored local file `outputs/behavior/logs/xd_test_evaluation.json`; aggregate counts are in `outputs/behavior/logs/xd_error_analysis.json`.

Recall is the main weakness: about 38% of target videos were missed at this operating point. The run demonstrates a working XD-Violence training and evaluation pipeline, but it does not establish suitability for safety decisions or generalization to campus cameras. The local checkpoint was trained on a partial shard and only six completed epochs.

## Separate UCF-Crime notebook result

The supplied notebook's saved outputs report 50 Fighting and 50 Normal source videos, with 288 balanced clips. On 62 held-out clips, the saved output reports 0.887 accuracy. Majority voting over 23 held-out source videos reports 0.913 accuracy, with confusion matrix `[[11, 1], [1, 10]]`. These outputs were not reproduced during this run and describe only the notebook's Fighting-versus-Normal subset, not the full UCF-Crime benchmark. Clip-level and source-video-level scores use different evaluation units and should not be compared directly.

The original notebook and its outputs are preserved at `notebooks/behavior/UCF_Crime_R3D18.original.ipynb`; the portable copy has outputs cleared. UCF-Crime and XD-Violence remain separate because their labels, sampling, and evaluation protocols differ.

## Remaining work for a stronger result

1. Re-run using the intended `0001-1004` shard or a more complete training collection, and record the exact source files and counts.
2. Repair or replace unreadable training videos and add a bounded video-reader timeout so one damaged MP4 cannot stall an epoch.
3. Compare clip sampling and pooling choices, then select the operating threshold according to the relative cost of missed incidents and false alarms.
4. Preserve a fresh validation-selected threshold and evaluate the held-out test split once for any changed model.
5. Validate on an authorized campus-style video set before making campus generalization claims.

The PDF brief mentions RWF-2000 for Member 2. The main path here follows the user's direct request for XD-Violence, while keeping the supplied UCF-Crime notebook as a separate experiment.
