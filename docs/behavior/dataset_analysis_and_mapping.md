# Dataset Protocol and Label Mapping

## Scope

P2 estimates whether a video window contains behavior associated with interpersonal violence. The main pipeline uses XD-Violence; the supplied UCF-Crime notebook is kept as a separate Fighting-versus-Normal experiment. Their videos and labels are not combined into a single train/test split.

## XD-Violence

XD-Violence contains long, untrimmed audio-visual videos. Its original protocol provides weak video labels for training and temporal frame intervals for the test set. The training code in this repository uses only RGB frames. It does not reproduce the source paper's audio and feature-fusion architecture.

The official filename tags use these codes:

| Code | Official category | P2 target | Rationale |
|---|---|---:|---|
| `B1` | Fighting | 1 | Direct physical conflict |
| `B2` | Shooting | 1 | Armed attack |
| `B4` | Riot | 1 | Collective violent disturbance |
| `B5` | Abuse | 1 | Physical abuse |
| `A` | Normal | 0 | Negative baseline |
| `B6` | Car accident | Exclude | Safety event, but not interpersonal violence |
| `G` | Explosion | Exclude | Safety event, but not interpersonal violence |

For a multi-label filename, the video is positive if it contains one or more target codes. Videos containing only `B6` and/or `G` are excluded; they are not relabeled as normal. The filename tags and code meanings come from the [official project page](https://roc-ng.github.io/XD-Violence/) and its [annotation readme](https://roc-ng.github.io/XD-Violence/images/ReadMe.md).

Training uses a multiple instance learning (MIL) baseline. Each source video is represented by four clips sampled across its duration. The model scores each clip and top-1 pooling creates one video score for the weak video label. The video, not the clip, is the training and evaluation unit. This reduces the incorrect assumption that every clip in a positive untrimmed video contains violence.

The script `scripts/prepare_xd_violence.py`:

1. Scans the official Train and Test video folders and reads the label codes from filenames.
2. Reads FPS and frame count without decoding entire videos into memory.
3. Splits official training videos into train and validation groups by source video, stratified by binary label.
4. Keeps the official test folder separate.
5. Parses the test interval file into start/end frame pairs and matches them to local video filenames.
6. Writes project-relative manifests under `data/behavior/splits/`.

The test annotation file supports temporal inspection. The current `evaluate.py` reports held-out **video-level** metrics; do not describe its output as frame-level event detection performance.

## UCF-Crime notebook track

The supplied notebook uses Fighting videos as positive and `Normal_Videos_for_Event_Recognition` videos as negative. Its checked-in output reports 50 files in each class and 288 balanced extracted clips: 89/89 train, 24/24 validation, and 31/31 test clips. The split is assigned by source video in the manifest before clip extraction.

This experiment is narrow: it is not a model trained on all UCF-Crime categories and it should not be compared directly with XD-Violence's video-level weak-label protocol. The original notebook is retained as a reference; the portable copy uses project-relative paths and avoids deleting previous extracted outputs.

## Data access and handling

The raw datasets are not included in this repository. Obtain them from their official academic access pages, review their licenses and access terms, and store them under `data/behavior/raw/`. Raw media, checkpoints, generated manifests, and run outputs are ignored by Git. Do not push raw benchmark videos or source annotations to a public repository.

## Data quality checks before training

- Confirm the train and test directories contain the intended official videos.
- Check the generated `xd_manifest_summary.json` for class counts and videos excluded as hazard-only.
- Resolve any unmatched annotation rows before reporting temporal evaluation.
- Confirm each video appears in exactly one split.
- Keep the test manifest untouched while selecting hyperparameters or thresholds.
- Record dataset version, preprocessing parameters, checkpoint hash, and device with every reported experiment.
