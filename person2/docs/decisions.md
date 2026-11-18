# Project Decisions

## Request and source brief

The project owner asked to combine the supplied UCF-Crime R3D-18 notebook with the supplied project ZIP, using XD-Violence for the ZIP-based P2 work. The attached brief assigns RWF-2000 to Member 2, but also describes the temporal recognition scope and requires video-level split isolation, fixed clips, evaluation, limitations, and a shared inference interface.

This repository follows the direct request: it preserves the UCF-Crime notebook and configures the main P2 path for XD-Violence. It does not silently switch the requested dataset to RWF-2000.

## Dataset separation

The UCF-Crime and XD-Violence experiments remain separate. UCF notebook labels use Fighting versus Normal; XD-Violence labels include weak video-level labels and official temporal test annotations. Combining their clips into one dataset would mix annotation protocols and produce unclear supervision.

## XD-Violence scope

This project's target is interpersonal violence: Fighting, Shooting, Riot, and Abuse are positive; Normal is negative. Car Accident and Explosion only examples are excluded from this task rather than relabeled as normal. This mapping follows the campus behavior goal and avoids conflating a non-human hazard with benign activity.

## Dataset files and training provenance

The local workspace now contains real XD-Violence media. The available training archive was `1005-2004.zip`, rather than the `0001-1004` shard first expected. Of its 1,000 listed videos, 996 were available locally; four were missing, including one file that failed an archive CRC check when extraction was attempted. The run therefore uses a subset and records its source in the manifest summary and model checkpoint. The 800 local test files are the official test folder; 179 hazard-only videos are excluded from the binary violence-versus-normal evaluation.

Generated `syn_*` clips and the supplied synthetic checkpoint remain software fixtures only. Their earlier 0.50 accuracy and 0.00 recall on 12 generated samples are not XD-Violence or UCF-Crime results.

The supplied UCF-Crime notebook has saved output for a 100-source-video Fighting/Normal subset. The output reports clip-level accuracy of 0.887 on 62 held-out clips and video-level majority-vote accuracy of 0.913 on 23 held-out source videos. These values are recorded as notebook output only; they have not been independently reproduced during this handoff.

## Scope boundary

P2 delivers temporal behavior recognition and its integration interface. It does not deliver the full five-member system, including weapon detection, person tracking, live DroidCam capture, event assessment, or the Streamlit dashboard.
