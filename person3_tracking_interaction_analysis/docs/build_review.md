# Completed build review

| Review | Starting readiness | Proposed plan | Completed build |
| --- | ---: | ---: | ---: |
| Engineering | 40/100 | 91/100 | **92/100** |
| Academic | 43/100 | 88/100 | **91/100** |

These are internal readiness scores for the available five-credit project scope. The department's marking rubric and the other team modules were unavailable.

| Criterion | Available | Engineering | Academic |
| --- | ---: | ---: | ---: |
| Academic problem and scope | 10 | 10 | 10 |
| Reproducibility and VS Code structure | 15 | 13 | 14 |
| Detection, tracking and gate integration | 20 | 19 | 19 |
| Movement and interaction correctness | 20 | 19 | 19 |
| Evaluation and interpretation | 20 | 17 | 16 |
| Tests and error handling | 7 | 7 | 7 |
| Report quality and attribution | 5 | 4 | 4 |
| Demo and viva | 3 | 3 | 2 |
| **Total** | **100** | **92** | **91** |

## Verified

- The Windows CLI ran in a new virtual environment with access to the already installed, version-matched host packages. The folder and output path both contained spaces. The dependency consistency check passed.
- All 91 automated tests passed. They include 67 retained detector-wrapper tests and 24 new tests for gate timing, track identities, analytics, I/O errors, and video and CSV exports.
- The synthetic replay decoded 54 input and 54 output frames at 20 FPS. It produced 34 active frames in two exact windows, even with the scripted inference error. The negative case produced a decodable video and header-only analytics CSVs. The notebook's three code cells executed in order under IPython, and the plot was inspected.
- Both corrected model modes processed the same 250-frame input on CPU. Gating reduced person-model calls from 250 to 208, increased processing time from 43.01 s to 69.22 s, and reduced track coverage from 91/95 to 75/95 among manually marked person boxes. On the gate-active subset both modes covered 75/75. See [the evaluation](../evaluation/results.md) for precise denominators and example failures.

## Remaining evidence

An isolated installation from the public package index and a launch using the VS Code interface still need verification. The terminal command corresponding to the launch configuration worked. Before the viva, reproduce the setup on the presentation laptop and rehearse the supplied trace. The real evaluation uses one previously seen scene, 30 sampled frames and one timed run per mode. Additional permission-cleared clips would improve evaluation coverage and make runtime variation measurable. The detector predictions in this clip have no object ground truth.
