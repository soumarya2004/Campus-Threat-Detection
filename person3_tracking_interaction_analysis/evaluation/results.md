# Evaluation results

On the 250-frame pedestrian clip, gating skipped **42 person-model calls (16.8%)** and added 250 candidate-object model calls. The full processing run took **69.22 s gated and 43.01 s continuous**. Person-track coverage over 95 manually labeled observations fell from **91/95 (95.8%) to 75/95 (78.9%)**. Both modes matched all 75 labeled observations sampled on frames where the gate was active.

## Real inference comparison

One sequential run per mode on a 12th Gen Intel Core i7-1255U CPU, Windows 11 build 26200, with `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`. Both runs used Python 3.13.1, OpenCV 5.0.0, Ultralytics 8.4.202, Supervision 0.31.4 and Trackers 2.7.0. Input frame rate was 25 FPS at 1280 × 720. Measured processing includes frame reads, both active model stages, ByteTrack, writing the video, and CSV generation. Startup and output-video readback are outside processing time.

| Measurement | Continuous | Detector-gated |
| --- | ---: | ---: |
| Frames decoded and written | 250 | 250 |
| Person stage active | 250/250 | 208/250 (83.2%) |
| Person model calls | 250 | 208 |
| Candidate-object model calls | 0 | 250 |
| Frames with candidate-object predictions | Stage unused | 24 frames, 28 boxes |
| Candidate-detector inference errors | Stage unused | 0 |
| Person stage processing | 36.50 s | 26.37 s |
| Candidate-object stage processing | Stage unused | 36.58 s |
| Model startup | 5.89 s | 7.77 s |
| Full processing interval | 43.01 s | 69.22 s |
| Measured processing throughput | 5.81 frames/s | 3.61 frames/s |

Gated event windows were **2–77, 80–145, 151–189, 199–219 and 244–249**, inclusive. The fifth ended at the clip boundary. The continuous run groups its output under one event ID for the full clip. The gated run restarts the tracker per event. Its 51 scoped IDs and the continuous run's 23 IDs describe tracker output and may count a person more than once.

| Check on the manual sample | Continuous | Detector-gated |
| --- | ---: | ---: |
| Matched observations, all sampled frames | 91/95 | 75/95 |
| Matched observations, 24 jointly active sampled frames | 75/75 | 75/75 |
| Matched observations labeled as partially occluded | 15/16 | 10/16 |
| Scoped ID changes, sampled transitions matched by both modes | 0/56 | 0/56 |
| Matched → unmatched, among 76 sampled transitions | 1 | 4 |
| Unmatched → matched, among 76 sampled transitions | 3 | 6 |
| Available proximity results, 101 eligible sampled pairs | 89/101 | 73/101 |
| Proximity agreement on available predictions | 89/89 | 73/73 |
| Agreement on available expected-close predictions | 4/4 | 3/3 |
| Agreement on available expected-far predictions | 85/85 | 70/70 |

All 20 gated misses were on inactive frames. The tracker IDs stayed stable over the 56 transitions in which both endpoints matched in both modes. This check samples five frames per window and can miss brief changes between labeled frames. The proximity sample includes 5 expected-close and 96 expected-far observations after dropping 4 near the threshold. Agreement excludes absent predictions, so it should be read alongside pair availability.

### Traceable examples

- **Frame 110 (4.40 s).** The candidate box straddles a pedestrian's upper body and shoulder bag. Visual inspection leaves the object prediction ambiguous. These boxes represent detections that can activate the gate. The clip has no object ground truth.
- **Frame 220 (8.80 s).** All four labeled people on the sample frame matched in continuous mode. The gated run was inactive following expiry at frame 219 and yielded no tracks on this frame. This accounts for four of the 20 gated misses.
- **Frame 225 (9.00 s).** In continuous mode, partially occluded subject `P07` had a best overlap of 0.5045, but that detection was assigned to a different labeled subject with a higher overlap. This counts as a miss under the one-to-one rule.
- **Frame 0 (0.00 s).** Continuous mode processed the frame and exported zero assigned tracks. The label sample had three people visible. A person-stage call is distinct from a usable track.

The number of person calls fell. On this hardware and clip, candidate-object inference cost more time than the avoided person work, and missed frames reduced coverage. A single seen clip and one timed run per mode give a limited engineering comparison. The sample cannot establish cross-scene tracking performance, object-detector precision or recall, or any claim about campus incidents.

## Deterministic replay checks

| Case | Known input | Verified output |
| --- | --- | --- |
| Two trigger windows | 54 frames, 20 FPS, scripted triggers on frames 4 and 30, one simulated detector error on frame 9 | 34 active frames, events on frames 4–20 and 30–46 inclusive. The simulated error was recorded and preserved the existing grace deadline. Raw IDs 0 and 1 appeared under both event IDs and kept separate histories. |
| Known approach and separation | Two supplied boxes change center separation by more than 5 px per frame while moving, then reverse. At frame 18 their centers are (167, 130) and (313, 148) px, about 147.11 px apart. | Proximity spans 18–20 and 31–35. Inclusive durations at 20 FPS are 3/20 = 0.15 s and 5/20 = 0.25 s. Pair records identify approach on event 1 and separation on event 2 after each initial observation. |
| No trigger | The same 54-frame video with positions supplied and all triggers disabled | Playable 54-frame output and frame log. No person-stage calls or event, movement, pair or proximity rows. Analytics CSVs preserve their headers. |

Replay uses known boxes and triggers. The real clip run uses both model checkpoints and supplies the performance and coverage comparison above. See the [protocol](protocol.md), [manual sample](pedestrian_sample.csv), [per-observation results](recorded_results.json) and [replay outputs](../outputs/example/).
