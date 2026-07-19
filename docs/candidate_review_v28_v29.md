# v28/v29 candidate review

Date: 2026-07-19

This document records local candidate work only. Neither model was deployed or
pushed. GitHub `main` remains the real-device-verified v6 baseline.

## Comparable offline results

All stress runs used 1000 generated frames sourced only from the untouched test
split at 320x224, confidence 0.15, and the same safe filter. The external set
contains 1378 negative-only images.

| Model | Stress recall | Stress precision | Motion recall | Cool-light recall | Camera recall | Camera negative-frame FP | External image FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v6 | 83.67% | 91.90% | 78.82% | 65.66% | 77.14% | 8.57% | 0.80% |
| v26 | 86.73% | 91.10% | 83.33% | 81.82% | 94.29% | 20.00% | 0.65% |
| v28 | 87.68% | 88.09% | 80.21% | 83.84% | 94.29% | 11.43% | 1.81% |
| v29 | 87.13% | 89.81% | 80.21% | 83.84% | 91.43% | 11.43% | 1.52% |

v28 improved cool-light recall but traded too much precision and external
negative performance. v29 added only previously unseen, training-only field
negatives and improved on v28's precision, but it still regressed against v26
on camera recall, motion recall, and external false positives. Both candidates
are rejected.

## Tracking candidate retained for device validation

The detector weight remains v26. The local program candidate changes the
temporal logic instead:

- apply dual-buffer and gimbal-coordinate alignment before association;
- use measured loop time in Kalman prediction;
- compensate known camera motion from actual pan/tilt output;
- allow weak detections to update only the manually selected trajectory;
- block weak detections from track birth, target switching, and aiming-light
  safety presence unless associated with that selected track;
- increase measurement noise for weak boxes so they cannot pull the trajectory
  abruptly.

`scripts/test_multitarget_motion_lock.py` covers a three-target sequence with
gimbal motion, a weak selected detection, and a visible neighbor. The selected
ID remains stable and the neighbor cannot take over after loss. All host-side
regression tests pass. This is not a substitute for real-device video and PROF
telemetry; deployment remains blocked until the camera is intentionally powered
and supervised.
