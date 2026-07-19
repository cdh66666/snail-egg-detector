# Verified v6 release manifest

This document identifies the MaixCAM v6 deployment baseline restored to `main` on 2026-07-19.

## Provenance

- Detector weights: `runs/detect/runs_yolo/snail_eggs_yolo11n_robust_v6_mined/weights/best.pt`
- Device model conversion timestamp: 2026-07-19 11:09:39 +08:00
- Device program source snapshot: `dist/maixcam_snail_eggs_yolo/device/main.py`, captured before the unverified v26 tracking changes
- Release program change: only the model path was changed from the later v26 conversion name back to the verified v6 artifact name
- Camera state during repository restoration: powered off; no new device validation is claimed

## SHA256

```text
DCBB36F7AADA642AD1C8BE089D49ACA053637833060E70D83533661941490C21  maixcam/main.py
B22411B77841B6F1B59924F5CA46EBAFAE96E2CA7798DEA78E7F1A56F7A44E66  snail_eggs_yolo11n_320x224_v6.cvimodel
DBDECCD9B6DA787CEDDC5FE266D1C0BE8FC75613D99C581938AD277CD4DE90E2  snail_eggs_yolo11n_320x224_v6.mud
BD2106462B80837815284F261926F9475DF004CC256E6FFF44E13E9C3BF9D336  release/maixcam_copy_to_device.zip
```

The v26 and later work remains experimental until it passes real-device video, tracking, false-positive, latency, and safety acceptance. It must not replace this release merely because an offline metric improves.
