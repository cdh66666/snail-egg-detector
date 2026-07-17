# MaixCAM Web Control and Color Backup

## Web control

`maixcam/main.py` starts a small HTTP server on `0.0.0.0:8000`. The handler only records commands; camera, GPIO, and PWM are still changed by the main vision loop.

Connect a phone and MaixCAM to the same LAN or hotspot, then open:

```text
http://<MAIXCAM_IP>:8000/?token=maixcam
```

The page provides Auto Track, Hold, Center, bounded manual pan/tilt, snapshot refresh, and low-power red aiming-light requests. A browser cannot be forcibly opened by MaixCAM, so the URL is printed in the device log and on the display.

The `aim_on` request never bypasses the fail-closed safety gate. The red auxiliary light is allowed only when at least one fresh YOLO egg candidate is present in the whole frame; when the whole frame has no valid egg candidate it is turned off. The high-power laser remains manual and is not controlled by this project.

## Pure color comparison mode

The web page's `Color Candidate ON` button creates `/root/snail_egg/enable_color_only`. The main loop then uses a broad pink-color blob detector and tracks every surviving pink candidate. This mode is intentionally a comparison baseline: it will also respond to pink objects, so it is not the default egg detector.

Turn it off with `Color Candidate OFF`, or remove the flag over SSH:

```bash
rm -f /root/snail_egg/enable_color_only
```

The normal YOLO path remains active when the flag is absent.

## Single-instance restart

Do not launch a second copy while an auto-start application is running. Stop the existing application, wait for the camera driver to release, and then start one copy. A second process can cause `Address in use` on port 8000 and `No buffer space available` from the camera driver.

## Verification

```bash
python -m py_compile maixcam/main.py maixcam/web_control.py
python scripts/test_aim_relay_safety.py
python scripts/test_closed_loop_aim.py
python scripts/test_target_lock_persistence.py
python scripts/test_gimbal_control_dynamics.py
```
