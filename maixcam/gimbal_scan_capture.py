"""Capture a safe +/-45 degree horizontal scan, then disable both servos."""

import os
import time as pytime

from maix import camera, pinmap, pwm


PAN_PWM_ID = 6
TILT_PWM_ID = 7
CENTER = 90
MIN_ANGLE = 45
MAX_ANGLE = 135
OUT_DIR = "/root/snail_egg/scan"


def clamp(value):
    return max(MIN_ANGLE, min(MAX_ANGLE, int(value)))


def duty(angle):
    return 2.5 + clamp(angle) / 180.0 * 10.0


os.makedirs(OUT_DIR, exist_ok=True)
pinmap.set_pin_function("A18", "PWM6")
pinmap.set_pin_function("A19", "PWM7")
pan = tilt = cam = None
try:
    pan = pwm.PWM(PAN_PWM_ID, freq=50, duty=duty(CENTER), enable=True)
    tilt = pwm.PWM(TILT_PWM_ID, freq=50, duty=duty(CENTER), enable=True)
    cam = camera.Camera(640, 480, buff_num=1)
    pytime.sleep(1.0)
    for offset in (-45, -30, -15, 0, 15, 30, 45):
        pan.duty(duty(CENTER + offset))
        tilt.duty(duty(CENTER))
        pytime.sleep(1.0)
        frame = cam.read()
        path = "%s/pan_%+03d.jpg" % (OUT_DIR, offset)
        frame.save(path)
        print("SCAN,PAN_OFFSET,%d,FILE,%s" % (offset, path))
finally:
    try:
        if pan is not None and tilt is not None:
            pan.duty(duty(CENTER))
            tilt.duty(duty(CENTER))
            pytime.sleep(0.8)
            print("SCAN,RETURN_CENTER")
    except Exception:
        pass
    for channel in (pan, tilt):
        try:
            if channel is not None:
                channel.disable()
        except Exception:
            pass
    print("SCAN,PWM_DISABLED")
