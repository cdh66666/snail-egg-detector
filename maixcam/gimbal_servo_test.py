"""Safe two-axis servo test for MaixCam.

Run this file manually on the device. It never writes auto_start.txt and
always disables PWM in finally. Pan is limited to 60..120 degrees and tilt
is limited to 80..120 degrees.
"""

from maix import pwm, pinmap
import time as pytime


PAN_MIN_ANGLE = 60
PAN_MAX_ANGLE = 120
TILT_MIN_ANGLE = 80
TILT_MAX_ANGLE = 120
CENTER_ANGLE = 90
FREQUENCY = 50
PAN_PWM_ID = 6
TILT_PWM_ID = 7


def clamp_angle(angle, min_angle, max_angle):
    angle = int(angle)
    return max(min_angle, min(max_angle, angle))


def duty_from_angle(angle):
    angle = max(0, min(180, int(angle)))
    return 2.5 + (float(angle) / 180.0) * 10.0


def move(pan, tilt, pan_angle, tilt_angle, pause=0.7):
    pan_angle = clamp_angle(pan_angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)
    tilt_angle = clamp_angle(tilt_angle, TILT_MIN_ANGLE, TILT_MAX_ANGLE)
    pan.duty(duty_from_angle(pan_angle))
    tilt.duty(duty_from_angle(tilt_angle))
    print("SERVO,PAN,%d,TILT,%d,LIMIT,PAN_60..120,TILT_80..120" % (pan_angle, tilt_angle))
    pytime.sleep(pause)


pan = None
tilt = None
try:
    # Current wiring: A18/PWM6 = horizontal, A19/PWM7 = pitch.
    pinmap.set_pin_function("A19", "PWM7")
    pinmap.set_pin_function("A18", "PWM6")
    pan = pwm.PWM(PAN_PWM_ID, freq=FREQUENCY, duty=duty_from_angle(CENTER_ANGLE), enable=True)
    tilt = pwm.PWM(TILT_PWM_ID, freq=FREQUENCY, duty=duty_from_angle(CENTER_ANGLE), enable=True)
    print("SERVO,INIT_OK")

    for angle in (90, 60, 90, 120, 90):
        move(pan, tilt, angle, CENTER_ANGLE)
    for angle in (90, 80, 90, 120, 90):
        move(pan, tilt, CENTER_ANGLE, angle)

    print("SERVO,PASS")
finally:
    for channel in (pan, tilt):
        try:
            if channel is not None:
                channel.disable()
        except Exception as exc:
            print("SERVO,DISABLE_ERROR,%s" % exc)
    print("SERVO,PWM_DISABLED")
