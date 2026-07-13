"""Safe two-axis servo test for MaixCam.

Run this file manually on the device. It never writes auto_start.txt and
always disables PWM in finally. Both axes are hard-limited to 45..135 degrees.
"""

from maix import pwm, pinmap
import time as pytime


MIN_ANGLE = 45
MAX_ANGLE = 135
CENTER_ANGLE = 90
FREQUENCY = 50
PAN_PWM_ID = 7
TILT_PWM_ID = 6


def clamp_angle(angle):
    angle = int(angle)
    return max(MIN_ANGLE, min(MAX_ANGLE, angle))


def duty_from_angle(angle):
    angle = clamp_angle(angle)
    return 2.5 + (float(angle) / 180.0) * 10.0


def move(pan, tilt, pan_angle, tilt_angle, pause=0.7):
    pan_angle = clamp_angle(pan_angle)
    tilt_angle = clamp_angle(tilt_angle)
    pan.duty(duty_from_angle(pan_angle))
    tilt.duty(duty_from_angle(tilt_angle))
    print("SERVO,PAN,%d,TILT,%d,LIMIT,45..135" % (pan_angle, tilt_angle))
    pytime.sleep(pause)


pan = None
tilt = None
try:
    # Current wiring: A19/PWM7 = horizontal, A18/PWM6 = pitch.
    pinmap.set_pin_function("A19", "PWM7")
    pinmap.set_pin_function("A18", "PWM6")
    pan = pwm.PWM(PAN_PWM_ID, freq=FREQUENCY, duty=duty_from_angle(CENTER_ANGLE), enable=True)
    tilt = pwm.PWM(TILT_PWM_ID, freq=FREQUENCY, duty=duty_from_angle(CENTER_ANGLE), enable=True)
    print("SERVO,INIT_OK")

    for angle in (90, 45, 90, 135, 90):
        move(pan, tilt, angle, CENTER_ANGLE)
    for angle in (90, 45, 90, 135, 90):
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

