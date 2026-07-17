import atexit
import os
import time as pytime

from maix import camera, display, image, nn, app, time, pwm, pinmap, gpio


# Copy this file to MaixVision as main.py after converting the ONNX model to MUD.
# Put the MUD and model file under /root/models/ on the MaixCam.

MODEL = "/root/models/snail_eggs_yolov8n_640x480.mud"
CONF_TH = 0.20
IOU_TH = 0.35
MAX_TARGETS = 32
MIN_MODEL_CONF = 0.20
STRONG_MODEL_CONF = 0.38
LOW_CONF_MIN_PINK_RATIO = 0.035
LOW_CONF_MIN_AREA = 18

# Red aiming-light relay only. The destructive laser remains manual.
AIM_RELAY_PIN = "A15"
AIM_RELAY_GPIO = "GPIOA15"
AIM_RELAY_DISABLE_FLAG = "/root/snail_egg/disable_aim_relay"
AIM_RELAY_NO_TARGET_TEST_FLAG = "/root/snail_egg/test_no_target_safety"
AIM_RELAY_STARTUP_TEST_DONE_FLAG = "/root/snail_egg/aim_relay_startup_test_done"
AIM_RELAY_ON_STABLE_FRAMES = 3
# Fail closed on the first frame without a fresh YOLO-confirmed egg mass.
# Predicted/Kalman-only boxes never keep the red aiming light enabled.
AIM_RELAY_OFF_MISSING_FRAMES = 1
AIM_RELAY_PULSE_S = 0.20
AIM_RELAY_STARTUP_TEST_ON_S = 1.0
AIM_RELAY_STARTUP_TEST_CYCLES = 1

# Closed-loop aiming uses the visible low-power red dot as measured feedback.
# It is runtime opt-in so a missing/misconfigured dot can never move the gimbal.
CLOSED_LOOP_AIM_FLAG = "/root/snail_egg/enable_closed_loop_aim"
AIM_SCAN_FLAG = "/root/snail_egg/enable_aim_scan"
AIM_DOT_LAB_THRESHOLDS = [[20, 100, 35, 127, -20, 127]]
AIM_DOT_MIN_PIXELS = 1
AIM_DOT_MAX_PIXELS = 420
AIM_DOT_MIN_SIDE = 1
AIM_DOT_MAX_SIDE = 32
AIM_DOT_MIN_RED = 120
# The GC4653 capture measured the real spot at RGB roughly (176, 98, 122):
# red dominance is only about 54 after auto white balance. Spatial and size
# gates below provide the stronger discrimination from pink egg pixels.
AIM_DOT_MIN_DOMINANCE = 45
AIM_DOT_MIN_DOMINANCE_RATIO = 0.30
AIM_DOT_MIN_LOCAL_CONTRAST = 12
AIM_DOT_RGB_STRIDE = 7
AIM_DOT_RGB_REFRESH_FRAMES = 3
AIM_DOT_EXPECTED_RADIUS_PX = 70
AIM_DOT_MAX_JUMP_PX = 65
AIM_DOT_FILTER_ALPHA = 0.55
AIM_DOT_STALE_FRAMES = 8
AIM_DOT_LOG_EVERY_N_FRAMES = 10
AIM_DOT_CONTROL_TOLERANCE_PX = 7
AIM_SCAN_SETTLE_FRAMES = 4
AIM_SCAN_DWELL_S = 0.30
AIM_SCAN_MARGIN_RATIO = 0.28

# Basic gimbal control. On the current wiring A18/PWM6 is pan and
# A19/PWM7 is tilt. Every command is clamped to the axis-specific limits
# before PWM output.
# Keep the gimbal completely still during detector startup. Tracking code can
# enable this explicitly after the camera has reached a stable state.
ENABLE_GIMBAL = False
# The controller is deliberately a second opt-in switch.  It cannot move a
# servo unless both this flag and ENABLE_GIMBAL are enabled.
ENABLE_GIMBAL_TRACKING = False
GIMBAL_ENABLE_FLAG = "/root/snail_egg/enable_gimbal_tracking"
GIMBAL_DRY_RUN_FLAG = "/root/snail_egg/gimbal_dry_run"
GIMBAL_DISABLE_FLAG = "/root/snail_egg/disable_gimbal"
# A monitor photographed by the camera is a different visual domain from real
# outdoor scenes. This opt-in flag lowers only the model score floor for the
# desktop tracking rig; removing it restores the production thresholds above.
SCREEN_TEST_FLAG = "/root/snail_egg/screen_tracking_test"
SCREEN_TEST_CONF_TH = 0.08
GIMBAL_FREQ = 50
GIMBAL_CENTER_DEG = 90
GIMBAL_PAN_MIN_DEG = 60
GIMBAL_PAN_MAX_DEG = 120
GIMBAL_TILT_MIN_DEG = 80
GIMBAL_TILT_MAX_DEG = 120
GIMBAL_CENTER_SETTLE_S = 0.35
GIMBAL_PAN_PWM_ID = 6
GIMBAL_TILT_PWM_ID = 7
# Control loop is slower than the detector on purpose. The detector/display
# can stay near 19 FPS while the hobby servos receive smooth 10 Hz updates.
GIMBAL_CONTROL_HZ = 10.0
GIMBAL_DEADZONE_X = 0.018
GIMBAL_DEADZONE_Y = 0.022
GIMBAL_MAX_STEP_DEG = 0.5
GIMBAL_MAX_RATE_DEG_S = 5.0
GIMBAL_MAX_ACCEL_DEG_S2 = 10.0
GIMBAL_ERROR_FILTER_TAU_S = 0.15
# In the velocity controller KP maps normalized image error to deg/s.
GIMBAL_PAN_KP = 52.0
GIMBAL_PAN_KI = 0.0
GIMBAL_PAN_KD = 2.5
GIMBAL_TILT_KP = 44.0
GIMBAL_TILT_KI = 0.0
GIMBAL_TILT_KD = 2.0
GIMBAL_PAN_SIGN = -1.0
GIMBAL_TILT_SIGN = -1.0
GIMBAL_MIN_STABLE = 3
GIMBAL_MIN_SCORE = 0.28
GIMBAL_LOG_EVERY_N_FRAMES = 10

# 0 = raw YOLO debug: draw every model detection, no color filtering.
# 1 = color debug: raw boxes are yellow, pink-passed boxes are green.
# 2 = final: only draw pink-passed targets with IDs.
# 3 = auto tune: color debug + save frames + print every raw/candidate stat.
RUN_MODE = 2

# Runtime speed profiles. The packaged release uses "full_frame": one 640x480
# inference per camera frame, no stale memory boxes. Tile profiles are kept only
# for experiments with smaller custom models.
SPEED_PROFILE = "full_frame"

# The current deployed model input is 640x480, so the detector sees the whole
# camera image in one pass. That avoids stale round-robin boxes for laser aiming.
FRAME_W = 640
FRAME_H = 480
# GC4653 stock lens: H-FOV 81 degrees, V-FOV 51 degrees. With the laser
# emitter 10 mm right and 60 mm below the camera, parallel-axis parallax at
# 1.0 m is approximately (+3.7, +30.2) pixels. The rounded point below is the
# fixed 1 m aim reference; production control does not need red-dot detection.
LASER_WORK_DISTANCE_MM = 1000
LASER_OFFSET_RIGHT_MM = 10
LASER_OFFSET_DOWN_MM = 60
FIXED_AIM_X = 324
FIXED_AIM_Y = 270
# The desktop monitor used for gimbal tuning is about 0.35 m away. A captured
# frame measured the visible aiming dot at y=326 there. Keep this calibration
# separate so the field profile remains correct at the required 1.0 m range.
SCREEN_TEST_AIM_X = 329
SCREEN_TEST_AIM_Y = 326
SCREEN_TEST_DISTANCE_MM = 350
# A visible aiming dot can itself produce a small low-confidence YOLO box.
# Reject only marker-sized boxes very close to the calibrated aim point.
AIM_MARKER_REJECT_RADIUS_PX = 32
AIM_MARKER_REJECT_MAX_W = 52
AIM_MARKER_REJECT_MAX_H = 56
# A 640x480 RGB888 frame is large on MaixCam. The SDK default is three
# buffers; one buffer is enough here and avoids the VI memory allocation crash.
CAMERA_BUFF_NUM = 1
USE_TILED_INFERENCE = False
TILE_OVERLAP = 160
MERGE_IOU = 0.45
ROUND_ROBIN_TILES = True
TILES_PER_FRAME = 3
MEMORY_TTL_FRAMES = 36

# Bring-up defaults: show what the model sees first. Keep laser disconnected.
# After field tuning, raise CONF_TH, turn ENABLE_COLOR_GATE back on, and require
# stable frames before sending targets to any actuator.
MIN_BOX_AREA = 18
# Reject dot-like detections that are too small to be an actionable egg mass.
# A thin mass is still accepted when either side exceeds this limit.
MAX_TINY_BOX_SIDE = 11
MAX_BOX_AREA_RATIO = 0.32
MAX_BOX_SIDE_RATIO = 0.86
MIN_ASPECT = 0.18
MAX_ASPECT = 5.5
ENABLE_COLOR_GATE = True
MIN_PINK_RATIO = 0.035
MIN_PINK_PIXELS = 1
MAX_RED_BAD_RATIO = 0.55
RED_BAD_DOMINANCE = 2.4
COLOR_GRID = 6
MAX_COLOR_CHECKS = 36
REQUIRE_STABLE_FRAMES = 2
# Keep a short prediction window when the detector misses a frame. At about
# 19 FPS this is roughly 0.3 seconds, long enough to bridge transient misses.
TRACK_MAX_MISSES = 24
TRACK_PREDICT_MAX_MISSES = 12
LOCK_REACQUIRE_RADIUS_PX = 75
LOCK_REACQUIRE_MAX_COST = 60.0
# The deployed 640x480 pipeline runs at about 5 FPS. Release a target after
# roughly five seconds of failed reacquisition so another visible egg mass can
# take over, while still bridging ordinary one- or two-frame detector misses.
LOCK_RELEASE_MISSING_FRAMES = 25
TRACK_MATCH_DISTANCE_SCALE = 1.35
TRACK_MATCH_MIN_DISTANCE_PX = 52.0
TRACK_KF_PROCESS_NOISE = 2.0
TRACK_KF_MEASUREMENT_NOISE = 16.0
TRACK_MAX_VELOCITY_PX = 18.0
GIMBAL_PREDICT_MAX_MISSES = 4
# 0 means automatically lock the first stable track; set a positive ID to
# lock a specific track later when the gimbal controller is connected.
LOCK_TARGET_ID = 0
PRIMARY_TARGET_POLICY = "robust_center"
PRIMARY_COLUMN_TOLERANCE_PX = 40
WARMUP_FRAMES = 6

if SPEED_PROFILE == "full_frame":
    USE_TILED_INFERENCE = False
    ROUND_ROBIN_TILES = False
    TILES_PER_FRAME = 1
    MEMORY_TTL_FRAMES = 0
elif SPEED_PROFILE == "accuracy":
    ROUND_ROBIN_TILES = False
    TILES_PER_FRAME = 6
    MEMORY_TTL_FRAMES = 48
elif SPEED_PROFILE == "balanced":
    ROUND_ROBIN_TILES = True
    TILES_PER_FRAME = 3
    MEMORY_TTL_FRAMES = 36
elif SPEED_PROFILE == "fast":
    ROUND_ROBIN_TILES = True
    TILES_PER_FRAME = 2
    MEMORY_TTL_FRAMES = 42
else:
    print("WARN,UNKNOWN_SPEED_PROFILE,%s" % SPEED_PROFILE)

# For laser aiming, False keeps boxes aligned with the current frame. Set True only
# if you prefer higher FPS and can tolerate one-frame image delay.
DUAL_BUFF = False

PRINT_EVERY_N_FRAMES = 10
STAT_EVERY_N_FRAMES = 30
DEBUG_DIR = "/root/snail_egg/debug"
DEBUG_CAPTURE_FLAG = "/root/snail_egg/capture_frames"
DEBUG_CAPTURE_ONCE_FLAG = "/root/snail_egg/capture_once"
DEBUG_OVERLAY_STREAM_FLAG = "/root/snail_egg/capture_overlay_stream"
DEBUG_OVERLAY_STREAM_EVERY_N_FRAMES = 3
DEBUG_OVERLAY_STREAM_MAX_SAVES = 120
DEBUG_SAVE_EVERY_N_FRAMES = 15
DEBUG_MAX_SAVES = 6
MANUAL_CAPTURE_EVERY_N_FRAMES = 60
MANUAL_CAPTURE_MAX_SAVES = 4
# Default to visual output for MaixVision/device use. The VSCode SSH helper
# creates /root/snail_egg/headless before running so automated tests do not
# block on display.Display().
ENABLE_DISPLAY = True
HEADLESS_FLAG = "/root/snail_egg/headless"
_tracks = []
_next_track_id = 1
_locked_track_id = 0
_locked_last_center = None
_locked_last_size = None
_locked_missing_frames = 0
_primary_center = None
_last_aim_dot = None


class AimRelay:
    """Non-blocking button-pulse controller for the red aiming light."""

    def __init__(self):
        pinmap.set_pin_function(AIM_RELAY_PIN, AIM_RELAY_GPIO)
        self.pin = gpio.GPIO(AIM_RELAY_GPIO, gpio.Mode.OUT)
        self.pin.value(0)
        self.logical_on = False
        self.desired_on = False
        self.transition_target = False
        self.pulses_left = 0
        self.phase = "IDLE"
        self.deadline = 0.0
        self.closed = False
        print("AIM_RELAY_INIT,%s,%s" % (AIM_RELAY_PIN, AIM_RELAY_GPIO))
        # The aiming-light controller is button driven and exposes no state
        # feedback. The proven hardware protocol starts every session with one
        # press to establish OFF before any later two-press ON command.
        self._blocking_set(False)
        self.desired_on = False
        self.transition_target = False
        self.phase = "IDLE"
        print("AIM_RELAY_STARTUP_SYNC,OFF")
        if file_exists(AIM_RELAY_STARTUP_TEST_DONE_FLAG):
            print("AIM_RELAY_STARTUP_TEST,SKIP,DONE")
        else:
            self._startup_test()
            try:
                with open(AIM_RELAY_STARTUP_TEST_DONE_FLAG, "w") as flag_file:
                    flag_file.write("1")
            except Exception as e:
                print("AIM_RELAY_STARTUP_TEST_FLAG_ERROR,%s" % e)

    def _blocking_press(self):
        self.pin.value(1)
        safe_sleep(AIM_RELAY_PULSE_S)
        self.pin.value(0)
        safe_sleep(AIM_RELAY_PULSE_S)

    def _blocking_set(self, target_on):
        for _index in range(2 if target_on else 1):
            self._blocking_press()
        self.logical_on = bool(target_on)

    def _startup_test(self):
        # State was synchronized to off in __init__; visibly flash once.
        safe_sleep(0.3)
        for cycle in range(AIM_RELAY_STARTUP_TEST_CYCLES):
            self._blocking_set(True)
            print("AIM_RELAY_STARTUP_TEST,%d,ON" % (cycle + 1))
            safe_sleep(AIM_RELAY_STARTUP_TEST_ON_S)
            self._blocking_set(False)
            print("AIM_RELAY_STARTUP_TEST,%d,OFF" % (cycle + 1))
            safe_sleep(0.3)
        self.desired_on = False
        self.transition_target = False
        self.phase = "IDLE"

    def _start_transition(self, target_on, now, force=False):
        if not force and self.logical_on == bool(target_on):
            return
        self.transition_target = bool(target_on)
        self.pulses_left = 2 if target_on else 1
        self.phase = "HIGH"
        self.pin.value(1)
        self.deadline = now + AIM_RELAY_PULSE_S
        print("AIM_RELAY_TRANSITION,%s" % ("ON" if target_on else "OFF"))

    def request(self, target_on):
        self.desired_on = bool(target_on)

    def update(self, now):
        if self.phase == "HIGH" and now >= self.deadline:
            self.pin.value(0)
            self.phase = "LOW"
            self.deadline = now + AIM_RELAY_PULSE_S
        elif self.phase == "LOW" and now >= self.deadline:
            self.pulses_left -= 1
            if self.pulses_left > 0:
                self.pin.value(1)
                self.phase = "HIGH"
                self.deadline = now + AIM_RELAY_PULSE_S
            else:
                self.phase = "IDLE"
                self.logical_on = self.transition_target
                print("AIM_RELAY_STATE,%s" % ("ON" if self.logical_on else "OFF"))
        if self.phase == "IDLE" and self.logical_on != self.desired_on:
            self._start_transition(self.desired_on, now)

    def label(self):
        if self.phase != "IDLE":
            return "AIM WAIT"
        return "AIM ON" if self.logical_on else "AIM OFF"

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.pin.value(0)
            if self.logical_on or self.transition_target:
                self.pin.value(1)
                safe_sleep(AIM_RELAY_PULSE_S)
                self.pin.value(0)
                safe_sleep(AIM_RELAY_PULSE_S)
            self.logical_on = False
            self.transition_target = False
            self.desired_on = False
            print("AIM_RELAY_CLOSED")
        except Exception as e:
            print("AIM_RELAY_CLOSE_ERROR,%s" % e)


def init_aim_relay():
    if file_exists(AIM_RELAY_DISABLE_FLAG):
        print("AIM_RELAY_SKIP,DISABLED")
        return None
    try:
        return AimRelay()
    except Exception as e:
        print("AIM_RELAY_INIT_ERROR,%s" % e)
        return None
_tile_scan_index = 0
_memory = []
_last_tile_info = "0/0"


def clamp_pan_offset(offset):
    return max(GIMBAL_PAN_MIN_DEG - GIMBAL_CENTER_DEG, min(GIMBAL_PAN_MAX_DEG - GIMBAL_CENTER_DEG, offset))


def clamp_tilt_offset(offset):
    return max(GIMBAL_TILT_MIN_DEG - GIMBAL_CENTER_DEG, min(GIMBAL_TILT_MAX_DEG - GIMBAL_CENTER_DEG, offset))


def servo_duty_from_angle(angle):
    # Standard hobby servo pulse: 0.5ms..2.5ms in a 20ms period.
    # That maps 0..180 degrees to 2.5%..12.5% duty at 50Hz.
    angle = max(0.0, min(180.0, angle))
    return 2.5 + (float(angle) / 180.0) * 10.0


def safe_sleep(seconds):
    try:
        pytime.sleep(seconds)
    except Exception:
        pass


def file_exists(path):
    try:
        return os.path.exists(path)
    except Exception:
        return False


def closed_loop_aim_requested():
    return file_exists(CLOSED_LOOP_AIM_FLAG)


def aim_scan_requested():
    return closed_loop_aim_requested() and file_exists(AIM_SCAN_FLAG)


def aim_relay_decision(fresh_target, detected_frames, missing_frames):
    """Return True/False for a relay transition, or None to keep its state."""
    if not fresh_target or missing_frames >= AIM_RELAY_OFF_MISSING_FRAMES:
        return False
    if detected_frames >= AIM_RELAY_ON_STABLE_FRAMES:
        return True
    return None


def gimbal_pwm_requested():
    if file_exists(GIMBAL_DISABLE_FLAG):
        return False
    return ENABLE_GIMBAL or file_exists(GIMBAL_ENABLE_FLAG)


def gimbal_tracking_requested():
    if file_exists(GIMBAL_DISABLE_FLAG):
        return False
    return (
        (ENABLE_GIMBAL and ENABLE_GIMBAL_TRACKING)
        or file_exists(GIMBAL_ENABLE_FLAG)
        or file_exists(GIMBAL_DRY_RUN_FLAG)
    )


SCREEN_TEST_MODE = file_exists(SCREEN_TEST_FLAG)
ACTIVE_CONF_TH = SCREEN_TEST_CONF_TH if SCREEN_TEST_MODE else CONF_TH
ACTIVE_MIN_MODEL_CONF = SCREEN_TEST_CONF_TH if SCREEN_TEST_MODE else MIN_MODEL_CONF
ACTIVE_GIMBAL_MIN_SCORE = SCREEN_TEST_CONF_TH if SCREEN_TEST_MODE else GIMBAL_MIN_SCORE
ACTIVE_AIM_X = SCREEN_TEST_AIM_X if SCREEN_TEST_MODE else FIXED_AIM_X
ACTIVE_AIM_Y = SCREEN_TEST_AIM_Y if SCREEN_TEST_MODE else FIXED_AIM_Y
ACTIVE_AIM_DISTANCE_MM = SCREEN_TEST_DISTANCE_MM if SCREEN_TEST_MODE else LASER_WORK_DISTANCE_MM


class Gimbal:
    def __init__(self):
        self.pan_offset = 0
        self.tilt_offset = 0
        self.command_count = 0
        pinmap.set_pin_function("A19", "PWM7")
        pinmap.set_pin_function("A18", "PWM6")
        center_duty = servo_duty_from_angle(GIMBAL_CENTER_DEG)
        self.pan = pwm.PWM(GIMBAL_PAN_PWM_ID, freq=GIMBAL_FREQ, duty=center_duty, enable=True)
        self.tilt = pwm.PWM(GIMBAL_TILT_PWM_ID, freq=GIMBAL_FREQ, duty=center_duty, enable=True)
        safe_sleep(GIMBAL_CENTER_SETTLE_S)

    def set_offsets(self, pan_offset, tilt_offset, source="CMD"):
        self.pan_offset = clamp_pan_offset(pan_offset)
        self.tilt_offset = clamp_tilt_offset(tilt_offset)
        pan_angle = max(GIMBAL_PAN_MIN_DEG, min(GIMBAL_PAN_MAX_DEG, GIMBAL_CENTER_DEG + self.pan_offset))
        tilt_angle = max(GIMBAL_TILT_MIN_DEG, min(GIMBAL_TILT_MAX_DEG, GIMBAL_CENTER_DEG + self.tilt_offset))
        self.pan.duty(servo_duty_from_angle(pan_angle))
        self.tilt.duty(servo_duty_from_angle(tilt_angle))
        self.command_count += 1
        if source != "TRACK" or self.command_count % GIMBAL_LOG_EVERY_N_FRAMES == 0:
            print(
                "GIMBAL,%s,PAN_OFFSET,%d,TILT_OFFSET,%d,PAN_ANGLE,%d,TILT_ANGLE,%d"
                % (source, self.pan_offset, self.tilt_offset, pan_angle, tilt_angle)
            )

    def close(self):
        try:
            self.set_offsets(0, 0, "STOP_CENTER")
            safe_sleep(GIMBAL_CENTER_SETTLE_S)
        except Exception as e:
            print("GIMBAL_STOP_CENTER_ERROR,%s" % e)
        try:
            self.pan.disable()
        except Exception as e:
            print("GIMBAL_DISABLE_PAN_ERROR,%s" % e)
        try:
            self.tilt.disable()
        except Exception as e:
            print("GIMBAL_DISABLE_TILT_ERROR,%s" % e)


class DryRunGimbal:
    """Records bounded commands without configuring pins or PWM hardware."""

    def __init__(self):
        self.pan_offset = 0.0
        self.tilt_offset = 0.0

    def set_offsets(self, pan_offset, tilt_offset, source="CMD"):
        self.pan_offset = clamp_pan_offset(pan_offset)
        self.tilt_offset = clamp_tilt_offset(tilt_offset)
        print(
            "GIMBAL,DRY_RUN,%s,PAN_OFFSET,%.2f,TILT_OFFSET,%.2f"
            % (source, self.pan_offset, self.tilt_offset)
        )

    def close(self):
        print("GIMBAL,DRY_RUN,CLOSE")


class GimbalAxisController:
    """Acceleration-limited image-error to angular-velocity controller.

    The output is an angle increment, not a raw PWM duty.  This makes the
    configured absolute mechanical limits and the per-update slew limit easy
    to enforce before touching the hardware.
    """

    def __init__(self, kp, ki, kd, sign):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.sign = float(sign)
        self.integral = 0.0
        self.previous_error = 0.0
        self.has_previous = False
        self.filtered_error = 0.0
        self.velocity = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.has_previous = False
        self.filtered_error = 0.0
        self.velocity = 0.0

    def step(self, error, dt):
        if dt <= 0.0:
            dt = 1.0 / GIMBAL_CONTROL_HZ
        alpha = 1.0 - pow(2.718281828, -dt / max(0.01, GIMBAL_ERROR_FILTER_TAU_S))
        if not self.has_previous:
            self.filtered_error = error
        else:
            self.filtered_error += alpha * (error - self.filtered_error)
        self.integral += self.filtered_error * dt
        # Integral is currently zero by default, but this clamp keeps future
        # field tuning from building a large wind-up near a hard stop.
        self.integral = max(-0.5, min(0.5, self.integral))
        derivative = 0.0
        if self.has_previous:
            derivative = (self.filtered_error - self.previous_error) / dt
        self.previous_error = self.filtered_error
        self.has_previous = True
        desired_rate = self.sign * (
            self.kp * self.filtered_error + self.ki * self.integral + self.kd * derivative
        )
        desired_rate = max(-GIMBAL_MAX_RATE_DEG_S, min(GIMBAL_MAX_RATE_DEG_S, desired_rate))
        if error == 0.0:
            desired_rate = 0.0
        max_rate_change = GIMBAL_MAX_ACCEL_DEG_S2 * dt
        rate_error = desired_rate - self.velocity
        rate_error = max(-max_rate_change, min(max_rate_change, rate_error))
        self.velocity += rate_error
        if desired_rate == 0.0 and abs(self.velocity) < max_rate_change:
            self.velocity = 0.0
        output = self.velocity * dt
        return max(-GIMBAL_MAX_STEP_DEG, min(GIMBAL_MAX_STEP_DEG, output))


class GimbalTracker:
    """Safety-gated target tracker layered on top of the gimbal PWM."""

    def __init__(self, gimbal):
        self.gimbal = gimbal
        self.pan = GimbalAxisController(GIMBAL_PAN_KP, GIMBAL_PAN_KI, GIMBAL_PAN_KD, GIMBAL_PAN_SIGN)
        self.tilt = GimbalAxisController(GIMBAL_TILT_KP, GIMBAL_TILT_KI, GIMBAL_TILT_KD, GIMBAL_TILT_SIGN)
        self.pan_offset = 0.0
        self.tilt_offset = 0.0
        self.last_update = 0.0
        self.hold_frames = 0

    def reset(self):
        self.pan.reset()
        self.tilt.reset()
        self.hold_frames = 0

    def hold(self):
        # Stop adding motion while preserving the filtered error history. This
        # avoids a derivative kick when a briefly hidden target returns.
        self.pan.velocity = 0.0
        self.tilt.velocity = 0.0
        self.hold_frames += 1

    @staticmethod
    def soften_deadzone(error, deadzone):
        """Remove sensor jitter without a discontinuous start at the boundary."""
        magnitude = abs(error)
        if magnitude <= deadzone:
            return 0.0
        scaled = (magnitude - deadzone) / max(0.001, 1.0 - deadzone)
        return scaled if error > 0.0 else -scaled

    def update(self, target, frame_w, frame_h, now, aim_dot=None, waypoint=None):
        if target is None:
            self.hold()
            return "HOLD,NO_FRESH_TARGET"
        predicted = getattr(target, "predicted", False)
        if predicted and getattr(target, "misses", GIMBAL_PREDICT_MAX_MISSES + 1) > GIMBAL_PREDICT_MAX_MISSES:
            self.reset()
            return "HOLD,PREDICTION_EXPIRED"
        if predicted:
            # Preserve filter/controller state across a brief detector miss,
            # but never steer a laser platform from extrapolation alone.
            self.hold()
            return "HOLD,PREDICT,%d" % target.track_id
        if getattr(target, "track_id", 0) <= 0:
            self.hold()
            return "HOLD,NO_TRACK_ID"
        if getattr(target, "stable", 0) < GIMBAL_MIN_STABLE:
            self.hold()
            return "HOLD,UNSTABLE"
        if not predicted and (getattr(target, "misses", 1) != 0 or getattr(target, "score", 0.0) < ACTIVE_GIMBAL_MIN_SCORE):
            self.hold()
            return "HOLD,QUALITY"

        closed_loop = closed_loop_aim_requested()
        if closed_loop and (aim_dot is None or not getattr(aim_dot, "fresh", False)):
            self.hold()
            return "HOLD,DOT_MISSING"

        interval = 1.0 / GIMBAL_CONTROL_HZ
        if self.last_update > 0.0 and now - self.last_update < interval:
            return "HOLD,RATE"
        dt = interval if self.last_update <= 0.0 else max(0.001, now - self.last_update)
        self.last_update = now

        if waypoint is None:
            desired_x = target.x + target.w * 0.5
            desired_y = target.y + target.h * 0.5
        else:
            desired_x, desired_y = waypoint
        reference_x = aim_dot.x if closed_loop else ACTIVE_AIM_X
        reference_y = aim_dot.y if closed_loop else ACTIVE_AIM_Y
        error_x = (desired_x - reference_x) / max(1.0, frame_w * 0.5)
        error_y = (desired_y - reference_y) / max(1.0, frame_h * 0.5)
        error_x = self.soften_deadzone(error_x, GIMBAL_DEADZONE_X)
        error_y = self.soften_deadzone(error_y, GIMBAL_DEADZONE_Y)

        self.pan_offset = clamp_pan_offset(self.pan_offset + self.pan.step(error_x, dt))
        self.tilt_offset = clamp_tilt_offset(self.tilt_offset + self.tilt.step(error_y, dt))
        self.hold_frames = 0
        self.gimbal.set_offsets(self.pan_offset, self.tilt_offset, "TRACK")
        state = "PREDICT" if predicted else "TRACK"
        return "%s,%d,EX,%.3f,EY,%.3f,PAN,%.2f,TILT,%.2f,REF,%d,%d,DES,%d,%d,RAWID,%d" % (
            state,
            getattr(target, "aim_lock_id", target.track_id),
            error_x,
            error_y,
            self.pan_offset,
            self.tilt_offset,
            int(reference_x),
            int(reference_y),
            int(desired_x),
            int(desired_y),
            target.track_id,
        )


def init_gimbal_tracker(gimbal):
    if gimbal is None or not gimbal_tracking_requested():
        print("GIMBAL_TRACK_SKIP,DISABLED")
        return None
    print("GIMBAL_TRACK_OK,HZ,%.1f,DEADZONE,%.3f,MAX_STEP,%.1f,AIM,%d,%d,DIST_MM,%d" % (
        GIMBAL_CONTROL_HZ,
        GIMBAL_DEADZONE_X,
        GIMBAL_MAX_STEP_DEG,
        ACTIVE_AIM_X,
        ACTIVE_AIM_Y,
        ACTIVE_AIM_DISTANCE_MM,
    ))
    return GimbalTracker(gimbal)


def init_gimbal():
    if file_exists(GIMBAL_DRY_RUN_FLAG) and not file_exists(GIMBAL_DISABLE_FLAG):
        print("GIMBAL_INIT_DRY_RUN")
        return DryRunGimbal()
    if not gimbal_pwm_requested():
        print("GIMBAL_SKIP,DISABLED")
        return None
    try:
        print("GIMBAL_INIT_START,PAN,%d-%d,TILT,%d-%d" % (
            GIMBAL_PAN_MIN_DEG, GIMBAL_PAN_MAX_DEG, GIMBAL_TILT_MIN_DEG, GIMBAL_TILT_MAX_DEG
        ))
        gimbal = Gimbal()
        print("GIMBAL_INIT_OK")
        return gimbal
    except Exception as e:
        print("GIMBAL_INIT_ERROR,%s" % e)
        return None


class DetectedBox:
    def __init__(self, x, y, w, h, score, track_id=0, predicted=False, stable=0, misses=0):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.score = float(score)
        self.track_id = int(track_id)
        self.predicted = bool(predicted)
        self.stable = int(stable)
        self.misses = int(misses)


def draw_cross(img, cx, cy, color):
    try:
        img.draw_cross(cx, cy, color, size=7, thickness=2)
    except Exception:
        img.draw_line(cx - 7, cy, cx + 7, cy, color, 2)
        img.draw_line(cx, cy - 7, cx, cy + 7, color, 2)


def obj_center(obj):
    return int(obj.x + obj.w // 2), int(obj.y + obj.h // 2)


def box_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def pink_pixel(r, g, b):
    # Pink egg pixels keep a useful blue component. Pure red/orange parts have
    # high R but low B, so reject only clearly red pixels.
    if r >= 125 and b < 50 and (r - g) >= 32:
        return False
    if r >= 150 and (r - b) > 135:
        return False
    if r >= 145 and g >= 75 and b < 50 and (r - g) >= 24:
        return False

    return (
        r >= 80
        and b >= 50
        and g >= 40
        and ((r - g) >= 10 or (b - g) >= 12)
        and (r - b) <= 120
        and (r - b) >= -45
    )


def red_bad_pixel(r, g, b):
    pure_red = r >= 125 and b < 50 and (r - g) >= 32
    orange_red = r >= 150 and g >= 70 and b < 62 and (r - b) >= 110
    return pure_red or orange_red


def pixel_to_rgb(px):
    try:
        if isinstance(px, (tuple, list)):
            if len(px) >= 3:
                return int(px[0]), int(px[1]), int(px[2])
            if len(px) == 1:
                px = px[0]
            else:
                return 0, 0, 0
        value = int(px)
        if value > 0xFFFF:
            return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
        r = ((value >> 11) & 0x1F) * 255 // 31
        g = ((value >> 5) & 0x3F) * 255 // 63
        b = (value & 0x1F) * 255 // 31
        return r, g, b
    except Exception:
        return 0, 0, 0


class AimDot:
    def __init__(self, x, y, score, fresh=True, misses=0):
        self.x = float(x)
        self.y = float(y)
        self.score = float(score)
        self.fresh = bool(fresh)
        self.misses = int(misses)


class AimDotDetector:
    """Detect and smooth the low-power red aiming dot in the camera image."""

    def __init__(self):
        self.x = None
        self.y = None
        self.score = 0.0
        self.misses = AIM_DOT_STALE_FRAMES + 1
        self.debug_frames = 0
        self.rgb_locked = False

    @staticmethod
    def _red_score(img, x, y, w, h):
        best = None
        # A laser spot often has a white center and a saturated red halo.
        # Sample a small grid and keep the strongest red-dominant pixel.
        for fy in (0.20, 0.50, 0.80):
            for fx in (0.20, 0.50, 0.80):
                px = max(0, min(FRAME_W - 1, int(x + w * fx)))
                py = max(0, min(FRAME_H - 1, int(y + h * fy)))
                r, g, b = pixel_to_rgb(img.get_pixel(px, py))
                dominance = r - max(g, b)
                score = r + 2.0 * dominance
                if best is None or score > best[0]:
                    best = (score, r, dominance)
        return best if best is not None else (0.0, 0, 0)

    @staticmethod
    def _pixel_metrics(img, x, y):
        r, g, b = pixel_to_rgb(img.get_pixel(int(x), int(y)))
        dominance = r - max(g, b)
        ratio = dominance / float(max(1, r))
        return r + 2.0 * dominance, r, dominance, ratio

    def _rgb_peak_candidate(self, img, center_x, center_y, radius, stride=AIM_DOT_RGB_STRIDE):
        # MaixPy LAB blobs can fragment a laser halo on textured/pink surfaces.
        # A sparse search in a tiny optical-axis ROI is deterministic and much
        # cheaper than scanning the full 640x480 image in Python.
        x0 = max(0, int(center_x - radius))
        y0 = max(0, int(center_y - radius))
        x1 = min(FRAME_W - 1, int(center_x + radius))
        y1 = min(FRAME_H - 1, int(center_y + radius))
        best = None
        best_any = None
        for py in range(y0, y1 + 1, stride):
            for px in range(x0, x1 + 1, stride):
                score, red_value, dominance, ratio_value = self._pixel_metrics(img, px, py)
                if best_any is None or score > best_any[0]:
                    best_any = (score, red_value, dominance, ratio_value, px, py)
                if (
                    red_value < AIM_DOT_MIN_RED
                    or dominance < AIM_DOT_MIN_DOMINANCE
                    or ratio_value < AIM_DOT_MIN_DOMINANCE_RATIO
                ):
                    continue
                dx = px - center_x
                dy = py - center_y
                ranked_score = score - (dx * dx + dy * dy) ** 0.5 * 0.25
                if best is None or ranked_score > best[0]:
                    best = (ranked_score, px, py)

        if best is None:
            return None, best_any

        _, peak_x, peak_y = best
        _, _, peak_dominance, _ = self._pixel_metrics(img, peak_x, peak_y)
        surround_dominance = []
        for ox, oy in ((-10, 0), (10, 0), (0, -10), (0, 10), (-7, -7), (7, -7), (7, 7), (-7, 7)):
            sx = max(0, min(FRAME_W - 1, peak_x + ox))
            sy = max(0, min(FRAME_H - 1, peak_y + oy))
            _, _, local_dominance, _ = self._pixel_metrics(img, sx, sy)
            surround_dominance.append(local_dominance)
        mean_surround = sum(surround_dominance) / float(len(surround_dominance))
        if peak_dominance - mean_surround < AIM_DOT_MIN_LOCAL_CONTRAST:
            return None, best_any

        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        peak_score = 0.0
        for py in range(max(0, peak_y - 5), min(FRAME_H, peak_y + 6)):
            for px in range(max(0, peak_x - 5), min(FRAME_W, peak_x + 6)):
                score, red_value, dominance, ratio_value = self._pixel_metrics(img, px, py)
                if (
                    red_value >= AIM_DOT_MIN_RED
                    and dominance >= AIM_DOT_MIN_DOMINANCE
                    and ratio_value >= AIM_DOT_MIN_DOMINANCE_RATIO
                ):
                    weight = max(1.0, float(dominance - AIM_DOT_MIN_DOMINANCE + 1))
                    weighted_x += px * weight
                    weighted_y += py * weight
                    total_weight += weight
                    peak_score = max(peak_score, score)
        if total_weight <= 0.0:
            return None, best_any
        return (peak_score, weighted_x / total_weight, weighted_y / total_weight), best_any

    def detect(self, img, enabled):
        if not enabled:
            self.misses = AIM_DOT_STALE_FRAMES + 1
            return None
        self.debug_frames += 1
        rgb_refresh = (not self.rgb_locked) or self.debug_frames % AIM_DOT_RGB_REFRESH_FRAMES == 0
        if self.rgb_locked and not rgb_refresh:
            if self.x is not None and self.misses <= AIM_DOT_STALE_FRAMES:
                return AimDot(self.x, self.y, self.score, self.misses == 0, self.misses)
            return None

        blobs = []
        try:
            tracking_previous = self.x is not None and self.misses <= AIM_DOT_STALE_FRAMES
            radius = AIM_DOT_MAX_JUMP_PX if tracking_previous else AIM_DOT_EXPECTED_RADIUS_PX
            center_x = self.x if tracking_previous else ACTIVE_AIM_X
            center_y = self.y if tracking_previous else ACTIVE_AIM_Y
            roi_x = max(0, int(center_x - radius))
            roi_y = max(0, int(center_y - radius))
            roi_w = min(FRAME_W - roi_x, int(radius * 2))
            roi_h = min(FRAME_H - roi_y, int(radius * 2))
            if not self.rgb_locked:
                blobs = img.find_blobs(
                    AIM_DOT_LAB_THRESHOLDS,
                    roi=[roi_x, roi_y, roi_w, roi_h],
                    x_stride=1,
                    y_stride=1,
                    area_threshold=AIM_DOT_MIN_PIXELS,
                    pixels_threshold=AIM_DOT_MIN_PIXELS,
                    merge=False,
                )
        except Exception as e:
            print("AIM_DOT_ERROR,%s" % e)
            blobs = []

        candidates = []
        expected_x = self.x if tracking_previous else ACTIVE_AIM_X
        expected_y = self.y if tracking_previous else ACTIVE_AIM_Y
        for blob in blobs:
            x, y, w, h = int(blob[0]), int(blob[1]), int(blob[2]), int(blob[3])
            pixels = int(blob[4])
            if w < AIM_DOT_MIN_SIDE or h < AIM_DOT_MIN_SIDE:
                continue
            if w > AIM_DOT_MAX_SIDE or h > AIM_DOT_MAX_SIDE:
                continue
            if pixels < AIM_DOT_MIN_PIXELS or pixels > AIM_DOT_MAX_PIXELS:
                continue
            cx = float(blob[5])
            cy = float(blob[6])
            dx = cx - expected_x
            dy = cy - expected_y
            distance = (dx * dx + dy * dy) ** 0.5
            if not tracking_previous and distance > AIM_DOT_EXPECTED_RADIUS_PX:
                continue
            if tracking_previous and distance > AIM_DOT_MAX_JUMP_PX:
                continue
            red_score, red_value, dominance = self._red_score(img, x, y, w, h)
            dominance_ratio = dominance / float(max(1, red_value))
            if (
                red_value < AIM_DOT_MIN_RED
                or dominance < AIM_DOT_MIN_DOMINANCE
                or dominance_ratio < AIM_DOT_MIN_DOMINANCE_RATIO
            ):
                continue
            compactness = pixels / float(max(1, w * h))
            score = red_score + compactness * 30.0 - distance * 0.25
            candidates.append((score, cx, cy))

        # The real spot has a white core and locally dominant red halo. Always
        # let a sparse fixed-axis RGB peak override LAB fragments, which can be
        # confused by pink egg granules after the spot reaches the target.
        rgb_peak, fallback_debug = self._rgb_peak_candidate(
            img,
            ACTIVE_AIM_X,
            ACTIVE_AIM_Y,
            AIM_DOT_EXPECTED_RADIUS_PX,
        )
        if rgb_peak is not None:
            self.rgb_locked = True
            candidates = [rgb_peak]
        elif self.rgb_locked:
            candidates = []

        if self.debug_frames % AIM_DOT_LOG_EVERY_N_FRAMES == 0 and fallback_debug is not None:
            score, red_value, dominance, ratio_value, px, py = fallback_debug
            print(
                "DOT_RGB_PEAK,%d,%d,R,%d,DOM,%d,RATIO,%.3f,SCORE,%.1f,BLOBS,%d"
                % (px, py, red_value, dominance, ratio_value, score, len(blobs))
            )

        if candidates:
            score, cx, cy = max(candidates, key=lambda item: item[0])
            if self.x is None or self.misses > AIM_DOT_STALE_FRAMES:
                self.x, self.y = cx, cy
            else:
                alpha = AIM_DOT_FILTER_ALPHA
                self.x += alpha * (cx - self.x)
                self.y += alpha * (cy - self.y)
            self.score = score
            self.misses = 0
            return AimDot(self.x, self.y, score, True, 0)

        self.misses += 1
        if self.x is not None and self.misses <= AIM_DOT_STALE_FRAMES:
            return AimDot(self.x, self.y, self.score, False, self.misses)
        return None


class AimWaypointPlanner:
    """Generate a bounded low-power aiming path inside the locked egg box."""

    PATTERN = (
        (0.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
    )

    def __init__(self):
        self.track_id = 0
        self.index = 0
        self.settled_frames = 0
        self.settled_since = 0.0

    def reset(self):
        self.track_id = 0
        self.index = 0
        self.settled_frames = 0
        self.settled_since = 0.0

    def point(self, target, aim_dot, now, scan_enabled):
        if target is None:
            self.settled_frames = 0
            return None
        track_id = getattr(target, "track_id", 0)
        if track_id != self.track_id:
            self.track_id = track_id
            self.index = 0
            self.settled_frames = 0
            self.settled_since = now
        if not scan_enabled:
            self.index = 0

        nx, ny = self.PATTERN[self.index]
        half_w = max(2.0, target.w * 0.5 * (1.0 - AIM_SCAN_MARGIN_RATIO))
        half_h = max(2.0, target.h * 0.5 * (1.0 - AIM_SCAN_MARGIN_RATIO))
        point = (target.x + target.w * 0.5 + nx * half_w, target.y + target.h * 0.5 + ny * half_h)

        if scan_enabled and aim_dot is not None and getattr(aim_dot, "fresh", False):
            dx = aim_dot.x - point[0]
            dy = aim_dot.y - point[1]
            if (dx * dx + dy * dy) ** 0.5 <= AIM_DOT_CONTROL_TOLERANCE_PX:
                self.settled_frames += 1
                if self.settled_frames == 1:
                    self.settled_since = now
                if (
                    self.settled_frames >= AIM_SCAN_SETTLE_FRAMES
                    and now - self.settled_since >= AIM_SCAN_DWELL_S
                ):
                    self.index = (self.index + 1) % len(self.PATTERN)
                    self.settled_frames = 0
                    self.settled_since = now
            else:
                self.settled_frames = 0
                self.settled_since = now
        return point


def color_gate(img, x, y, w, h):
    if not ENABLE_COLOR_GATE or RUN_MODE == 0:
        return True, 1.0, 0.0
    total = 0
    pink = 0
    red_bad = 0
    step_x = max(1, w // COLOR_GRID)
    step_y = max(1, h // COLOR_GRID)
    start_x = x + max(0, w // 10)
    start_y = y + max(0, h // 10)
    end_x = x + w - max(0, w // 10)
    end_y = y + h - max(0, h // 10)
    try:
        yy = start_y
        while yy < end_y:
            xx = start_x
            while xx < end_x:
                r, g, b = pixel_to_rgb(img.get_pixel(xx, yy))
                if pink_pixel(r, g, b):
                    pink += 1
                if red_bad_pixel(r, g, b):
                    red_bad += 1
                total += 1
                if total >= MAX_COLOR_CHECKS:
                    break
                xx += step_x
            if total >= MAX_COLOR_CHECKS:
                break
            yy += step_y
    except Exception:
        return False, 0.0, 0.0
    ratio = pink / total if total > 0 else 0.0
    red_ratio = red_bad / total if total > 0 else 0.0
    if red_ratio > MAX_RED_BAD_RATIO and red_bad > pink * RED_BAD_DOMINANCE:
        return False, ratio, red_ratio
    return pink >= MIN_PINK_PIXELS and ratio >= MIN_PINK_RATIO, ratio, red_ratio


def pass_geometry(obj, frame_w, frame_h):
    w = int(obj.w)
    h = int(obj.h)
    area = w * h
    if area < MIN_BOX_AREA:
        return False
    if w <= MAX_TINY_BOX_SIDE and h <= MAX_TINY_BOX_SIDE:
        return False
    if area / float(frame_w * frame_h) > MAX_BOX_AREA_RATIO:
        return False
    if w / float(frame_w) > MAX_BOX_SIDE_RATIO or h / float(frame_h) > MAX_BOX_SIDE_RATIO:
        return False
    aspect = w / float(h) if h > 0 else 99
    return MIN_ASPECT <= aspect <= MAX_ASPECT


def is_aim_marker_box(obj):
    w = int(obj.w)
    h = int(obj.h)
    if w > AIM_MARKER_REJECT_MAX_W or h > AIM_MARKER_REJECT_MAX_H:
        return False
    cx = int(obj.x) + w // 2
    cy = int(obj.y) + h // 2
    marker_x = _last_aim_dot.x if _last_aim_dot is not None else ACTIVE_AIM_X
    marker_y = _last_aim_dot.y if _last_aim_dot is not None else ACTIVE_AIM_Y
    return (
        abs(cx - marker_x) <= AIM_MARKER_REJECT_RADIUS_PX
        and abs(cy - marker_y) <= AIM_MARKER_REJECT_RADIUS_PX
    )


def filter_candidates(img, objs, frame_w, frame_h, frame_id):
    kept = []
    rows = []
    best_pink_ratio = 0.0
    best_red_ratio = 0.0
    for idx, obj in enumerate(objs):
        geometry_ok = pass_geometry(obj, frame_w, frame_h)
        if not geometry_ok:
            rows.append((idx, obj, False, False, 0.0, 0.0))
            continue
        # In monitor mode, reject only the actually measured dot. Outdoors,
        # retain the calibrated fixed-point fallback for startup frames.
        if ((_last_aim_dot is not None or not SCREEN_TEST_MODE) and is_aim_marker_box(obj)):
            rows.append((idx, obj, True, False, 0.0, 0.0))
            continue
        x = max(0, int(obj.x))
        y = max(0, int(obj.y))
        w = min(frame_w - x, int(obj.w))
        h = min(frame_h - y, int(obj.h))
        if RUN_MODE == 3 and frame_id < 2 and idx < 4:
            sx = x + w // 2
            sy = y + h // 2
            try:
                px = img.get_pixel(sx, sy)
                r0, g0, b0 = pixel_to_rgb(px)
                print("PIX,%d,%d,%d,%d,%s,%d,%d,%d" % (frame_id, idx, sx, sy, str(px), r0, g0, b0))
            except Exception as e:
                print("PIX_ERROR,%d,%d,%s" % (frame_id, idx, e))
        ok, pink_ratio, red_ratio = color_gate(img, x, y, w, h)
        rows.append((idx, obj, True, ok, pink_ratio, red_ratio))
        if pink_ratio > best_pink_ratio:
            best_pink_ratio = pink_ratio
        if red_ratio > best_red_ratio:
            best_red_ratio = red_ratio
        area = w * h
        red_reject = red_ratio > MAX_RED_BAD_RATIO and red_ratio > pink_ratio * RED_BAD_DOMINANCE
        score_ok = obj.score >= ACTIVE_MIN_MODEL_CONF
        if SCREEN_TEST_MODE:
            # Monitor gamma and viewing angle can remove most measured pink
            # even though the trained detector still recognizes the egg mass.
            # This branch is opt-in and never changes the outdoor profile.
            color_ok = True
        elif score_ok and obj.score >= STRONG_MODEL_CONF:
            color_ok = not red_reject
        else:
            color_ok = ok and pink_ratio >= LOW_CONF_MIN_PINK_RATIO and area >= LOW_CONF_MIN_AREA
        if not color_ok or not score_ok:
            continue
        kept.append(obj)
    return kept, best_pink_ratio, best_red_ratio, rows


class ScalarKalman:
    """Small constant-velocity Kalman filter for one scalar value."""

    def __init__(self, value):
        self.pos = float(value)
        self.vel = 0.0
        self.p00 = 25.0
        self.p01 = 0.0
        self.p11 = 9.0

    def predict(self):
        self.pos += self.vel
        self.vel = max(-TRACK_MAX_VELOCITY_PX, min(TRACK_MAX_VELOCITY_PX, self.vel))
        old_p01 = self.p01
        self.p00 = self.p00 + 2.0 * old_p01 + self.p11 + TRACK_KF_PROCESS_NOISE
        self.p01 = old_p01 + self.p11
        self.p11 += TRACK_KF_PROCESS_NOISE * 0.25

    def update(self, measurement):
        measurement = float(measurement)
        innovation = measurement - self.pos
        innovation_cov = self.p00 + TRACK_KF_MEASUREMENT_NOISE
        if innovation_cov <= 0.0:
            return
        k0 = self.p00 / innovation_cov
        k1 = self.p01 / innovation_cov
        old_p01 = self.p01
        self.pos += k0 * innovation
        self.vel += k1 * innovation
        self.p00 = max(0.01, (1.0 - k0) * self.p00)
        self.p01 = (1.0 - k0) * old_p01
        self.p11 = max(0.01, self.p11 - k1 * old_p01)


def track_box(track):
    cx = track["cx"].pos
    cy = track["cy"].pos
    width = max(2.0, track["w"].pos)
    height = max(2.0, track["h"].pos)
    return (
        int(cx - width * 0.5),
        int(cy - height * 0.5),
        int(width),
        int(height),
    )


def track_object(track, predicted):
    x, y, w, h = track_box(track)
    score = track["score"] if not predicted else max(0.0, track["score"] * 0.98)
    return DetectedBox(x, y, w, h, score, track["id"], predicted, track["stable"], track["misses"])


def dedupe_track_objects(objs):
    ordered = sorted(
        objs,
        key=lambda obj: (1 if getattr(obj, "predicted", False) else 0, getattr(obj, "misses", 0), -obj.score),
    )
    kept = []
    for obj in ordered:
        box = (int(obj.x), int(obj.y), int(obj.w), int(obj.h))
        cx, cy = obj_center(obj)
        duplicate = False
        for other in kept:
            other_box = (int(other.x), int(other.y), int(other.w), int(other.h))
            ox, oy = obj_center(other)
            near = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < max(
                18.0, 0.35 * max(obj.w, obj.h, other.w, other.h)
            )
            if box_iou(box, other_box) >= 0.15 or near:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)
    return kept


def new_track(obj):
    global _next_track_id
    cx, cy = obj_center(obj)
    track = {
        "id": _next_track_id,
        "cx": ScalarKalman(cx),
        "cy": ScalarKalman(cy),
        "w": ScalarKalman(max(2, obj.w)),
        "h": ScalarKalman(max(2, obj.h)),
        "score": obj.score,
        "stable": 1,
        "misses": 0,
        "age": 1,
    }
    _next_track_id += 1
    track["obj"] = track_object(track, False)
    return track


def track_match_cost(obj, track):
    predicted_box = track_box(track)
    measured_box = (int(obj.x), int(obj.y), int(obj.w), int(obj.h))
    predicted_cx, predicted_cy = obj_center(DetectedBox(*predicted_box, 0.0))
    measured_cx, measured_cy = obj_center(obj)
    dx = measured_cx - predicted_cx
    dy = measured_cy - predicted_cy
    distance = (dx * dx + dy * dy) ** 0.5
    scale = max(TRACK_MATCH_MIN_DISTANCE_PX, max(predicted_box[2], predicted_box[3]) * TRACK_MATCH_DISTANCE_SCALE)
    overlap = box_iou(measured_box, predicted_box)
    if overlap < 0.02 and distance > scale:
        return None
    return distance / scale + (1.0 - overlap) * 0.65


def update_tracks(objs):
    global _tracks
    for track in _tracks:
        track["cx"].predict()
        track["cy"].predict()
        track["w"].predict()
        track["h"].predict()

    pairs = []
    for det_idx, obj in enumerate(objs):
        for track_idx, track in enumerate(_tracks):
            cost = track_match_cost(obj, track)
            if cost is not None:
                pairs.append((cost, det_idx, track_idx))
    pairs.sort(key=lambda item: item[0])

    used_detections = set()
    used_tracks = set()
    next_tracks = []
    for cost, det_idx, track_idx in pairs:
        if det_idx in used_detections or track_idx in used_tracks:
            continue
        obj = objs[det_idx]
        track = _tracks[track_idx]
        used_detections.add(det_idx)
        used_tracks.add(track_idx)
        track["cx"].update(obj_center(obj)[0])
        track["cy"].update(obj_center(obj)[1])
        track["w"].update(max(2, obj.w))
        track["h"].update(max(2, obj.h))
        track["score"] = obj.score
        track["stable"] += 1
        track["age"] += 1
        track["misses"] = 0
        track["obj"] = track_object(track, False)
        next_tracks.append(track)

    for track_idx, track in enumerate(_tracks):
        if track_idx in used_tracks:
            continue
        track["misses"] += 1
        track["age"] += 1
        if track["misses"] <= TRACK_MAX_MISSES:
            track["obj"] = track_object(track, True)
            next_tracks.append(track)

    for det_idx, obj in enumerate(objs):
        if det_idx not in used_detections:
            next_tracks.append(new_track(obj))

    _tracks = next_tracks
    visible = [
        track["obj"]
        for track in _tracks
        if track["stable"] >= REQUIRE_STABLE_FRAMES
        and track["misses"] <= TRACK_PREDICT_MAX_MISSES
    ]
    deduped = dedupe_track_objects(visible)
    return deduped


def select_primary_target(objs):
    global _locked_track_id, _locked_last_center, _locked_last_size, _locked_missing_frames, _primary_center
    if PRIMARY_TARGET_POLICY in ("leftmost", "robust_center"):
        fresh = [obj for obj in objs if not getattr(obj, "predicted", False)]
        locked_fresh = [obj for obj in fresh if getattr(obj, "track_id", 0) == _locked_track_id]
        locked_predicted = [
            obj
            for obj in objs
            if getattr(obj, "track_id", 0) == _locked_track_id and getattr(obj, "predicted", False)
        ]
        if locked_fresh:
            chosen = locked_fresh[0]
            chosen.aim_lock_id = 1
            _primary_center = obj_center(chosen)
            _locked_last_center = _primary_center
            _locked_last_size = (chosen.w, chosen.h)
            _locked_missing_frames = 0
            return chosen
        if not fresh:
            if locked_predicted and getattr(locked_predicted[0], "misses", 99) <= GIMBAL_PREDICT_MAX_MISSES:
                locked_predicted[0].aim_lock_id = 1
                return locked_predicted[0]
            return None
        if _locked_track_id > 0:
            _locked_missing_frames += 1
            if _locked_last_center is not None and fresh:
                lx, ly = _locked_last_center
                lw, lh = _locked_last_size if _locked_last_size is not None else (1.0, 1.0)
                reacquire = []
                for obj in fresh:
                    cx, cy = obj_center(obj)
                    distance = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                    width_ratio = obj.w / max(1.0, lw)
                    height_ratio = obj.h / max(1.0, lh)
                    old_aspect = lw / max(1.0, lh)
                    new_aspect = obj.w / max(1.0, obj.h)
                    aspect_ratio = new_aspect / max(0.01, old_aspect)
                    if (
                        distance <= LOCK_REACQUIRE_RADIUS_PX
                        and 0.45 <= width_ratio <= 2.2
                        and 0.45 <= height_ratio <= 2.2
                        and 0.45 <= aspect_ratio <= 2.0
                    ):
                        size_cost = abs(1.0 - width_ratio) + abs(1.0 - height_ratio)
                        shape_cost = abs(1.0 - aspect_ratio)
                        reacquire.append((distance + 18.0 * size_cost + 24.0 * shape_cost, obj))
                if reacquire:
                    best_reacquire = min(reacquire, key=lambda item: item[0])
                    if best_reacquire[0] <= LOCK_REACQUIRE_MAX_COST:
                        chosen = best_reacquire[1]
                        _locked_track_id = getattr(chosen, "track_id", 0)
                        chosen.aim_lock_id = 1
                        _primary_center = obj_center(chosen)
                        _locked_last_center = _primary_center
                        _locked_last_size = (chosen.w, chosen.h)
                        _locked_missing_frames = 0
                        print(
                            "LOCK,REACQUIRE,%d,%d,%d,COST,%.1f"
                            % (_locked_track_id, _primary_center[0], _primary_center[1], best_reacquire[0])
                        )
                        return chosen
            if locked_predicted and getattr(locked_predicted[0], "misses", 99) <= GIMBAL_PREDICT_MAX_MISSES:
                locked_predicted[0].aim_lock_id = 1
                return locked_predicted[0]
            if _locked_missing_frames <= LOCK_RELEASE_MISSING_FRAMES:
                return None
            print("LOCK,RELEASE,%d,MISSES,%d" % (_locked_track_id, _locked_missing_frames))
            _locked_track_id = 0
        if PRIMARY_TARGET_POLICY == "leftmost":
            min_x = min(obj_center(obj)[0] for obj in fresh)
            left_column = [obj for obj in fresh if obj_center(obj)[0] <= min_x + PRIMARY_COLUMN_TOLERANCE_PX]
            chosen = sorted(left_column, key=lambda obj: obj_center(obj)[1])[0]
        else:
            # Prefer a large, confident, central mass. It survives camera
            # motion and a bright aiming spot much better than a tiny edge box.
            def robust_score(obj):
                cx, cy = obj_center(obj)
                dx = (cx - FRAME_W * 0.5) / max(1.0, FRAME_W * 0.5)
                dy = (cy - FRAME_H * 0.5) / max(1.0, FRAME_H * 0.5)
                center_weight = max(0.25, 1.25 - (dx * dx + dy * dy) ** 0.5)
                border_weight = 0.35 if (
                    obj.x <= 4
                    or obj.y <= 4
                    or obj.x + obj.w >= FRAME_W - 4
                    or obj.y + obj.h >= FRAME_H - 4
                ) else 1.0
                return obj.w * obj.h * (0.5 + obj.score) * center_weight * border_weight

            chosen = max(fresh, key=robust_score)
        _locked_track_id = getattr(chosen, "track_id", 0)
        chosen.aim_lock_id = 1
        _primary_center = obj_center(chosen)
        _locked_last_center = _primary_center
        _locked_last_size = (chosen.w, chosen.h)
        _locked_missing_frames = 0
        return chosen
    ids = [getattr(obj, "track_id", 0) for obj in objs]
    if LOCK_TARGET_ID > 0:
        _locked_track_id = LOCK_TARGET_ID if LOCK_TARGET_ID in ids else 0
    for obj in objs:
        if getattr(obj, "track_id", 0) == _locked_track_id:
            _locked_last_center = obj_center(obj)
            _locked_missing_frames = 0
            return obj
    if _locked_track_id > 0:
        _locked_missing_frames += 1
        if _locked_last_center is not None and objs:
            lx, ly = _locked_last_center
            nearby = sorted(
                objs,
                key=lambda obj: (obj_center(obj)[0] - lx) ** 2 + (obj_center(obj)[1] - ly) ** 2,
            )
            chosen = nearby[0]
            cx, cy = obj_center(chosen)
            if ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5 <= LOCK_REACQUIRE_RADIUS_PX:
                _locked_track_id = getattr(chosen, "track_id", 0)
                _locked_last_center = (cx, cy)
                _locked_missing_frames = 0
                return chosen
        if _locked_missing_frames < LOCK_RELEASE_MISSING_FRAMES:
            return None
        _locked_track_id = 0
        _locked_last_center = None
    if objs:
        chosen = sorted(objs, key=lambda obj: obj.score, reverse=True)[0]
        _locked_track_id = getattr(chosen, "track_id", 0)
        _locked_last_center = obj_center(chosen)
        _locked_missing_frames = 0
        return chosen
    return None


def merge_overlaps(objs):
    ordered = sorted(objs, key=lambda obj: obj.score, reverse=True)
    kept = []
    for obj in ordered:
        box = (int(obj.x), int(obj.y), int(obj.w), int(obj.h))
        duplicate = False
        for other in kept:
            other_box = (int(other.x), int(other.y), int(other.w), int(other.h))
            cx, cy = obj_center(obj)
            ox, oy = obj_center(other)
            near = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < max(
                16.0, 0.30 * max(obj.w, obj.h, other.w, other.h)
            )
            if box_iou(box, other_box) >= 0.18 or near:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)
    return kept


def sort_targets(objs, frame_h=320):
    row_h = max(24, frame_h // 12)
    enriched = []
    for obj in objs:
        cx, cy = obj_center(obj)
        enriched.append((obj, cx, cy))
    enriched.sort(key=lambda t: ((t[2] // row_h), t[1]))
    return enriched[:MAX_TARGETS]


def tile_origins(frame_w, frame_h, tile_w, tile_h):
    if not USE_TILED_INFERENCE:
        return [(0, 0)]
    step_x = max(1, tile_w - TILE_OVERLAP)
    step_y = max(1, tile_h - TILE_OVERLAP)
    xs = [0]
    ys = [0]
    x = step_x
    while x < frame_w - tile_w:
        xs.append(x)
        x += step_x
    if frame_w > tile_w:
        xs.append(frame_w - tile_w)
    y = step_y
    while y < frame_h - tile_h:
        ys.append(y)
        y += step_y
    if frame_h > tile_h:
        ys.append(frame_h - tile_h)
    return [(x, y) for y in ys for x in xs]


def crop_tile(img, x, y, w, h):
    try:
        return img.crop(x, y, w, h)
    except Exception:
        try:
            return img.copy(x, y, w, h)
        except Exception:
            return None


def remember_detections(new_objs):
    global _memory
    next_memory = []
    for item in _memory:
        item["ttl"] -= 1
        if item["ttl"] > 0:
            next_memory.append(item)

    for obj in new_objs:
        box = (int(obj.x), int(obj.y), int(obj.w), int(obj.h))
        best_idx = -1
        best_score = 0.0
        for idx, item in enumerate(next_memory):
            score = box_iou(box, item["box"])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= 0.35:
            next_memory[best_idx] = {"obj": obj, "box": box, "ttl": MEMORY_TTL_FRAMES}
        else:
            next_memory.append({"obj": obj, "box": box, "ttl": MEMORY_TTL_FRAMES})

    objs = merge_overlaps([item["obj"] for item in next_memory])
    _memory = [{"obj": obj, "box": (int(obj.x), int(obj.y), int(obj.w), int(obj.h)), "ttl": MEMORY_TTL_FRAMES} for obj in objs]
    return objs


def detect_frame(detector, img, frame_id):
    global _tile_scan_index, _last_tile_info
    frame_w = FRAME_W
    frame_h = FRAME_H
    tile_w = detector.input_width()
    tile_h = detector.input_height()
    all_objs = []
    raw_count = 0
    candidate_rows_all = []
    best_pink_ratio = 0.0
    best_red_ratio = 0.0

    origins = tile_origins(frame_w, frame_h, tile_w, tile_h)
    if ROUND_ROBIN_TILES and len(origins) > 1:
        selected_origins = []
        start = _tile_scan_index % len(origins)
        count = min(TILES_PER_FRAME, len(origins))
        for offset in range(count):
            selected_origins.append(origins[(start + offset) % len(origins)])
        _tile_scan_index = (start + count) % len(origins)
        _last_tile_info = "%d/%d" % (_tile_scan_index if _tile_scan_index > 0 else len(origins), len(origins))
    else:
        selected_origins = origins
        _last_tile_info = "all/%d" % len(origins)

    for tx, ty in selected_origins:
        tile = img if (tx == 0 and ty == 0 and frame_w == tile_w and frame_h == tile_h) else crop_tile(img, tx, ty, tile_w, tile_h)
        if tile is None:
            continue
        objs = detector.detect(tile, conf_th=ACTIVE_CONF_TH, iou_th=IOU_TH)
        raw_count += len(objs)
        candidates, pink_ratio, red_ratio, candidate_rows = filter_candidates(
            tile, objs, tile_w, tile_h, frame_id
        )
        if pink_ratio > best_pink_ratio:
            best_pink_ratio = pink_ratio
        if red_ratio > best_red_ratio:
            best_red_ratio = red_ratio
        for row in candidate_rows:
            raw_idx, obj, geometry_ok, color_ok, pink_row, red_row = row
            mapped = DetectedBox(tx + int(obj.x), ty + int(obj.y), int(obj.w), int(obj.h), obj.score)
            candidate_rows_all.append((raw_idx, mapped, geometry_ok, color_ok, pink_row, red_row))
        for obj in candidates:
            all_objs.append(DetectedBox(tx + int(obj.x), ty + int(obj.y), int(obj.w), int(obj.h), obj.score))

    remembered = remember_detections(all_objs) if ROUND_ROBIN_TILES else merge_overlaps(all_objs)
    return raw_count, remembered, best_pink_ratio, best_red_ratio, candidate_rows_all


print("YOLO SNAIL EGG DETECTOR BOOT")
gimbal = init_gimbal()
gimbal_tracker = init_gimbal_tracker(gimbal)
aim_relay = init_aim_relay()
aim_dot_detector = AimDotDetector()
aim_waypoint_planner = AimWaypointPlanner()
if aim_relay:
    atexit.register(aim_relay.close)
print(
    "CFG,PROFILE,%s,RUN_MODE,%d,FRAME,%dx%d,TILED,%d,RR,%d,TILES_PER_FRAME,%d,TTL,%d,DETECT_CONF,%.3f,MIN_MODEL,%.3f,STRONG_MODEL,%.3f,IOU,%.3f,MIN_PINK,%.4f"
    % (
        SPEED_PROFILE,
        RUN_MODE,
        FRAME_W,
        FRAME_H,
        1 if USE_TILED_INFERENCE else 0,
        1 if ROUND_ROBIN_TILES else 0,
        TILES_PER_FRAME,
        MEMORY_TTL_FRAMES,
        ACTIVE_CONF_TH,
        ACTIVE_MIN_MODEL_CONF,
        STRONG_MODEL_CONF,
        IOU_TH,
        MIN_PINK_RATIO,
    )
)
try:
    os.makedirs(DEBUG_DIR, exist_ok=True)
except Exception as e:
    print("DEBUG_DIR_ERROR,%s" % e)
try:
    if os.path.exists(HEADLESS_FLAG):
        ENABLE_DISPLAY = False
except Exception:
    pass
detector = nn.YOLOv8(model=MODEL, dual_buff=DUAL_BUFF)
print("INIT,DETECTOR_OK")
print("INIT,TILES,%d" % len(tile_origins(FRAME_W, FRAME_H, detector.input_width(), detector.input_height())))
cam = camera.Camera(
    FRAME_W,
    FRAME_H,
    detector.input_format(),
    buff_num=CAMERA_BUFF_NUM,
)
print("INIT,CAMERA_OK")
if ENABLE_DISPLAY:
    disp = display.Display()
    print("INIT,DISPLAY_OK")
else:
    disp = None
    print("INIT,DISPLAY_SKIP")
frame_id = 0
debug_saved = 0
manual_capture_saved = 0
overlay_stream_saved = 0
aim_detect_frames = 0
aim_missing_frames = AIM_RELAY_OFF_MISSING_FRAMES
print("INIT,LOOP_START")

while not app.need_exit():
    if RUN_MODE == 3 and frame_id < 3:
        print("TRACE,%d,READ_BEGIN" % frame_id)
    img = cam.read()
    dot_enabled = bool(
        closed_loop_aim_requested()
        and aim_relay is not None
        # Do not look for feedback while the two button pulses are still
        # changing the physical light state; pink image detail can otherwise
        # become the initial dot anchor before the real spot exists.
        and aim_relay.logical_on
        and aim_relay.phase == "IDLE"
    )
    aim_dot = aim_dot_detector.detect(img, dot_enabled)
    _last_aim_dot = aim_dot
    if frame_id % AIM_DOT_LOG_EVERY_N_FRAMES == 0:
        if aim_dot is None:
            print("DOT,NONE,ENABLED,%d" % int(dot_enabled))
        else:
            print(
                "DOT,%d,%d,SCORE,%.1f,FRESH,%d,MISSES,%d"
                % (
                    int(aim_dot.x),
                    int(aim_dot.y),
                    aim_dot.score,
                    int(aim_dot.fresh),
                    aim_dot.misses,
                )
            )
    if RUN_MODE == 3 and frame_id < 3:
        print("TRACE,%d,READ_OK" % frame_id)
        try:
            img.save("%s/trace_raw_%03d.jpg" % (DEBUG_DIR, frame_id))
            print("TRACE,%d,SAVE_OK" % frame_id)
        except Exception as e:
            print("TRACE,%d,SAVE_ERROR,%s" % (frame_id, e))
        print("TRACE,%d,DETECT_BEGIN" % frame_id)
    raw_count, candidates, best_pink_ratio, best_red_ratio, candidate_rows = detect_frame(detector, img, frame_id)
    if file_exists(DEBUG_CAPTURE_ONCE_FLAG):
        try:
            capture_path = "%s/once_%06d.jpg" % (DEBUG_DIR, frame_id)
            img.save(capture_path)
            os.remove(DEBUG_CAPTURE_ONCE_FLAG)
            print("MANUAL_CAPTURE_ONCE,%s" % capture_path)
        except Exception as e:
            print("MANUAL_CAPTURE_ONCE_ERROR,%s" % e)
    if (
        manual_capture_saved < MANUAL_CAPTURE_MAX_SAVES
        and file_exists(DEBUG_CAPTURE_FLAG)
        and frame_id % MANUAL_CAPTURE_EVERY_N_FRAMES == 0
    ):
        try:
            capture_path = "%s/manual_%06d.jpg" % (DEBUG_DIR, frame_id)
            img.save(capture_path)
            manual_capture_saved += 1
            print("MANUAL_CAPTURE,%s" % capture_path)
        except Exception as e:
            print("MANUAL_CAPTURE_ERROR,%s" % e)
    if RUN_MODE == 3 and frame_id < 3:
        print("TRACE,%d,DETECT_OK,%d" % (frame_id, raw_count))
    stable = update_tracks(candidates)
    targets = [] if frame_id < WARMUP_FRAMES else sort_targets(stable, FRAME_H)
    if file_exists(AIM_RELAY_NO_TARGET_TEST_FLAG):
        targets = []
    primary_obj = select_primary_target([item[0] for item in targets])
    primary_id = getattr(primary_obj, "aim_lock_id", getattr(primary_obj, "track_id", 0)) if primary_obj else 0
    fresh_aim_present = bool(primary_obj is not None and not getattr(primary_obj, "predicted", False))
    if fresh_aim_present:
        aim_detect_frames += 1
        aim_missing_frames = 0
    else:
        aim_detect_frames = 0
        aim_missing_frames += 1
    if aim_relay:
        relay_decision = aim_relay_decision(fresh_aim_present, aim_detect_frames, aim_missing_frames)
        if relay_decision is not None:
            aim_relay.request(relay_decision)
        aim_relay.update(pytime.time())
    aim_waypoint = aim_waypoint_planner.point(
        primary_obj,
        aim_dot,
        pytime.time(),
        bool(aim_relay is not None and aim_relay.logical_on and aim_scan_requested()),
    )
    control_status = "OFF"
    if gimbal_tracker is not None:
        control_status = gimbal_tracker.update(
            primary_obj,
            FRAME_W,
            FRAME_H,
            pytime.time(),
            aim_dot=aim_dot,
            waypoint=aim_waypoint,
        )
        if control_status.startswith("TRACK,") or control_status.startswith("PREDICT,"):
            print("AIM,%s" % control_status)
        elif frame_id % GIMBAL_LOG_EVERY_N_FRAMES == 0 and control_status != "HOLD,RATE":
            print("AIM,%s" % control_status)
    save_debug = (
        RUN_MODE == 3
        and debug_saved < DEBUG_MAX_SAVES
        and frame_id % DEBUG_SAVE_EVERY_N_FRAMES == 0
    )
    if save_debug:
        try:
            img.save("%s/raw_%03d.jpg" % (DEBUG_DIR, frame_id))
        except Exception as e:
            print("SAVE_RAW_ERROR,%d,%s" % (frame_id, e))

    fps_now = time.fps()
    img.draw_string(2, 2, "FPS %.1f" % fps_now, image.COLOR_GREEN)
    status_label = "WARM" if frame_id < WARMUP_FRAMES else "EGGS"
    img.draw_string(2, 18, "M%d RAW %d CAND %d %s %d LOCK %d" % (RUN_MODE, raw_count, len(candidates), status_label, len(targets), primary_id),
                    image.COLOR_GREEN if targets else image.COLOR_RED)
    img.draw_string(
        2,
        34,
        "PINK %.2f RED %.2f T %s" % (best_pink_ratio, best_red_ratio, _last_tile_info),
        image.COLOR_GREEN if candidates else image.COLOR_RED,
    )
    if gimbal_tracker is None:
        gimbal_label = "GIMBAL OFF"
    elif primary_obj is None:
        gimbal_label = "GIMBAL HOLD"
    elif getattr(primary_obj, "predicted", False):
        gimbal_label = "GIMBAL HOLD PRED"
    else:
        gimbal_label = "GIMBAL TRACK %d" % primary_id
    if aim_dot is not None:
        dot_label = "DOT" if aim_dot.fresh else "DOT HOLD"
    elif closed_loop_aim_requested():
        dot_label = "DOT LOST"
    else:
        dot_label = "OPEN LOOP"
    aim_label = "%s %s" % (aim_relay.label() if aim_relay else "AIM DISABLED", dot_label)
    img.draw_string(2, 50, "%s %s" % (gimbal_label, aim_label), image.COLOR_BLUE if gimbal_tracker else image.COLOR_RED)

    if aim_dot is not None:
        dot_x, dot_y = int(aim_dot.x), int(aim_dot.y)
        draw_cross(img, dot_x, dot_y, image.COLOR_RED if aim_dot.fresh else image.COLOR_YELLOW)
        img.draw_rect(dot_x - 5, dot_y - 5, 10, 10, color=image.COLOR_RED, thickness=1)
    if aim_waypoint is not None:
        waypoint_x, waypoint_y = int(aim_waypoint[0]), int(aim_waypoint[1])
        draw_cross(img, waypoint_x, waypoint_y, image.COLOR_BLUE)

    if RUN_MODE == 0:
        raw_targets = sort_targets([row[1] for row in candidate_rows], FRAME_H)
        for idx, item in enumerate(raw_targets):
            obj, cx, cy = item
            img.draw_rect(int(obj.x), int(obj.y), int(obj.w), int(obj.h), color=image.COLOR_YELLOW, thickness=1)
            draw_cross(img, cx, cy, image.COLOR_YELLOW)

    elif RUN_MODE == 1 or RUN_MODE == 3:
        raw_targets = sort_targets([row[1] for row in candidate_rows], FRAME_H)
        for idx, item in enumerate(raw_targets):
            obj, cx, cy = item
            img.draw_rect(int(obj.x), int(obj.y), int(obj.w), int(obj.h), color=image.COLOR_YELLOW, thickness=1)

    for idx, item in enumerate(targets):
        obj, cx, cy = item
        target_id = idx + 1
        is_primary = obj is primary_obj
        predicted = getattr(obj, "predicted", False)
        x = int(obj.x)
        y = int(obj.y)
        w = int(obj.w)
        h = int(obj.h)
        box_color = image.COLOR_YELLOW if predicted else image.COLOR_GREEN
        box_thickness = 3 if is_primary else 2
        img.draw_rect(x, y, w, h, color=box_color, thickness=box_thickness)
        draw_cross(img, cx, cy, image.COLOR_RED if not is_primary else image.COLOR_BLUE)
        label_y = y - 14 if y >= 14 else y
        img.draw_string(x, label_y, str(target_id), box_color)

        if frame_id % PRINT_EVERY_N_FRAMES == 0:
            print(
                "EGG,%d,%d,%d,%d,%d,%d,%d,%.3f,%.4f,%.4f"
                % (
                    target_id,
                    cx,
                    cy,
                    x,
                    y,
                    w,
                    h,
                    obj.score,
                    cx / FRAME_W,
                    cy / FRAME_H,
                )
            )

    if frame_id % STAT_EVERY_N_FRAMES == 0:
        print(
            "STAT,%d,FPS,%.1f,RAW,%d,CAND,%d,EGGS,%d,TILE,%s"
            % (frame_id, fps_now, raw_count, len(candidates), len(targets), _last_tile_info)
        )

    if RUN_MODE == 3 and frame_id % PRINT_EVERY_N_FRAMES == 0:
        print(
            "FRAME,%d,RAW,%d,CAND,%d,EGGS,%d,PINK,%.4f,RED,%.4f"
            % (frame_id, raw_count, len(candidates), len(targets), best_pink_ratio, best_red_ratio)
        )
        for row in candidate_rows:
            raw_idx, obj, geometry_ok, color_ok, pink_ratio, red_ratio = row
            print(
                "OBJ,%d,%d,%d,%d,%d,%d,%d,%.3f,%d,%d,%.4f,%.4f"
                % (
                    frame_id,
                    raw_idx,
                    int(obj.x),
                    int(obj.y),
                    int(obj.w),
                    int(obj.h),
                    int(obj.x + obj.w // 2),
                    obj.score,
                    1 if geometry_ok else 0,
                    1 if color_ok else 0,
                    pink_ratio,
                    red_ratio,
                )
            )

    if save_debug:
        try:
            img.save("%s/overlay_%03d.jpg" % (DEBUG_DIR, frame_id))
            debug_saved += 1
            print("SNAPSHOT,%d,%s" % (frame_id, DEBUG_DIR))
        except Exception as e:
            print("SAVE_OVERLAY_ERROR,%d,%s" % (frame_id, e))

    if (
        overlay_stream_saved < DEBUG_OVERLAY_STREAM_MAX_SAVES
        and file_exists(DEBUG_OVERLAY_STREAM_FLAG)
        and frame_id % DEBUG_OVERLAY_STREAM_EVERY_N_FRAMES == 0
    ):
        try:
            overlay_path = "%s/demo_overlay_%06d.jpg" % (DEBUG_DIR, frame_id)
            img.save(overlay_path)
            overlay_stream_saved += 1
            if overlay_stream_saved >= DEBUG_OVERLAY_STREAM_MAX_SAVES:
                os.remove(DEBUG_OVERLAY_STREAM_FLAG)
            print("DEMO_OVERLAY,%s" % overlay_path)
        except Exception as e:
            print("DEMO_OVERLAY_ERROR,%s" % e)

    if disp:
        disp.show(img)
    frame_id += 1

if gimbal:
    gimbal.close()
if aim_relay:
    aim_relay.close()
print("YOLO SNAIL EGG DETECTOR STOP")
