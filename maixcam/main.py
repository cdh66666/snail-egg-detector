import atexit
import os
import threading
import time as pytime

from maix import camera, display, image, nn, app, time, pwm, pinmap, gpio
try:
    from maix import http
except Exception:
    http = None
try:
    from web_control import WebControl
except Exception:
    WebControl = None


# Copy this file to MaixVision as main.py after converting the ONNX model to MUD.
# Put the MUD and model file under /root/models/ on the MaixCam.

MODEL = "/root/models/snail_eggs_yolo11n_320x224_v26.mud"
USE_FAST_MODEL = True
FAST_MODEL_IS_YOLO11 = True
# Run inference at the maintenance floor, but only expose high-confidence
# detections for initial selection. A selected target may continue at the lower
# threshold only when the measurement agrees with its predicted trajectory.
CONF_TH = 0.10
IOU_TH = 0.35
MAX_TARGETS = 32
DISCOVERY_MODEL_CONF = 0.35
LOCK_MAINTAIN_MODEL_CONF = 0.10
MIN_MODEL_CONF = LOCK_MAINTAIN_MODEL_CONF
STRONG_MODEL_CONF = 0.35 if USE_FAST_MODEL else 0.38
LOW_CONF_MIN_PINK_RATIO = 0.035
LOW_CONF_MIN_AREA = 18

# Red aiming-light relay only. The destructive laser remains manual.
AIM_RELAY_PIN = "A15"
AIM_RELAY_GPIO = "GPIOA15"
AIM_RELAY_DISABLE_FLAG = "/root/snail_egg/disable_aim_relay"
AIM_RELAY_NO_TARGET_TEST_FLAG = "/root/snail_egg/test_no_target_safety"
AIM_RELAY_STARTUP_TEST_DONE_FLAG = "/root/snail_egg/aim_relay_startup_test_done"
AIM_RELAY_ON_STABLE_FRAMES = 3
# Keep the harmless red aiming light on while any fresh valid egg detection is
# present anywhere in the camera view. Brief detector gaps are tolerated, but
# two seconds without a fresh valid egg turns it off.
AIM_RELAY_OFF_MISSING_S = 2.0
AIM_RELAY_PULSE_S = 0.20
AIM_RELAY_STARTUP_TEST_ON_S = 1.0
AIM_RELAY_STARTUP_TEST_CYCLES = 1

# Closed-loop aiming uses the visible low-power red dot as measured feedback.
# It is runtime opt-in so a missing/misconfigured dot can never move the gimbal.
CLOSED_LOOP_AIM_FLAG = "/root/snail_egg/enable_closed_loop_aim"
AIM_DOT_LAB_THRESHOLDS = [[20, 100, 35, 127, -20, 127]]
AIM_DOT_MIN_PIXELS = 1
AIM_DOT_MAX_PIXELS = 420
AIM_DOT_MIN_SIDE = 1
AIM_DOT_MAX_SIDE = 32
AIM_DOT_MIN_RED = 120
# The live GC4653 test measured the visible spot at R=176..181 with red
# dominance 37..46 and ratio 0.20..0.26 after display/camera white balance.
# Position, local contrast, size and jump gates provide the stronger
# discrimination from broad pink egg regions.
AIM_DOT_MIN_DOMINANCE = 45
AIM_DOT_MIN_DOMINANCE_RATIO = 0.30
AIM_DOT_RGB_MIN_DOMINANCE = 30
AIM_DOT_RGB_MIN_DOMINANCE_RATIO = 0.17
AIM_DOT_MIN_LOCAL_CONTRAST = 8
AIM_DOT_RGB_STRIDE = 7
AIM_DOT_RGB_REFRESH_FRAMES = 1
AIM_DOT_RGB_REACQUIRE_FRAMES = 3
# The red spot is rigidly attached to the camera. Search more often while it
# is visible and less often while lost; this keeps dot search from starving
# YOLO while retaining prompt reacquisition.
AIM_DOT_LOOP_EVERY_N_FRAMES = 10
AIM_DOT_TRACK_LOOP_EVERY_N_FRAMES = 2
AIM_DOT_EXPECTED_RADIUS_PX = 70
AIM_DOT_MAX_JUMP_PX = 65
AIM_DOT_FILTER_ALPHA = 0.55
AIM_DOT_STALE_FRAMES = 8
AIM_DOT_FIXED_FALLBACK_AFTER_MISSES = 6
AIM_DOT_LOG_EVERY_N_FRAMES = 10
AIM_DOT_CONTROL_TOLERANCE_PX = 9
AIM_SCAN_SETTLE_FRAMES = 2
AIM_SCAN_DWELL_S = 0.12
SELECTED_LOSS_CONFIRM_DETECTIONS = 8
SELECTED_GOAL_FILTER_ALPHA = 0.30
SELECTED_LOCK_CONFIRM_DETECTIONS = 4
SELECTED_LOCK_TIMEOUT_DETECTIONS = 8
SELECTED_LOCK_MAX_CENTER_SHIFT_BOXES = 0.45
SELECTED_LOCK_MAX_SIZE_RATIO = 1.65
# Measured on the installed camera/gimbal pair with +/-5 degree moves.
# The nominal lens FOV over-commanded the servos and left the target short.
# These equivalent FOV values are used as calibrated pixels-per-degree.
CAMERA_H_FOV_DEG = 55.7
CAMERA_V_FOV_DEG = 36.5

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
SCREEN_TEST_CONF_TH = 0.05
GIMBAL_FREQ = 50
GIMBAL_CENTER_DEG = 90
GIMBAL_PAN_MIN_DEG = 60
GIMBAL_PAN_MAX_DEG = 120
GIMBAL_TILT_MIN_DEG = 80
GIMBAL_TILT_MAX_DEG = 120
GIMBAL_CENTER_SETTLE_S = 0.35
GIMBAL_PAN_PWM_ID = 6
GIMBAL_TILT_PWM_ID = 7
# Slow non-PID tracking profile. Image error is smoothed and converted directly
# to a tiny bounded angle step. Reversals must brake through zero first.
GIMBAL_CONTROL_HZ = 30.0
GIMBAL_DEADZONE_X = 0.015
GIMBAL_DEADZONE_Y = 0.018
GIMBAL_SLOW_MAX_STEP_DEG = 0.35
GIMBAL_SLOW_STEP_CHANGE_DEG = 0.060
GIMBAL_SLOW_FULL_SPEED_ERROR = 0.08
GIMBAL_SLOW_ERROR_FILTER_TAU_S = 0.04
# The vision controller may run slower, but physical PWM commands are
# interpolated independently at 50 Hz with bounded speed, acceleration and
# jerk. This removes the visible 12 Hz staircase without adding PID overshoot.
GIMBAL_TRAJECTORY_HZ = 50.0
GIMBAL_TRAJECTORY_MAX_SPEED_DEG_S = 3.0
GIMBAL_TRAJECTORY_MAX_ACCEL_DEG_S2 = 8.0
GIMBAL_TRAJECTORY_MAX_JERK_DEG_S3 = 40.0
GIMBAL_TRAJECTORY_POSITION_EPS_DEG = 0.008
GIMBAL_TRAJECTORY_SPEED_EPS_DEG_S = 0.04
# Open-loop servos do not report shaft angle. Keep PWM command and estimated
# physical angle separate so image-motion compensation does not pretend that a
# newly written duty cycle has already moved the camera. These conservative
# response values must be refined from supervised real-device telemetry.
GIMBAL_SERVO_ESTIMATE_TAU_S = 0.25
GIMBAL_SERVO_ESTIMATE_MAX_SPEED_DEG_S = 90.0
# Avoid restarting a rest-to-rest quintic segment on every tiny 30 Hz vision
# update. Frequent replans look fine on a static move but lag badly on a curve.
GIMBAL_TARGET_REPLAN_EPS_DEG = 0.20
GIMBAL_RECENTER_NO_TARGET_S = 2.0
GIMBAL_RECENTER_RATE_DEG_S = 3.0
AUTO_CYCLE_FLAG = "/root/snail_egg/enable_auto_cycle"
AUTO_CYCLE_HOLD_S = 1.25
AUTO_CYCLE_TOLERANCE_PX = 12.0
GIMBAL_PAN_SIGN = -1.0
GIMBAL_TILT_SIGN = -1.0
GIMBAL_MIN_STABLE = 3
GIMBAL_MIN_SCORE = 0.28
GIMBAL_LOG_EVERY_N_FRAMES = 10
AIM_LOG_EVERY_N_FRAMES = 5

# 0 = raw YOLO debug: draw every model detection, no color filtering.
# 1 = color debug: raw boxes are yellow, pink-passed boxes are green.
# 2 = final: only draw pink-passed targets with IDs.
# 3 = auto tune: color debug + save frames + print every raw/candidate stat.
RUN_MODE = 2
SHOW_OVERLAY_TEXT = False

# Runtime speed profiles use one inference per current camera frame. No stale
# round-robin tiles are used for actuator aiming.
SPEED_PROFILE = "full_frame"

# Use the compact rectangular YOLO11 INT8 model exclusively on MaixCAM.
FRAME_W = 320 if USE_FAST_MODEL else 640
FRAME_H = 224 if FAST_MODEL_IS_YOLO11 else (256 if USE_FAST_MODEL else 480)
FRAME_SCALE = FRAME_W / 640.0
# The installed camera/gimbal pair is calibrated above. With the laser
# emitter 10 mm right and 60 mm below the camera, parallel-axis parallax at
# 1.0 m is approximately (+3.7, +30.2) pixels. The rounded point below is the
# fixed aim reference. A raw device capture measured the visible red spot at
# approximately (154, 121) in the 320x224 control frame.
LASER_WORK_DISTANCE_MM = 1000
LASER_OFFSET_RIGHT_MM = 10
LASER_OFFSET_DOWN_MM = 60
FIXED_AIM_X = int(round(308 * FRAME_SCALE))
FIXED_AIM_Y = int(round(259 * FRAME_H / 480.0))
# The desktop monitor used for gimbal tuning is about 0.35 m away. A captured
# frame measured the visible aiming dot at y=326 there. Keep this calibration
# separate so the field profile remains correct at the required 1.0 m range.
SCREEN_TEST_AIM_X = FIXED_AIM_X
SCREEN_TEST_AIM_Y = FIXED_AIM_Y
SCREEN_TEST_DISTANCE_MM = 350
# A visible aiming dot can itself produce a small low-confidence YOLO box.
# Reject only marker-sized boxes very close to the calibrated aim point.
AIM_MARKER_REJECT_RADIUS_PX = max(16, int(round(32 * FRAME_SCALE)))
AIM_MARKER_REJECT_MAX_W = max(26, int(round(52 * FRAME_SCALE)))
AIM_MARKER_REJECT_MAX_H = max(28, int(round(56 * FRAME_H / 480.0)))
if USE_FAST_MODEL:
    AIM_DOT_MAX_PIXELS = 120
    AIM_DOT_MAX_SIDE = 18
    AIM_DOT_RGB_STRIDE = 4
    AIM_DOT_RGB_REFRESH_FRAMES = 3
    AIM_DOT_EXPECTED_RADIUS_PX = 35
    AIM_DOT_MAX_JUMP_PX = 22
    AIM_DOT_CONTROL_TOLERANCE_PX = 5
# Keep two capture buffers. Three buffers plus YOLO dual buffering and the
# physical display can exhaust the MaixCAM media pool during Display() init;
# two still prevents the camera read from blocking behind inference.
# MaixCAM's CSI/TPU path is more stable with one capture buffer when the
# compact YOLO11 model is used. Extra buffers increased latency and caused the
# synchronous detector call to wait behind stale frames on the real device.
CAMERA_BUFF_NUM = 1
CAMERA_TARGET_FPS = 60
WEB_STREAM_PORT = 8001
WEB_STREAM_MAX_CLIENTS = 4
WEB_STREAM_EVERY_N_FRAMES = 1
USE_TILED_INFERENCE = False
TILE_OVERLAP = 160
MERGE_IOU = 0.45
ROUND_ROBIN_TILES = True
TILES_PER_FRAME = 3
MEMORY_TTL_FRAMES = 36

# Bring-up defaults: show what the model sees first. Keep laser disconnected.
# After field tuning, raise CONF_TH, turn ENABLE_COLOR_GATE back on, and require
# stable frames before sending targets to any actuator.
MIN_BOX_AREA = 8 if USE_FAST_MODEL else 18
# Reject dot-like detections that are too small to be an actionable egg mass.
# A thin mass is still accepted when either side exceeds this limit.
MAX_TINY_BOX_SIDE = 6 if USE_FAST_MODEL else 11
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
# Color is a secondary sanity check after YOLO, not the detector. Keep the
# sample budget small on MaixCAM: Python get_pixel() is much more expensive
# than the NPU result and was dominating the measured detection stage.
STRONG_COLOR_CHECKS = 2
LOW_CONF_COLOR_CHECKS = 4
REQUIRE_STABLE_FRAMES = 2
# Keep a short prediction window when the detector misses a frame. At about
# 19 FPS this is roughly 0.3 seconds, long enough to bridge transient misses.
TRACK_MAX_MISSES = 60
# At the current 18-22 Hz loop and every-other-frame YOLO cadence, forty
# prediction frames bridge about two seconds of motion-induced detector loss.
# Prediction velocity decays on every miss so stale boxes settle instead of
# drifting across the view.
TRACK_PREDICT_MAX_MISSES = 40
LOCK_REACQUIRE_RADIUS_PX = 38 if USE_FAST_MODEL else 75
LOCK_REACQUIRE_MAX_COST = 60.0
# The deployed 640x480 pipeline runs at about 5 FPS. Release a target after
# roughly five seconds of failed reacquisition so another visible egg mass can
# take over, while still bridging ordinary one- or two-frame detector misses.
LOCK_RELEASE_MISSING_FRAMES = 25
# Switch to a fresh visible candidate after a short unmatched-lock window.
LOCK_SWITCH_MISSING_FRAMES = 6
TRACK_MATCH_DISTANCE_SCALE = 1.35
TRACK_MATCH_MIN_DISTANCE_PX = 45.0 if USE_FAST_MODEL else 90.0
TRACK_ASSOC_MAX_COST = 1.45
TRACK_LOCKED_LOW_ASSOC_MAX_COST = 0.92
TRACK_LOCKED_STRONG_ASSOC_MAX_COST = 0.72
TRACK_LOCKED_ASSOC_AMBIGUITY_MARGIN = 0.18
TRACK_ASSOC_MAX_SIZE_RATIO = 1.9
TRACK_KF_PROCESS_NOISE = 2.0
TRACK_KF_MEASUREMENT_NOISE = 16.0
TRACK_MAX_VELOCITY_PX = 27.0 if USE_FAST_MODEL else 54.0
# Track velocity is stored in pixels per nominal camera frame. Folding the
# measured loop interval into prediction prevents a slow inference or a burst
# of web encoding work from looking like several identical camera frames.
TRACK_REFERENCE_FPS = 20.0
TRACK_MIN_DT_FRAMES = 0.35
TRACK_MAX_DT_FRAMES = 2.50
GIMBAL_PREDICT_MAX_MISSES = 12
GIMBAL_COAST_MAX_MISSES = 2
# 0 means automatically lock the first stable track; set a positive ID to
# lock a specific track later when the gimbal controller is connected.
LOCK_TARGET_ID = 0
PRIMARY_TARGET_POLICY = "robust_center"
PRIMARY_COLUMN_TOLERANCE_PX = 20 if USE_FAST_MODEL else 40
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

# MaixPy's official dual-buffer path overlaps CPU preprocessing and NPU work.
# Keep it enabled; with camera buff_num=1 this is the lowest-latency supported
# combination on the installed MaixCAM runtime.
DUAL_BUFF = True
# YOLO inference is the expensive stage. The tracker/control/display loop runs
# on every camera frame; YOLO refreshes the tracks every second frame and the
# Kalman track bridges the intervening frame. This keeps a fresh detector rate
# near 3 Hz on the installed runtime without freezing the web stream.
DETECT_EVERY_N_FRAMES = 2

# Per-target serial logging is useful during a lab trace but blocks the
# MaixVision UART in normal operation. Structured STAT/PROF telemetry remains.
PRINT_EVERY_N_FRAMES = 0
STAT_EVERY_N_FRAMES = 30
PROFILE_EVERY_N_FRAMES = 120
DEBUG_DIR = "/root/snail_egg/debug"
DEBUG_CAPTURE_FLAG = "/root/snail_egg/capture_frames"
DEBUG_CAPTURE_ONCE_FLAG = "/root/snail_egg/capture_once"
DEBUG_OVERLAY_STREAM_FLAG = "/root/snail_egg/capture_overlay_stream"
DEBUG_OVERLAY_STREAM_EVERY_N_FRAMES = 3
DEBUG_OVERLAY_STREAM_MAX_SAVES = 120
TELEMETRY_FLAG = "/root/snail_egg/enable_telemetry_file"
TELEMETRY_PATH = "/root/snail_egg/telemetry.csv"
DEBUG_SAVE_EVERY_N_FRAMES = 15
DEBUG_MAX_SAVES = 6
MANUAL_CAPTURE_EVERY_N_FRAMES = 60
MANUAL_CAPTURE_MAX_SAVES = 4
WEB_CONTROL_PORT = 8000
WEB_CONTROL_POLL_EVERY_N_FRAMES = 2
WEB_CONTROL_DISABLE_FLAG = "/root/snail_egg/disable_web_control"
RUNTIME_FLAG_POLL_EVERY_N_FRAMES = 15
# Default to visual output for MaixVision/device use. The VSCode SSH helper
# creates /root/snail_egg/headless before running so automated tests do not
# block on display.Display().
ENABLE_DISPLAY = True
HEADLESS_FLAG = "/root/snail_egg/headless"
# Text is drawn on every output frame. Skipping text redraws makes the labels
# visibly blink because each camera frame starts from a clean image buffer.
OVERLAY_TEXT_EVERY_N_FRAMES = 1
_overlay_text_enabled = True
_web_overlay_enabled = False
_tracks = []
_next_track_id = 1
_locked_track_id = 0
_locked_last_center = None
_locked_last_size = None
_locked_missing_frames = 0
_primary_center = None
_manual_lock_active = False
_track_last_pan_offset = None
_track_last_tilt_offset = None
_pending_detection_pan_offset = None
_pending_detection_tilt_offset = None
_last_aim_dot = None
_previous_aim_dot = None
_auto_cycle_track_id = 0
_auto_cycle_since = 0.0
_auto_cycle_index = -1


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


def quintic_trajectory_duration(distance, initial_velocity=0.0, initial_acceleration=0.0):
    """Choose a rest-to-rest duration that respects speed and acceleration."""
    distance = abs(float(distance))
    if distance <= GIMBAL_TRAJECTORY_POSITION_EPS_DEG:
        return 0.0
    speed_time = 1.875 * distance / GIMBAL_TRAJECTORY_MAX_SPEED_DEG_S
    acceleration_time = pow(
        5.8 * distance / GIMBAL_TRAJECTORY_MAX_ACCEL_DEG_S2,
        0.5,
    )
    jerk_time = pow(
        60.0 * distance / GIMBAL_TRAJECTORY_MAX_JERK_DEG_S3,
        1.0 / 3.0,
    )
    state_time = max(
        2.0 * abs(initial_velocity) / GIMBAL_TRAJECTORY_MAX_ACCEL_DEG_S2,
        pow(2.0 * abs(initial_acceleration) / GIMBAL_TRAJECTORY_MAX_JERK_DEG_S3, 0.5),
    )
    return max(0.45, speed_time, acceleration_time, jerk_time, state_time)


def quintic_trajectory_coefficients(position, velocity, acceleration, target, duration):
    """Build a quintic segment ending with zero velocity and acceleration."""
    duration = max(0.001, float(duration))
    delta = float(target) - float(position)
    t2 = duration * duration
    t3 = t2 * duration
    t4 = t3 * duration
    t5 = t4 * duration
    a0 = float(position)
    a1 = float(velocity)
    a2 = 0.5 * float(acceleration)
    a3 = (20.0 * delta - 12.0 * velocity * duration - 3.0 * acceleration * t2) / (2.0 * t3)
    a4 = (-30.0 * delta + 16.0 * velocity * duration + 3.0 * acceleration * t2) / (2.0 * t4)
    a5 = (12.0 * delta - 6.0 * velocity * duration - acceleration * t2) / (2.0 * t5)
    return (a0, a1, a2, a3, a4, a5)


def quintic_trajectory_sample(coefficients, elapsed, duration, target):
    if elapsed >= duration:
        return float(target), 0.0, 0.0
    t = max(0.0, float(elapsed))
    a0, a1, a2, a3, a4, a5 = coefficients
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    position = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
    velocity = a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3 + 5.0 * a5 * t4
    acceleration = 2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t2 + 20.0 * a5 * t3
    return position, velocity, acceleration


def safe_sleep(seconds):
    try:
        pytime.sleep(seconds)
    except Exception:
        pass


class LatestJpegStreamer:
    """Publish only the newest JPEG so a slow client cannot backlog vision."""

    def __init__(self, streamer):
        self.streamer = streamer
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._latest = None
        self._sequence = 0
        self._stop = False
        self._closed = False
        self.fps = 0.0
        self.write_ms = 0.0
        self._window_start = pytime.time()
        self._window_frames = 0
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def submit(self, jpeg):
        with self._lock:
            self._latest = jpeg
            self._sequence += 1
        self._ready.set()

    def _writer_loop(self):
        clock = getattr(pytime, "monotonic", pytime.time)
        sent_sequence = -1
        while not self._stop:
            self._ready.wait(0.2)
            self._ready.clear()
            with self._lock:
                jpeg = self._latest
                sequence = self._sequence
            if jpeg is None or sequence == sent_sequence:
                continue
            started = clock()
            try:
                self.streamer.write(jpeg)
                sent_sequence = sequence
                self.write_ms = (clock() - started) * 1000.0
                self._window_frames += 1
                elapsed = pytime.time() - self._window_start
                if elapsed >= 1.0:
                    self.fps = self._window_frames / max(0.001, elapsed)
                    self._window_frames = 0
                    self._window_start = pytime.time()
            except Exception as e:
                print("WEB_STREAM_WRITE_ERROR,%s" % e)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop = True
        self._ready.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.streamer.stop()
        except Exception:
            pass


def file_exists(path):
    try:
        return os.path.exists(path)
    except Exception:
        return False


def write_telemetry(line):
    if not file_exists(TELEMETRY_FLAG):
        return
    try:
        with open(TELEMETRY_PATH, "a") as telemetry_file:
            telemetry_file.write(line + "\n")
    except Exception as e:
        print("TELEMETRY_ERROR,%s" % e)


def closed_loop_aim_requested():
    return file_exists(CLOSED_LOOP_AIM_FLAG)


def aim_relay_decision(fresh_target, detected_frames, missing_frames, missing_seconds=0.0):
    """Return True/False for a relay transition, or None to keep its state."""
    if missing_seconds >= AIM_RELAY_OFF_MISSING_S:
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
        self.command_pan_offset = 0.0
        self.command_tilt_offset = 0.0
        self.command_count = 0
        pinmap.set_pin_function("A19", "PWM7")
        pinmap.set_pin_function("A18", "PWM6")
        center_duty = servo_duty_from_angle(GIMBAL_CENTER_DEG)
        self.pan = pwm.PWM(GIMBAL_PAN_PWM_ID, freq=GIMBAL_FREQ, duty=center_duty, enable=True)
        self.tilt = pwm.PWM(GIMBAL_TILT_PWM_ID, freq=GIMBAL_FREQ, duty=center_duty, enable=True)
        self._lock = threading.Lock()
        self._pending = (0.0, 0.0)
        self._applied = (0.0, 0.0)
        self._velocity = (0.0, 0.0)
        self._acceleration = (0.0, 0.0)
        self._trajectory_target = (0.0, 0.0)
        self._trajectory_start = 0.0
        self._trajectory_duration = 0.0
        self._trajectory_coefficients = None
        self.motion_phase = "hold"
        self._stop = False
        self._closed = False
        self._estimate_last = 0.0
        safe_sleep(GIMBAL_CENTER_SETTLE_S)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _apply_offsets(self, pan_offset, tilt_offset):
        pan_angle = max(GIMBAL_PAN_MIN_DEG, min(GIMBAL_PAN_MAX_DEG, GIMBAL_CENTER_DEG + pan_offset))
        tilt_angle = max(GIMBAL_TILT_MIN_DEG, min(GIMBAL_TILT_MAX_DEG, GIMBAL_CENTER_DEG + tilt_offset))
        self.pan.duty(servo_duty_from_angle(pan_angle))
        self.tilt.duty(servo_duty_from_angle(tilt_angle))
        self._applied = (pan_offset, tilt_offset)
        self.command_pan_offset = pan_offset
        self.command_tilt_offset = tilt_offset

    def _update_estimated_offsets(self, now):
        dt = 1.0 / GIMBAL_TRAJECTORY_HZ if self._estimate_last <= 0.0 else max(0.001, now - self._estimate_last)
        self._estimate_last = now
        alpha = 1.0 - pow(2.718281828, -dt / max(0.01, GIMBAL_SERVO_ESTIMATE_TAU_S))
        max_step = GIMBAL_SERVO_ESTIMATE_MAX_SPEED_DEG_S * dt
        pan_delta = max(-max_step, min(max_step, (self._applied[0] - self.pan_offset) * alpha))
        tilt_delta = max(-max_step, min(max_step, (self._applied[1] - self.tilt_offset) * alpha))
        self.pan_offset = clamp_pan_offset(self.pan_offset + pan_delta)
        self.tilt_offset = clamp_tilt_offset(self.tilt_offset + tilt_delta)

    def _writer_loop(self):
        clock = getattr(pytime, "monotonic", pytime.time)
        while not self._stop:
            tick_start = clock()
            with self._lock:
                pending = self._pending
            target_changed = max(
                abs(pending[0] - self._trajectory_target[0]),
                abs(pending[1] - self._trajectory_target[1]),
            ) >= GIMBAL_TARGET_REPLAN_EPS_DEG
            if self._trajectory_coefficients is None:
                target_changed = max(
                    abs(pending[0] - self._applied[0]),
                    abs(pending[1] - self._applied[1]),
                ) > GIMBAL_TRAJECTORY_POSITION_EPS_DEG
            if target_changed:
                pan_duration = quintic_trajectory_duration(
                    pending[0] - self._applied[0], self._velocity[0], self._acceleration[0]
                )
                tilt_duration = quintic_trajectory_duration(
                    pending[1] - self._applied[1], self._velocity[1], self._acceleration[1]
                )
                self._trajectory_duration = max(pan_duration, tilt_duration)
                self._trajectory_start = tick_start
                self._trajectory_target = pending
                self.motion_phase = "sync"
                if self._trajectory_duration > 0.0:
                    self._trajectory_coefficients = (
                        quintic_trajectory_coefficients(
                            self._applied[0], self._velocity[0], self._acceleration[0],
                            pending[0], self._trajectory_duration,
                        ),
                        quintic_trajectory_coefficients(
                            self._applied[1], self._velocity[1], self._acceleration[1],
                            pending[1], self._trajectory_duration,
                        ),
                    )
                else:
                    self._trajectory_coefficients = None
            if self._trajectory_coefficients is not None:
                try:
                    elapsed = tick_start - self._trajectory_start
                    next_pan, pan_velocity, pan_acceleration = quintic_trajectory_sample(
                        self._trajectory_coefficients[0], elapsed,
                        self._trajectory_duration, self._trajectory_target[0],
                    )
                    next_tilt, tilt_velocity, tilt_acceleration = quintic_trajectory_sample(
                        self._trajectory_coefficients[1], elapsed,
                        self._trajectory_duration, self._trajectory_target[1],
                    )
                    self._velocity = (pan_velocity, tilt_velocity)
                    self._acceleration = (pan_acceleration, tilt_acceleration)
                    self._apply_offsets(next_pan, next_tilt)
                    if elapsed >= self._trajectory_duration:
                        self._trajectory_coefficients = None
                        self._velocity = (0.0, 0.0)
                        self._acceleration = (0.0, 0.0)
                        self.motion_phase = "hold"
                except Exception as e:
                    print("GIMBAL_PWM_ERROR,%s" % e)
            else:
                self._velocity = (0.0, 0.0)
                self._acceleration = (0.0, 0.0)
            self._update_estimated_offsets(clock())
            tick_elapsed = clock() - tick_start
            safe_sleep(max(0.001, 1.0 / GIMBAL_TRAJECTORY_HZ - tick_elapsed))

    def set_offsets(self, pan_offset, tilt_offset, source="CMD"):
        target_pan = clamp_pan_offset(pan_offset)
        target_tilt = clamp_tilt_offset(tilt_offset)
        pan_angle = max(GIMBAL_PAN_MIN_DEG, min(GIMBAL_PAN_MAX_DEG, GIMBAL_CENTER_DEG + target_pan))
        tilt_angle = max(GIMBAL_TILT_MIN_DEG, min(GIMBAL_TILT_MAX_DEG, GIMBAL_CENTER_DEG + target_tilt))
        with self._lock:
            self._pending = (target_pan, target_tilt)
        self.command_count += 1
        if source not in ("TRACK", "RECENTER") or self.command_count % GIMBAL_LOG_EVERY_N_FRAMES == 0:
            print(
                "GIMBAL,%s,PAN_OFFSET,%d,TILT_OFFSET,%d,PAN_ANGLE,%d,TILT_ANGLE,%d"
                % (source, target_pan, target_tilt, pan_angle, tilt_angle)
            )

    def hold(self):
        """Freeze the active trajectory at the latest applied PWM position."""
        with self._lock:
            self._pending = self._applied
            self._trajectory_target = self._applied
            self._trajectory_coefficients = None
            self._trajectory_duration = 0.0
            self._velocity = (0.0, 0.0)
            self._acceleration = (0.0, 0.0)
            self.motion_phase = "hold"

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Shutdown must never start an unsolicited sweep. Freeze at the latest
        # applied position, stop the worker, then disable both PWM outputs.
        self.hold()
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
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

    def hold(self):
        return None

    def close(self):
        print("GIMBAL,DRY_RUN,CLOSE")


class GimbalAxisController:
    """Slow, bounded image-error follower with no PID state."""

    def __init__(self, sign):
        self.sign = float(sign)
        self.filtered_error = 0.0
        self.command_step = 0.0
        self.initialized = False

    def reset(self):
        self.filtered_error = 0.0
        self.command_step = 0.0
        self.initialized = False

    def hold(self):
        self.filtered_error = 0.0
        self.command_step = 0.0
        self.initialized = False

    def step(self, error, dt):
        if dt <= 0.0:
            dt = 1.0 / GIMBAL_CONTROL_HZ
        alpha = 1.0 - pow(2.718281828, -dt / max(0.01, GIMBAL_SLOW_ERROR_FILTER_TAU_S))
        if not self.initialized:
            self.filtered_error = error
            self.initialized = True
        else:
            self.filtered_error += alpha * (error - self.filtered_error)

        magnitude = abs(self.filtered_error)
        if error == 0.0 or magnitude < 0.001:
            desired_step = 0.0
        else:
            speed_ratio = min(1.0, magnitude / max(0.01, GIMBAL_SLOW_FULL_SPEED_ERROR))
            desired_step = self.sign * (1.0 if self.filtered_error > 0.0 else -1.0)
            desired_step *= GIMBAL_SLOW_MAX_STEP_DEG * speed_ratio

        # Never reverse in one update. First reduce the old command to zero,
        # then start moving in the other direction on a later update.
        max_change = GIMBAL_SLOW_STEP_CHANGE_DEG
        if self.command_step * desired_step < 0.0:
            desired_step = 0.0
        delta = max(-max_change, min(max_change, desired_step - self.command_step))
        self.command_step += delta
        if desired_step == 0.0 and abs(self.command_step) <= max_change:
            self.command_step = 0.0
        return self.command_step


class GimbalTracker:
    """Safety-gated target tracker layered on top of the gimbal PWM."""

    def __init__(self, gimbal):
        self.gimbal = gimbal
        self.pan = GimbalAxisController(GIMBAL_PAN_SIGN)
        self.tilt = GimbalAxisController(GIMBAL_TILT_SIGN)
        self.pan_offset = 0.0
        self.tilt_offset = 0.0
        self.last_update = 0.0
        self.hold_frames = 0
        self.closed_loop = closed_loop_aim_requested()

    def reset(self):
        self.pan.reset()
        self.tilt.reset()
        self.hold_frames = 0

    def sync_to_offsets(self, pan_offset, tilt_offset):
        """Adopt the current physical command before auto control resumes."""
        self.pan_offset = clamp_pan_offset(pan_offset)
        self.tilt_offset = clamp_tilt_offset(tilt_offset)
        self.reset()
        self.last_update = 0.0

    def hold(self):
        self.pan.hold()
        self.tilt.hold()
        freeze = getattr(self.gimbal, "hold", None)
        if freeze is not None:
            freeze()
        self.hold_frames += 1

    @staticmethod
    def _approach_zero(value, amount):
        if value > amount:
            return value - amount
        if value < -amount:
            return value + amount
        return 0.0

    def recenter(self, now):
        """Return toward the optical center after a sustained empty view."""
        interval = 1.0 / GIMBAL_CONTROL_HZ
        if self.last_update > 0.0 and now - self.last_update < interval:
            return "HOLD,RATE"
        dt = interval if self.last_update <= 0.0 else max(0.001, now - self.last_update)
        self.last_update = now
        step = GIMBAL_RECENTER_RATE_DEG_S * dt
        self.pan.reset()
        self.tilt.reset()
        self.pan_offset = self._approach_zero(self.pan_offset, step)
        self.tilt_offset = self._approach_zero(self.tilt_offset, step)
        self.gimbal.set_offsets(self.pan_offset, self.tilt_offset, "RECENTER")
        return "RECENTER,PAN,%.2f,TILT,%.2f" % (self.pan_offset, self.tilt_offset)

    @staticmethod
    def soften_deadzone(error, deadzone):
        """Remove sensor jitter without a discontinuous start at the boundary."""
        magnitude = abs(error)
        if magnitude <= deadzone:
            return 0.0
        scaled = (magnitude - deadzone) / max(0.001, 1.0 - deadzone)
        return scaled if error > 0.0 else -scaled

    def update(self, target, frame_w, frame_h, now, aim_dot=None, waypoint=None, use_closed_loop=None):
        if target is None:
            self.hold()
            return "HOLD,NO_FRESH_TARGET"
        predicted = getattr(target, "predicted", False)
        if predicted and getattr(target, "misses", GIMBAL_PREDICT_MAX_MISSES + 1) > GIMBAL_PREDICT_MAX_MISSES:
            self.reset()
            return "HOLD,PREDICTION_EXPIRED"
        if predicted and getattr(target, "misses", GIMBAL_COAST_MAX_MISSES + 1) > GIMBAL_COAST_MAX_MISSES:
            # Coast through at most two detector frames. Longer extrapolation
            # holds position until a fresh YOLO observation returns.
            self.hold()
            return "HOLD,PREDICT_LIMIT,%d" % target.track_id
        if getattr(target, "track_id", 0) <= 0:
            self.hold()
            return "HOLD,NO_TRACK_ID"
        if getattr(target, "stable", 0) < GIMBAL_MIN_STABLE:
            self.hold()
            return "HOLD,UNSTABLE"
        if not predicted and (getattr(target, "misses", 1) != 0 or getattr(target, "score", 0.0) < ACTIVE_GIMBAL_MIN_SCORE):
            self.hold()
            return "HOLD,QUALITY"

        # A manually selected target may use the calibrated camera-to-aim point
        # when the visible red dot is temporarily not measurable. The actual
        # dot object remains untouched, so telemetry never claims false
        # closed-loop feedback.
        closed_loop = self.closed_loop if use_closed_loop is None else bool(use_closed_loop)
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
    print("GIMBAL_TRACK_OK,MODE,SLOW_SLEW,HZ,%.1f,DEADZONE,%.3f,MAX_STEP,%.2f,AIM,%d,%d,DIST_MM,%d" % (
        GIMBAL_CONTROL_HZ,
        GIMBAL_DEADZONE_X,
        GIMBAL_SLOW_MAX_STEP_DEG,
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
    if not (ENABLE_DISPLAY or _web_overlay_enabled):
        return
    try:
        img.draw_cross(cx, cy, color, size=7, thickness=2)
    except Exception:
        img.draw_line(cx - 7, cy, cx + 7, cy, color, 2)
        img.draw_line(cx, cy - 7, cx, cy + 7, color, 2)


def draw_text(img, x, y, text, color):
    if SHOW_OVERLAY_TEXT and (ENABLE_DISPLAY or _web_overlay_enabled) and _overlay_text_enabled:
        img.draw_string(x, y, text, color)


def draw_rect(img, x, y, w, h, color, thickness=1):
    if ENABLE_DISPLAY or _web_overlay_enabled:
        img.draw_rect(x, y, w, h, color=color, thickness=thickness)


def obj_center(obj):
    return int(obj.x + obj.w // 2), int(obj.y + obj.h // 2)


class SelectedLockGate:
    """Require repeatable same-target observations before closed-loop motion."""

    def __init__(self):
        self.reset()

    def reset(self, obj=None):
        self.attempts = 0
        self.consistent = 0
        self.confirmed = False
        self.failed = False
        self.reference = None
        if obj is not None:
            self.reference = self._measurement(obj)
            self.consistent = 1

    @staticmethod
    def _measurement(obj):
        cx, cy = obj_center(obj)
        return float(cx), float(cy), float(obj.w), float(obj.h)

    def observe(self, obj):
        if self.confirmed or self.failed:
            return self.state()
        self.attempts += 1
        fresh = obj is not None and not getattr(obj, "predicted", False)
        if fresh and self.reference is not None:
            cx, cy, width, height = self._measurement(obj)
            old_cx, old_cy, old_width, old_height = self.reference
            center_shift = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
            scale = max(4.0, (old_width * old_width + old_height * old_height) ** 0.5)
            width_ratio = width / max(1.0, old_width)
            height_ratio = height / max(1.0, old_height)
            max_ratio = SELECTED_LOCK_MAX_SIZE_RATIO
            repeatable = (
                center_shift <= scale * SELECTED_LOCK_MAX_CENTER_SHIFT_BOXES
                and 1.0 / max_ratio <= width_ratio <= max_ratio
                and 1.0 / max_ratio <= height_ratio <= max_ratio
            )
            if repeatable:
                self.consistent += 1
                alpha = 0.35
                self.reference = (
                    old_cx + alpha * (cx - old_cx),
                    old_cy + alpha * (cy - old_cy),
                    old_width + alpha * (width - old_width),
                    old_height + alpha * (height - old_height),
                )
            else:
                self.consistent = 0
        else:
            self.consistent = 0
        if self.consistent >= SELECTED_LOCK_CONFIRM_DETECTIONS:
            self.confirmed = True
        elif self.attempts >= SELECTED_LOCK_TIMEOUT_DETECTIONS:
            self.failed = True
        return self.state()

    def state(self):
        if self.confirmed:
            return "stable"
        if self.failed:
            return "approach_hold"
        return "confirming"


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
                    or dominance < AIM_DOT_RGB_MIN_DOMINANCE
                    or ratio_value < AIM_DOT_RGB_MIN_DOMINANCE_RATIO
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
                    and dominance >= AIM_DOT_RGB_MIN_DOMINANCE
                    and ratio_value >= AIM_DOT_RGB_MIN_DOMINANCE_RATIO
                ):
                    weight = max(1.0, float(dominance - AIM_DOT_RGB_MIN_DOMINANCE + 1))
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
            blobs = img.find_blobs(
                AIM_DOT_LAB_THRESHOLDS,
                roi=[roi_x, roi_y, roi_w, roi_h],
                x_stride=2,
                y_stride=2,
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

        # LAB blob extraction runs in native code and is the normal tracking
        # path. The Python RGB search is deliberately reserved for initial or
        # occasional reacquisition; running hundreds of get_pixel calls on
        # every frame costs more time than YOLO itself on MaixCAM.
        fallback_debug = None
        rgb_reacquire = not candidates and (
            not self.rgb_locked
            or self.debug_frames % AIM_DOT_RGB_REACQUIRE_FRAMES == 0
        )
        if rgb_reacquire:
            rgb_peak, fallback_debug = self._rgb_peak_candidate(
                img,
                ACTIVE_AIM_X,
                ACTIVE_AIM_Y,
                AIM_DOT_EXPECTED_RADIUS_PX,
            )
            if rgb_peak is not None:
                self.rgb_locked = True
                candidates = [rgb_peak]

        if self.debug_frames % AIM_DOT_LOG_EVERY_N_FRAMES == 0 and fallback_debug is not None:
            score, red_value, dominance, ratio_value, px, py = fallback_debug
            print(
                "DOT_RGB_PEAK,%d,%d,R,%d,DOM,%d,RATIO,%.3f,SCORE,%.1f,BLOBS,%d"
                % (px, py, red_value, dominance, ratio_value, score, len(blobs))
            )

        if candidates:
            score, cx, cy = max(candidates, key=lambda item: item[0])
            self.rgb_locked = True
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
    """Center a selected target, then signal the manual-adjust handoff."""

    def __init__(self):
        self.track_id = 0
        self.settled_frames = 0
        self.settled_since = 0.0
        self.centered_ready = False

    def reset(self):
        self.track_id = 0
        self.settled_frames = 0
        self.settled_since = 0.0
        self.centered_ready = False

    def take_centered_ready(self):
        ready = self.centered_ready
        self.centered_ready = False
        return ready

    def point(self, target, aim_dot, now, hold_when_centered=False):
        if target is None:
            self.settled_frames = 0
            return None
        track_id = getattr(target, "track_id", 0)
        if track_id != self.track_id:
            self.track_id = track_id
            self.settled_frames = 0
            self.settled_since = now
            self.centered_ready = False
        point = (target.x + target.w * 0.5, target.y + target.h * 0.5)
        if hold_when_centered and aim_dot is not None and getattr(aim_dot, "fresh", False):
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
                    self.centered_ready = True
                    self.settled_frames = 0
                    self.settled_since = now
            else:
                self.settled_frames = 0
                self.settled_since = now
        return point


def color_gate(img, x, y, w, h, max_checks=MAX_COLOR_CHECKS):
    if not ENABLE_COLOR_GATE or RUN_MODE == 0:
        return True, 1.0, 0.0
    total = 0
    pink = 0
    red_bad = 0
    grid = 2 if max_checks <= 4 else 4
    step_x = max(1, w // grid)
    step_y = max(1, h // grid)
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
                if total >= max_checks:
                    break
                xx += step_x
            if total >= max_checks:
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
        # Keep the model/color gate independent of track coordinates. With NPU
        # dual buffering this box still belongs to the previous capture; track
        # association must happen only after gimbal/time alignment below.
        score_ok = obj.score >= ACTIVE_MIN_MODEL_CONF
        if not score_ok:
            rows.append((idx, obj, True, False, 0.0, 0.0))
            continue
        color_checks = STRONG_COLOR_CHECKS if obj.score >= STRONG_MODEL_CONF else LOW_CONF_COLOR_CHECKS
        ok, pink_ratio, red_ratio = color_gate(img, x, y, w, h, color_checks)
        rows.append((idx, obj, True, ok, pink_ratio, red_ratio))
        if pink_ratio > best_pink_ratio:
            best_pink_ratio = pink_ratio
        if red_ratio > best_red_ratio:
            best_red_ratio = red_ratio
        area = w * h
        red_reject = red_ratio > MAX_RED_BAD_RATIO and red_ratio > pink_ratio * RED_BAD_DOMINANCE
        if SCREEN_TEST_MODE:
            # Monitor gamma and viewing angle can remove most measured pink
            # even though the trained detector still recognizes the egg mass.
            # This branch is opt-in and never changes the outdoor profile.
            color_ok = True
        elif score_ok and obj.score >= STRONG_MODEL_CONF:
            color_ok = not red_reject
        elif score_ok and _manual_lock_active:
            # A weak box is not allowed to create a target, but a selected
            # trajectory may need it under shade or washed-out light. Defer
            # the positive color decision until post-alignment association;
            # retain only the strong red-object rejection here.
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

    def predict(self, dt_frames=1.0):
        dt_frames = max(TRACK_MIN_DT_FRAMES, min(TRACK_MAX_DT_FRAMES, float(dt_frames)))
        self.pos += self.vel * dt_frames
        self.vel = max(-TRACK_MAX_VELOCITY_PX, min(TRACK_MAX_VELOCITY_PX, self.vel))
        old_p01 = self.p01
        process_scale = max(0.5, dt_frames)
        self.p00 = self.p00 + 2.0 * dt_frames * old_p01 + dt_frames * dt_frames * self.p11 + TRACK_KF_PROCESS_NOISE * process_scale
        self.p01 = old_p01 + dt_frames * self.p11
        self.p11 += TRACK_KF_PROCESS_NOISE * 0.25 * process_scale

    def shift(self, delta):
        # Known camera motion changes image position, not target velocity.
        self.pos += float(delta)

    def update(self, measurement, measurement_noise=None):
        measurement = float(measurement)
        innovation = measurement - self.pos
        if measurement_noise is None:
            measurement_noise = TRACK_KF_MEASUREMENT_NOISE
        measurement_noise = max(1.0, float(measurement_noise))
        innovation_cov = self.p00 + measurement_noise
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


def measurement_noise_for_score(score):
    """Make weak, trajectory-gated boxes observe without pulling hard."""
    weakness = max(0.0, DISCOVERY_MODEL_CONF - float(score)) / max(0.01, DISCOVERY_MODEL_CONF)
    return TRACK_KF_MEASUREMENT_NOISE * (1.0 + 2.5 * min(1.0, weakness))


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


def current_gimbal_offsets():
    controller = globals().get("gimbal")
    if controller is None:
        return 0.0, 0.0
    return (
        float(getattr(controller, "pan_offset", 0.0)),
        float(getattr(controller, "tilt_offset", 0.0)),
    )


def gimbal_image_shift(from_pan, from_tilt, to_pan, to_tilt):
    pixels_per_pan_deg = FRAME_W / CAMERA_H_FOV_DEG
    pixels_per_tilt_deg = FRAME_H / CAMERA_V_FOV_DEG
    dx = -(float(to_pan) - float(from_pan)) * pixels_per_pan_deg / GIMBAL_PAN_SIGN
    dy = -(float(to_tilt) - float(from_tilt)) * pixels_per_tilt_deg / GIMBAL_TILT_SIGN
    return dx, dy


def shift_measurements_to_current_frame(objs, dx, dy):
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return objs
    shifted = []
    for obj in objs:
        shifted.append(
            DetectedBox(
                int(round(obj.x + dx)),
                int(round(obj.y + dy)),
                int(obj.w),
                int(obj.h),
                float(obj.score),
            )
        )
    return shifted


def track_match_cost(obj, track):
    predicted_box = track_box(track)
    measured_box = (int(obj.x), int(obj.y), int(obj.w), int(obj.h))
    predicted_cx = predicted_box[0] + predicted_box[2] * 0.5
    predicted_cy = predicted_box[1] + predicted_box[3] * 0.5
    measured_cx, measured_cy = obj_center(obj)
    dx = measured_cx - predicted_cx
    dy = measured_cy - predicted_cy
    distance = (dx * dx + dy * dy) ** 0.5
    scale = max(TRACK_MATCH_MIN_DISTANCE_PX, max(predicted_box[2], predicted_box[3]) * TRACK_MATCH_DISTANCE_SCALE)
    overlap = box_iou(measured_box, predicted_box)
    width_ratio = max(
        measured_box[2] / max(2.0, predicted_box[2]),
        predicted_box[2] / max(2.0, measured_box[2]),
    )
    height_ratio = max(
        measured_box[3] / max(2.0, predicted_box[3]),
        predicted_box[3] / max(2.0, measured_box[3]),
    )
    predicted_aspect = predicted_box[2] / max(2.0, predicted_box[3])
    measured_aspect = measured_box[2] / max(2.0, measured_box[3])
    aspect_ratio = max(
        measured_aspect / max(0.05, predicted_aspect),
        predicted_aspect / max(0.05, measured_aspect),
    )
    if max(width_ratio, height_ratio, aspect_ratio) > TRACK_ASSOC_MAX_SIZE_RATIO:
        return None
    sigma_x = max(3.0, (track["cx"].p00 + TRACK_KF_MEASUREMENT_NOISE) ** 0.5)
    sigma_y = max(3.0, (track["cy"].p00 + TRACK_KF_MEASUREMENT_NOISE) ** 0.5)
    normalized_innovation = ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2) ** 0.5
    if overlap < 0.02 and (distance > scale or normalized_innovation > 3.5):
        return None
    size_cost = (width_ratio - 1.0) + (height_ratio - 1.0)
    shape_cost = aspect_ratio - 1.0
    confidence_penalty = max(0.0, DISCOVERY_MODEL_CONF - obj.score) / max(0.01, DISCOVERY_MODEL_CONF)
    cost = (
        min(2.0, normalized_innovation / 3.0) * 0.42
        + (1.0 - overlap) * 0.30
        + min(2.0, size_cost) * 0.16
        + min(1.0, shape_cost) * 0.07
        + confidence_penalty * 0.05
    )
    low_measurement = obj.score < DISCOVERY_MODEL_CONF
    if low_measurement:
        if not _manual_lock_active or track["id"] != _locked_track_id:
            return None
        if cost > TRACK_LOCKED_LOW_ASSOC_MAX_COST:
            return None
    elif cost > TRACK_ASSOC_MAX_COST:
        return None
    return cost


def update_tracks(objs, dt_frames=1.0):
    global _tracks, _track_last_pan_offset, _track_last_tilt_offset
    current_pan, current_tilt = current_gimbal_offsets()
    if _track_last_pan_offset is None:
        _track_last_pan_offset = current_pan
        _track_last_tilt_offset = current_tilt
    camera_dx, camera_dy = gimbal_image_shift(
        _track_last_pan_offset,
        _track_last_tilt_offset,
        current_pan,
        current_tilt,
    )
    _track_last_pan_offset = current_pan
    _track_last_tilt_offset = current_tilt
    for track in _tracks:
        track["cx"].shift(camera_dx)
        track["cy"].shift(camera_dy)
        track["cx"].predict(dt_frames)
        track["cy"].predict(dt_frames)
        track["w"].predict(dt_frames)
        track["h"].predict(dt_frames)

    for obj in objs:
        obj.associated_track_id = 0
    pairs = []
    for det_idx, obj in enumerate(objs):
        for track_idx, track in enumerate(_tracks):
            cost = track_match_cost(obj, track)
            if cost is not None:
                pairs.append((cost, det_idx, track_idx))
    used_detections = set()
    used_tracks = set()
    next_tracks = []

    def accept_measurement(track, obj):
        measurement_noise = measurement_noise_for_score(obj.score)
        track["cx"].update(obj_center(obj)[0], measurement_noise)
        track["cy"].update(obj_center(obj)[1], measurement_noise)
        track["w"].update(max(2, obj.w), measurement_noise * 1.25)
        track["h"].update(max(2, obj.h), measurement_noise * 1.25)
        track["score"] = obj.score
        track["stable"] += 1
        track["age"] += 1
        track["misses"] = 0
        track["obj"] = track_object(track, False)
        next_tracks.append(track)

    # A phone-selected track is safety-critical. Reserve it from the ordinary
    # greedy matcher, otherwise a nearby strong detection can steal its ID when
    # two egg masses merge or split during camera motion. If two measurements
    # are nearly equally plausible, coast instead of guessing.
    reserved_locked_tracks = set()
    if _manual_lock_active and _locked_track_id > 0:
        for track_idx, track in enumerate(_tracks):
            if track["id"] != _locked_track_id:
                continue
            reserved_locked_tracks.add(track_idx)
            locked_pairs = []
            for det_idx, obj in enumerate(objs):
                cost = track_match_cost(obj, track)
                if cost is not None:
                    locked_pairs.append((cost, det_idx))
            locked_pairs.sort(key=lambda item: item[0])
            if locked_pairs:
                best_cost, best_det_idx = locked_pairs[0]
                second_cost = locked_pairs[1][0] if len(locked_pairs) > 1 else None
                best_obj = objs[best_det_idx]
                max_cost = (
                    TRACK_LOCKED_STRONG_ASSOC_MAX_COST
                    if best_obj.score >= DISCOVERY_MODEL_CONF
                    else TRACK_LOCKED_LOW_ASSOC_MAX_COST
                )
                unambiguous = (
                    second_cost is None
                    or second_cost - best_cost >= TRACK_LOCKED_ASSOC_AMBIGUITY_MARGIN
                )
                if best_cost <= max_cost and unambiguous:
                    best_obj.associated_track_id = track["id"]
                    used_detections.add(best_det_idx)
                    used_tracks.add(track_idx)
                    accept_measurement(track, best_obj)
            break

    pairs = [
        item for item in pairs
        if item[1] not in used_detections and item[2] not in reserved_locked_tracks
    ]
    pairs.sort(key=lambda item: item[0])
    for cost, det_idx, track_idx in pairs:
        if det_idx in used_detections or track_idx in used_tracks:
            continue
        obj = objs[det_idx]
        track = _tracks[track_idx]
        used_detections.add(det_idx)
        used_tracks.add(track_idx)
        obj.associated_track_id = track["id"]
        accept_measurement(track, obj)

    for track_idx, track in enumerate(_tracks):
        if track_idx in used_tracks:
            continue
        track["misses"] += 1
        track["age"] += 1
        track["cx"].vel *= 0.82
        track["cy"].vel *= 0.82
        track["w"].vel *= 0.75
        track["h"].vel *= 0.75
        if track["misses"] <= TRACK_MAX_MISSES:
            track["obj"] = track_object(track, True)
            next_tracks.append(track)

    birth_threshold = SCREEN_TEST_CONF_TH if SCREEN_TEST_MODE else DISCOVERY_MODEL_CONF
    for det_idx, obj in enumerate(objs):
        if det_idx not in used_detections and obj.score >= birth_threshold:
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


def fresh_detection_for_safety(objs):
    """Count only discoverable boxes or weak boxes tied to the selected track."""
    for obj in objs:
        if obj.score >= DISCOVERY_MODEL_CONF:
            return True
        if (
            _manual_lock_active
            and _locked_track_id > 0
            and getattr(obj, "associated_track_id", 0) == _locked_track_id
        ):
            return True
    return False


def clear_target_lock():
    global _locked_track_id, _locked_last_center, _locked_last_size, _locked_missing_frames, _primary_center, _manual_lock_active
    _locked_track_id = 0
    _locked_last_center = None
    _locked_last_size = None
    _locked_missing_frames = 0
    _primary_center = None
    _manual_lock_active = False


def auto_cycle_enabled():
    return file_exists(AUTO_CYCLE_FLAG)


def switch_auto_cycle_target(target_objects, now):
    """Advance a supervised auto test to the next stable visible track."""
    global _locked_track_id, _locked_last_center, _locked_last_size, _auto_cycle_index
    if not target_objects:
        return False
    ordered = sorted(target_objects, key=lambda obj: (obj_center(obj)[1], obj_center(obj)[0]))
    # Use an ordinal over the current sorted detections. Track IDs can be
    # recreated after a camera move, so using the old ID as the cycle cursor
    # can repeatedly select the first surviving track.
    _auto_cycle_index = (_auto_cycle_index + 1) % len(ordered)
    next_obj = ordered[_auto_cycle_index]
    _locked_track_id = int(getattr(next_obj, "track_id", 0))
    _locked_last_center = obj_center(next_obj)
    _locked_last_size = (next_obj.w, next_obj.h)
    print("AUTO_CYCLE,SWITCH,%d,%d,%d" % (_locked_track_id, _locked_last_center[0], _locked_last_center[1]))
    return True


def lock_target(obj):
    global _locked_track_id, _locked_last_center, _locked_last_size, _locked_missing_frames, _primary_center, _manual_lock_active
    track_id = int(getattr(obj, "track_id", 0))
    if track_id <= 0:
        return 0
    _locked_track_id = track_id
    _primary_center = obj_center(obj)
    _locked_last_center = _primary_center
    _locked_last_size = (obj.w, obj.h)
    _locked_missing_frames = 0
    _manual_lock_active = True
    obj.aim_lock_id = 1
    print("LOCK,WEB_SELECT,%d,%d,%d" % (track_id, _primary_center[0], _primary_center[1]))
    return track_id


def target_at_point(objs, point_x, point_y):
    hits = []
    for obj in objs:
        if getattr(obj, "predicted", False):
            continue
        margin = max(3.0, min(obj.w, obj.h) * 0.08)
        if (
            obj.x - margin <= point_x <= obj.x + obj.w + margin
            and obj.y - margin <= point_y <= obj.y + obj.h + margin
        ):
            cx, cy = obj_center(obj)
            distance = (cx - point_x) ** 2 + (cy - point_y) ** 2
            hits.append((obj.w * obj.h, distance, obj))
    return min(hits, key=lambda item: (item[0], item[1]))[2] if hits else None


def select_primary_target(objs, manual_lock=False):
    global _locked_track_id, _locked_last_center, _locked_last_size, _locked_missing_frames, _primary_center
    if PRIMARY_TARGET_POLICY in ("leftmost", "robust_center"):
        raw_fresh = [obj for obj in objs if not getattr(obj, "predicted", False)]
        fresh = raw_fresh
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
            if manual_lock:
                # A phone-selected target is safety-critical: a nearby fresh
                # detection is not evidence that it is the same object. Keep
                # the original track prediction when available and otherwise
                # stop the gimbal until that exact track returns. Never switch
                # to another egg just because the selected one is occluded.
                if locked_predicted and getattr(locked_predicted[0], "misses", 99) <= GIMBAL_PREDICT_MAX_MISSES:
                    locked_predicted[0].aim_lock_id = 1
                    _primary_center = obj_center(locked_predicted[0])
                    _locked_last_center = _primary_center
                    _locked_last_size = (locked_predicted[0].w, locked_predicted[0].h)
                    return locked_predicted[0]
                return None
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
            switch_missing_frames = globals().get("LOCK_SWITCH_MISSING_FRAMES", 6)
            if fresh and _locked_missing_frames >= switch_missing_frames:
                # The old lock is no longer supported by a fresh detection;
                # release it so the current candidates can be selected.
                _locked_track_id = 0
                _locked_last_center = None
                _locked_last_size = None
                _locked_missing_frames = 0
            if _locked_track_id > 0 and _locked_missing_frames <= LOCK_RELEASE_MISSING_FRAMES:
                return None
            if _locked_track_id > 0:
                print("LOCK,RELEASE,%d,MISSES,%d" % (_locked_track_id, _locked_missing_frames))
                _locked_track_id = 0
        if manual_lock:
            return None
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
if gimbal:
    atexit.register(gimbal.close)
gimbal_tracker = init_gimbal_tracker(gimbal)
aim_relay = init_aim_relay()
aim_dot_detector = AimDotDetector()
aim_waypoint_planner = AimWaypointPlanner()
web_control = None
if WebControl is not None and not file_exists(WEB_CONTROL_DISABLE_FLAG):
    try:
        web_control = WebControl(port=WEB_CONTROL_PORT)
        web_control.start()
        atexit.register(web_control.stop)
    except Exception as e:
        print("WEB_CONTROL_ERROR,%s" % e)
jpeg_streamer = None
if http is not None and not file_exists(WEB_CONTROL_DISABLE_FLAG):
    try:
        native_streamer = http.JpegStreamer("0.0.0.0", WEB_STREAM_PORT, WEB_STREAM_MAX_CLIENTS)
        native_streamer.start()
        jpeg_streamer = LatestJpegStreamer(native_streamer)
        atexit.register(jpeg_streamer.close)
        print("WEB_STREAM,LISTEN,0.0.0.0,%d" % WEB_STREAM_PORT)
    except Exception as e:
        jpeg_streamer = None
        print("WEB_STREAM_ERROR,%s" % e)
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
detector_class = nn.YOLO11 if "yolo11" in MODEL.lower() else nn.YOLOv8
detector = detector_class(model=MODEL, dual_buff=DUAL_BUFF)
print("INIT,DETECTOR_OK")
print("INIT,TILES,%d" % len(tile_origins(FRAME_W, FRAME_H, detector.input_width(), detector.input_height())))
cam = camera.Camera(
    FRAME_W,
    FRAME_H,
    detector.input_format(),
    fps=CAMERA_TARGET_FPS,
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
last_yolo_frame = -DETECT_EVERY_N_FRAMES
last_raw_count = 0
last_candidate_count = 0
last_pink_ratio = 0.0
last_red_ratio = 0.0
last_candidate_rows = []
debug_saved = 0
manual_capture_saved = 0
overlay_stream_saved = 0
aim_detect_frames = 0
aim_missing_frames = 0
last_fresh_egg_time = 0.0
web_mode = "select"
web_pan = 0.0
web_tilt = 0.0
web_aim_override = None
web_estop = False
last_control_mode = None
selected_loss_detections = 0
selected_lock_gate = SelectedLockGate()
selected_lock_state = "none"
selected_goal_valid = False
selected_goal_pan = 0.0
selected_goal_tilt = 0.0
selected_loss_active = False
selected_last_box = None
selected_last_box_pan = 0.0
selected_last_box_tilt = 0.0
profile_count = 0
profile_detect_count = 0
profile_read_s = 0.0
profile_dot_s = 0.0
profile_detect_s = 0.0
profile_control_draw_s = 0.0
profile_encode_s = 0.0
profile_display_s = 0.0
profile_stream_s = 0.0
stream_fps = 0.0
detect_window_start = pytime.time()
detect_window_frames = 0
detect_fps = 0.0
print("INIT,LOOP_START")
profile_clock = getattr(pytime, "monotonic", pytime.time)
track_clock_last = profile_clock()
closed_loop_enabled_runtime = closed_loop_aim_requested()
no_target_test_runtime = file_exists(AIM_RELAY_NO_TARGET_TEST_FLAG)
capture_once_runtime = file_exists(DEBUG_CAPTURE_ONCE_FLAG)
capture_frames_runtime = file_exists(DEBUG_CAPTURE_FLAG)
overlay_stream_runtime = file_exists(DEBUG_OVERLAY_STREAM_FLAG)

while not app.need_exit():
    profile_loop_start = profile_clock()
    track_elapsed = max(0.0, profile_loop_start - track_clock_last)
    track_clock_last = profile_loop_start
    track_dt_frames = max(
        TRACK_MIN_DT_FRAMES,
        min(TRACK_MAX_DT_FRAMES, track_elapsed * TRACK_REFERENCE_FPS),
    )
    _overlay_text_enabled = (frame_id % OVERLAY_TEXT_EVERY_N_FRAMES) == 0
    web_snapshot_requested = False
    _web_overlay_enabled = ENABLE_DISPLAY
    if frame_id % RUNTIME_FLAG_POLL_EVERY_N_FRAMES == 0:
        no_target_test_runtime = file_exists(AIM_RELAY_NO_TARGET_TEST_FLAG)
        capture_once_runtime = file_exists(DEBUG_CAPTURE_ONCE_FLAG)
        capture_frames_runtime = file_exists(DEBUG_CAPTURE_FLAG)
        overlay_stream_runtime = file_exists(DEBUG_OVERLAY_STREAM_FLAG)
    run_detection = frame_id - last_yolo_frame >= DETECT_EVERY_N_FRAMES
    if RUN_MODE == 3 and frame_id < 3:
        print("TRACE,%d,READ_BEGIN" % frame_id)
    img = cam.read()
    profile_read_done = profile_clock()
    dot_enabled = bool(
        closed_loop_enabled_runtime
        and aim_relay is not None
        # Do not look for feedback while the two button pulses are still
        # changing the physical light state; pink image detail can otherwise
        # become the initial dot anchor before the real spot exists.
        and aim_relay.logical_on
        and aim_relay.phase == "IDLE"
    )
    dot_loop_period = AIM_DOT_TRACK_LOOP_EVERY_N_FRAMES if (
        _previous_aim_dot is not None and getattr(_previous_aim_dot, "fresh", False)
    ) else AIM_DOT_LOOP_EVERY_N_FRAMES
    if dot_enabled and _previous_aim_dot is not None and frame_id % dot_loop_period != 0:
        aim_dot = AimDot(
            _previous_aim_dot.x,
            _previous_aim_dot.y,
            _previous_aim_dot.score,
            _previous_aim_dot.fresh,
            _previous_aim_dot.misses,
        )
    else:
        aim_dot = aim_dot_detector.detect(img, dot_enabled)
    # A fixed calibration point is useful as a visual hint, but it is not
    # measured feedback. Never label it fresh or let closed-loop control treat
    # it as a visible red dot.
    if (
        dot_enabled
        and aim_dot is None
        and aim_dot_detector.misses >= AIM_DOT_FIXED_FALLBACK_AFTER_MISSES
    ):
        aim_dot = AimDot(ACTIVE_AIM_X, ACTIVE_AIM_Y, 0.0, False, aim_dot_detector.misses)
    profile_dot_done = profile_clock()
    _last_aim_dot = aim_dot
    # YOLO dual buffering returns the previous frame's detections. Pair them
    # with the previous red-dot measurement so closed-loop error stays aligned.
    control_aim_dot = _previous_aim_dot if DUAL_BUFF else aim_dot
    _previous_aim_dot = aim_dot
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
    if run_detection:
        raw_count, candidates, best_pink_ratio, best_red_ratio, candidate_rows = detect_frame(detector, img, frame_id)
        capture_pan, capture_tilt = current_gimbal_offsets()
        if DUAL_BUFF:
            if _pending_detection_pan_offset is not None:
                measurement_dx, measurement_dy = gimbal_image_shift(
                    _pending_detection_pan_offset,
                    _pending_detection_tilt_offset,
                    capture_pan,
                    capture_tilt,
                )
                candidates = shift_measurements_to_current_frame(candidates, measurement_dx, measurement_dy)
            _pending_detection_pan_offset = capture_pan
            _pending_detection_tilt_offset = capture_tilt
        detect_window_frames += 1
        detect_elapsed = pytime.time() - detect_window_start
        if detect_elapsed >= 1.0:
            detect_fps = detect_window_frames / max(0.001, detect_elapsed)
            detect_window_frames = 0
            detect_window_start = pytime.time()
        last_yolo_frame = frame_id
        last_raw_count = raw_count
        last_candidate_count = len(candidates)
        last_pink_ratio = best_pink_ratio
        last_red_ratio = best_red_ratio
        last_candidate_rows = candidate_rows
    else:
        # Empty detections advance the Kalman/velocity tracks without claiming
        # that a fresh YOLO target exists. Relay decisions are updated only on
        # YOLO refresh frames, so a valid target does not flicker between them.
        raw_count = 0
        candidates = []
        best_pink_ratio = last_pink_ratio
        best_red_ratio = last_red_ratio
        candidate_rows = []
    profile_detect_done = profile_clock()
    if capture_once_runtime:
        try:
            capture_path = "%s/once_%06d.jpg" % (DEBUG_DIR, frame_id)
            img.save(capture_path)
            os.remove(DEBUG_CAPTURE_ONCE_FLAG)
            capture_once_runtime = False
            print("MANUAL_CAPTURE_ONCE,%s" % capture_path)
        except Exception as e:
            print("MANUAL_CAPTURE_ONCE_ERROR,%s" % e)
    if (
        manual_capture_saved < MANUAL_CAPTURE_MAX_SAVES
        and capture_frames_runtime
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
    stable = update_tracks(candidates, track_dt_frames)
    targets = [] if frame_id < WARMUP_FRAMES else sort_targets(stable, FRAME_H)
    if no_target_test_runtime:
        targets = []
    if web_control is not None and frame_id % WEB_CONTROL_POLL_EVERY_N_FRAMES == 0:
        web_mode, web_pan, web_tilt, web_aim_override, web_estop = web_control.get_control()
    elif web_control is None:
        web_mode, web_pan, web_tilt, web_aim_override, web_estop = "auto", 0.0, 0.0, None, False
    target_objects = [item[0] for item in targets]
    selection_request = web_control.consume_selection_request() if web_control is not None else None
    selection_changed = False
    if selection_request is not None:
        action, select_x, select_y = selection_request
        aim_waypoint_planner.reset()
        if action == "clear":
            clear_target_lock()
            selected_lock_gate.reset()
            selected_lock_state = "none"
            selected_goal_valid = False
            selected_loss_active = False
            selected_last_box = None
            selection_changed = True
        else:
            selected_obj = target_at_point(target_objects, select_x * FRAME_W, select_y * FRAME_H)
            if selected_obj is not None:
                selected_track_id = lock_target(selected_obj)
                # Freeze the clicked target's identity and initial box. From
                # this point on, YOLO detections cannot replace the target or
                # make its lock box disappear during a camera move.
                current_pan_offset = float(getattr(gimbal, "pan_offset", 0.0)) if gimbal is not None else 0.0
                current_tilt_offset = float(getattr(gimbal, "tilt_offset", 0.0)) if gimbal is not None else 0.0
                desired_x = selected_obj.x + selected_obj.w * 0.5
                desired_y = selected_obj.y + selected_obj.h * 0.5
                pixels_per_pan_deg = FRAME_W / CAMERA_H_FOV_DEG
                pixels_per_tilt_deg = FRAME_H / CAMERA_V_FOV_DEG
                reference_x = ACTIVE_AIM_X
                reference_y = ACTIVE_AIM_Y
                selected_goal_pan = clamp_pan_offset(
                    current_pan_offset
                    + GIMBAL_PAN_SIGN * (desired_x - reference_x) / pixels_per_pan_deg
                )
                selected_goal_tilt = clamp_tilt_offset(
                    current_tilt_offset
                    + GIMBAL_TILT_SIGN * (desired_y - reference_y) / pixels_per_tilt_deg
                )
                selected_goal_valid = True
                selected_last_box = (
                    int(selected_obj.x), int(selected_obj.y),
                    int(selected_obj.w), int(selected_obj.h),
                )
                selected_last_box_pan = current_pan_offset
                selected_last_box_tilt = current_tilt_offset
                selected_loss_active = False
                selected_loss_detections = 0
                selected_lock_gate.reset(selected_obj)
                selected_lock_state = selected_lock_gate.state()
                if web_control is not None:
                    web_control.confirm_selection(selected_track_id)
                # Do not move on the single click. The next detector frames
                # decide whether this is a trustworthy lock or only a target
                # estimate for conservative manual hand-off.
                web_mode = "selected"
                selection_changed = True
            else:
                clear_target_lock()
                web_control.reject_selection()
                web_mode = "select"
                selection_changed = True
    if web_mode == "selected":
        primary_obj = select_primary_target(target_objects, manual_lock=True)
    elif web_mode == "auto":
        primary_obj = select_primary_target(target_objects)
    else:
        primary_obj = None
    primary_id = getattr(primary_obj, "aim_lock_id", getattr(primary_obj, "track_id", 0)) if primary_obj else 0
    fresh_aim_present = bool(primary_obj is not None and not getattr(primary_obj, "predicted", False))
    # Optional supervised demo mode: once the current auto target is centered
    # and remains fresh, advance to the next visible track. This is disabled by
    # default so ordinary auto tracking never changes targets unexpectedly.
    if web_mode == "auto" and auto_cycle_enabled() and primary_obj is not None and fresh_aim_present:
        aim_cx, aim_cy = obj_center(primary_obj)
        center_error = ((aim_cx - ACTIVE_AIM_X) ** 2 + (aim_cy - ACTIVE_AIM_Y) ** 2) ** 0.5
        if center_error <= AUTO_CYCLE_TOLERANCE_PX:
            if _auto_cycle_since <= 0.0:
                _auto_cycle_since = pytime.time()
            elif pytime.time() - _auto_cycle_since >= AUTO_CYCLE_HOLD_S:
                if switch_auto_cycle_target(target_objects, pytime.time()):
                    _auto_cycle_since = pytime.time()
                    primary_obj = select_primary_target(target_objects)
                    primary_id = getattr(primary_obj, "track_id", 0) if primary_obj else 0
        else:
            _auto_cycle_since = 0.0
    elif web_mode != "auto":
        _auto_cycle_since = 0.0
    if web_mode == "selected" and run_detection:
        if fresh_aim_present:
            selected_loss_detections = 0
        else:
            selected_loss_detections += 1
        if not selected_lock_gate.confirmed and not selected_lock_gate.failed:
            selected_lock_state = selected_lock_gate.observe(primary_obj if fresh_aim_present else None)
            if selected_lock_gate.confirmed and web_control is not None:
                web_control.mark_stable_tracking(selected_track_id)
        elif selected_lock_gate.confirmed:
            selected_lock_state = "stable"
        else:
            selected_lock_state = "approach_hold"
    elif web_mode != "selected":
        selected_loss_detections = 0
    fresh_egg_present = fresh_detection_for_safety(candidates)
    if run_detection:
        if fresh_egg_present:
            aim_detect_frames += 1
            aim_missing_frames = 0
            last_fresh_egg_time = pytime.time()
        else:
            aim_detect_frames = 0
            aim_missing_frames += 1
    if aim_relay:
        if web_estop or web_aim_override is False:
            relay_decision = False
        elif run_detection:
            # A web "ON" request never bypasses the whole-view egg safety gate.
            missing_seconds = pytime.time() - last_fresh_egg_time if last_fresh_egg_time else 999.0
            relay_decision = aim_relay_decision(
                fresh_egg_present, aim_detect_frames, aim_missing_frames, missing_seconds
            )
        else:
            relay_decision = None
        if relay_decision is not None:
            aim_relay.request(relay_decision)
        aim_relay.update(pytime.time())
    selected_dot_fresh = bool(
        web_mode == "selected"
        and control_aim_dot is not None
        and getattr(control_aim_dot, "fresh", False)
    )
    selected_open_loop = bool(web_mode == "selected" and not selected_dot_fresh)
    # In selected mode only, use the calibrated fixed point as an explicit
    # fallback when the dot is lost. This does not alter the real dot
    # measurement reported to the web UI.
    planner_aim_dot = control_aim_dot
    if selected_open_loop:
        planner_aim_dot = AimDot(ACTIVE_AIM_X, ACTIVE_AIM_Y, 0.0, True, 0)
    current_pan_offset = float(getattr(gimbal, "pan_offset", 0.0)) if gimbal is not None else 0.0
    current_tilt_offset = float(getattr(gimbal, "tilt_offset", 0.0)) if gimbal is not None else 0.0
    if web_mode == "selected" and selected_lock_gate.confirmed and fresh_aim_present and primary_obj is not None:
        desired_x = primary_obj.x + primary_obj.w * 0.5
        desired_y = primary_obj.y + primary_obj.h * 0.5
        reference_x = planner_aim_dot.x if planner_aim_dot is not None else ACTIVE_AIM_X
        reference_y = planner_aim_dot.y if planner_aim_dot is not None else ACTIVE_AIM_Y
        pixels_per_pan_deg = FRAME_W / CAMERA_H_FOV_DEG
        pixels_per_tilt_deg = FRAME_H / CAMERA_V_FOV_DEG
        raw_goal_pan = current_pan_offset + GIMBAL_PAN_SIGN * (desired_x - reference_x) / pixels_per_pan_deg
        raw_goal_tilt = current_tilt_offset + GIMBAL_TILT_SIGN * (desired_y - reference_y) / pixels_per_tilt_deg
        raw_goal_pan = clamp_pan_offset(raw_goal_pan)
        raw_goal_tilt = clamp_tilt_offset(raw_goal_tilt)
        if not selected_goal_valid or selection_changed:
            selected_goal_pan = raw_goal_pan
            selected_goal_tilt = raw_goal_tilt
            selected_goal_valid = True
        else:
            alpha = SELECTED_GOAL_FILTER_ALPHA
            selected_goal_pan += alpha * (raw_goal_pan - selected_goal_pan)
            selected_goal_tilt += alpha * (raw_goal_tilt - selected_goal_tilt)
        selected_last_box = (int(primary_obj.x), int(primary_obj.y), int(primary_obj.w), int(primary_obj.h))
        selected_last_box_pan = current_pan_offset
        selected_last_box_tilt = current_tilt_offset
    if (
        web_mode == "selected"
        and not selected_lock_gate.confirmed
        and selected_lock_gate.failed
    ):
        # Never chase a target from one unstable observation. Finish only the
        # original click-derived angle, then hand control to the operator.
        aim_waypoint_planner.reset()
        if web_control is not None:
            web_control.begin_conservative_finish(
                current_pan_offset,
                current_tilt_offset,
                selected_goal_pan if selected_goal_valid else current_pan_offset,
                selected_goal_tilt if selected_goal_valid else current_tilt_offset,
                selected_track_id,
            )
        web_mode = "manual"
        web_pan = selected_goal_pan if selected_goal_valid else current_pan_offset
        web_tilt = selected_goal_tilt if selected_goal_valid else current_tilt_offset
        primary_obj = None
        primary_id = 0
        selected_loss_active = True
        selected_loss_detections = 0
        selected_lock_state = "approach_hold"
    if web_mode == "selected" and selected_lock_gate.confirmed and selected_loss_detections >= SELECTED_LOSS_CONFIRM_DETECTIONS:
        # Once a selected target is confirmed lost, finish the last reliable
        # angular plan slowly, then hold. Do not resume from later detections.
        aim_waypoint_planner.reset()
        if web_control is not None:
            web_control.begin_lost_finish(
                current_pan_offset,
                current_tilt_offset,
                selected_goal_pan if selected_goal_valid else current_pan_offset,
                selected_goal_tilt if selected_goal_valid else current_tilt_offset,
                selected_track_id,
            )
        web_mode = "manual"
        web_pan = selected_goal_pan if selected_goal_valid else current_pan_offset
        web_tilt = selected_goal_tilt if selected_goal_valid else current_tilt_offset
        primary_obj = None
        primary_id = 0
        selected_loss_active = True
        selected_loss_detections = 0
    control_target = primary_obj if (web_mode != "selected" or selected_lock_gate.confirmed) else None
    aim_waypoint = aim_waypoint_planner.point(
        control_target,
        planner_aim_dot,
        pytime.time(),
        hold_when_centered=(web_mode == "selected"),
    )
    if web_mode == "selected" and selected_lock_gate.confirmed and aim_waypoint_planner.take_centered_ready():
        current_pan_offset = float(getattr(gimbal, "pan_offset", 0.0)) if gimbal is not None else 0.0
        current_tilt_offset = float(getattr(gimbal, "tilt_offset", 0.0)) if gimbal is not None else 0.0
        if web_control is not None:
            web_control.begin_manual_adjust(current_pan_offset, current_tilt_offset, primary_id)
        web_mode = "manual"
        web_pan = current_pan_offset
        web_tilt = current_tilt_offset
        aim_waypoint = None
    control_status = "OFF"
    if gimbal_tracker is not None:
        if web_estop or web_mode == "hold":
            gimbal_tracker.hold()
            control_status = "WEB,HOLD"
        elif web_mode == "manual":
            # Send only the destination. Gimbal's independent 50 Hz
            # jerk-limited trajectory produces the physical motion.
            gimbal.set_offsets(web_pan, web_tilt, "WEB")
            control_status = "WEB,MANUAL,PAN,%.1f,TILT,%.1f" % (web_pan, web_tilt)
        elif web_mode in ("auto", "selected"):
            if web_mode in ("auto", "selected"):
                if last_control_mode != web_mode or selection_changed:
                    gimbal_tracker.sync_to_offsets(
                        float(getattr(gimbal, "pan_offset", 0.0)),
                        float(getattr(gimbal, "tilt_offset", 0.0)),
                    )
                now = pytime.time()
                no_target_seconds = now - last_fresh_egg_time if last_fresh_egg_time else 999.0
                if web_mode == "auto" and primary_obj is None and no_target_seconds >= GIMBAL_RECENTER_NO_TARGET_S:
                    control_status = gimbal_tracker.recenter(now)
                else:
                    control_status = gimbal_tracker.update(
                        primary_obj,
                        FRAME_W,
                        FRAME_H,
                        now,
                        aim_dot=control_aim_dot,
                        waypoint=aim_waypoint,
                        # Automatic inspection must remain usable with the
                        # aiming light forced off. Phone-selected precision
                        # tracking still uses measured red-dot feedback.
                        use_closed_loop=(selected_dot_fresh if web_mode == "selected" else False),
                    )
        else:
            gimbal_tracker.hold()
            control_status = "WEB,SELECT_WAIT"
        last_control_mode = web_mode
        if (
            (control_status.startswith("TRACK,") or control_status.startswith("PREDICT,"))
            and frame_id % AIM_LOG_EVERY_N_FRAMES == 0
        ):
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
    if web_control is not None and frame_id % WEB_CONTROL_POLL_EVERY_N_FRAMES == 0:
        web_control.update_status({
            "fps": round(float(fps_now), 2),
            "loop_fps": round(float(fps_now), 2),
            "detect_hz": round(float(detect_fps), 2),
            "stream_fps": round(float(stream_fps), 2),
            "stream_port": WEB_STREAM_PORT if jpeg_streamer is not None else 0,
            "raw": last_raw_count,
            "candidates": last_candidate_count,
            "eggs": len(targets),
            "primary": primary_id,
            "relay": aim_relay.label() if aim_relay else "disabled",
            "dot": "fresh" if aim_dot is not None and aim_dot.fresh else "lost",
            "aim_mode": "fixed" if selected_open_loop else "closed_loop",
            "selected_lock_state": selected_lock_state,
            "control": control_status,
            "gimbal_pan": round(float(getattr(gimbal, "pan_offset", 0.0)), 2) if gimbal is not None else 0.0,
            "gimbal_tilt": round(float(getattr(gimbal, "tilt_offset", 0.0)), 2) if gimbal is not None else 0.0,
            "gimbal_phase": str(getattr(gimbal, "motion_phase", "hold")) if gimbal is not None else "off",
            "web_url": "http://<maixcam-ip>:%d/?token=%s" % (WEB_CONTROL_PORT, web_control.token),
        })
    if web_control is not None:
        web_snapshot_requested = web_control.consume_frame_request()
        _web_overlay_enabled = ENABLE_DISPLAY or web_snapshot_requested
    draw_text(img, 2, 2, "FPS %.1f YOLO/%d" % (fps_now, DETECT_EVERY_N_FRAMES), image.COLOR_GREEN)
    status_label = "WARM" if frame_id < WARMUP_FRAMES else "EGGS"
    draw_text(img, 2, 18, "M%d RAW %d CAND %d %s %d LOCK %d" % (RUN_MODE, last_raw_count, last_candidate_count, status_label, len(targets), primary_id),
              image.COLOR_GREEN if targets else image.COLOR_RED)
    draw_text(
        img,
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
    elif closed_loop_enabled_runtime:
        dot_label = "DOT LOST"
    else:
        dot_label = "OPEN LOOP"
    aim_label = "%s %s" % (aim_relay.label() if aim_relay else "AIM DISABLED", dot_label)
    draw_text(img, 2, 50, "%s %s" % (gimbal_label, aim_label), image.COLOR_BLUE if gimbal_tracker else image.COLOR_RED)
    if web_control is not None:
        draw_text(img, 2, 66, "WEB :8000 token %s" % web_control.token, image.COLOR_BLUE)

    if aim_dot is not None:
        dot_x, dot_y = int(aim_dot.x), int(aim_dot.y)
        draw_cross(img, dot_x, dot_y, image.COLOR_RED if aim_dot.fresh else image.COLOR_YELLOW)
        draw_rect(img, dot_x - 5, dot_y - 5, 10, 10, image.COLOR_RED, 1)
    if aim_waypoint is not None:
        waypoint_x, waypoint_y = int(aim_waypoint[0]), int(aim_waypoint[1])
        draw_cross(img, waypoint_x, waypoint_y, image.COLOR_BLUE)

    if RUN_MODE == 0:
        raw_targets = sort_targets([row[1] for row in candidate_rows], FRAME_H)
        for idx, item in enumerate(raw_targets):
            obj, cx, cy = item
            draw_rect(img, int(obj.x), int(obj.y), int(obj.w), int(obj.h), image.COLOR_YELLOW, 1)
            draw_cross(img, cx, cy, image.COLOR_YELLOW)

    elif RUN_MODE == 1 or RUN_MODE == 3:
        raw_targets = sort_targets([row[1] for row in candidate_rows], FRAME_H)
        for idx, item in enumerate(raw_targets):
            obj, cx, cy = item
            draw_rect(img, int(obj.x), int(obj.y), int(obj.w), int(obj.h), image.COLOR_YELLOW, 1)

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
        box_thickness = 3 if is_primary and web_mode == "selected" else 2
        draw_rect(img, x, y, w, h, box_color, box_thickness)
        if is_primary:
            draw_cross(img, cx, cy, image.COLOR_BLUE)
        label_y = y - 14 if y >= 14 else y
        display_id = "S%d" % target_id if is_primary and web_mode == "selected" else str(target_id)
        draw_text(img, x, label_y, display_id, box_color)

        if PRINT_EVERY_N_FRAMES > 0 and frame_id % PRINT_EVERY_N_FRAMES == 0:
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

    if selected_loss_active and selected_last_box is not None and gimbal is not None:
        # Keep the last reliable selected box visible while the slow final
        # angular plan completes. This is a display-only prediction; it never
        # feeds a new detection back into the controller.
        last_x, last_y, last_w, last_h = selected_last_box
        ghost_cx = last_x + last_w * 0.5 + (
            float(getattr(gimbal, "pan_offset", 0.0)) - selected_last_box_pan
        ) * (FRAME_W / CAMERA_H_FOV_DEG)
        ghost_cy = last_y + last_h * 0.5 + (
            float(getattr(gimbal, "tilt_offset", 0.0)) - selected_last_box_tilt
        ) * (FRAME_H / CAMERA_V_FOV_DEG)
        ghost_x = int(ghost_cx - last_w * 0.5)
        ghost_y = int(ghost_cy - last_h * 0.5)
        draw_rect(img, ghost_x, ghost_y, last_w, last_h, image.COLOR_YELLOW, 3)
        draw_cross(img, int(ghost_cx), int(ghost_cy), image.COLOR_YELLOW)

    if frame_id % STAT_EVERY_N_FRAMES == 0:
        stat_line = "STAT,%d,LOOP_FPS,%.1f,DETECT_HZ,%.1f,STREAM_FPS,%.1f,TRACK_DT,%.2f,YOLO_FRAME,%d,RAW,%d,CAND,%d,EGGS,%d,TILE,%s" % (
            frame_id, fps_now, detect_fps, stream_fps, track_dt_frames, last_yolo_frame,
            last_raw_count, last_candidate_count, len(targets), _last_tile_info
        )
        print(stat_line)
        write_telemetry(stat_line)
        if primary_obj is None:
            track_line = "TRACK,%d,DETECT,%d,RAW,%d,CAND,%d,EGGS,%d,PRIMARY,0,RAWID,0,PCX,-1,PCY,-1,PRED,0,MISSES,-1,FRESH,%d" % (
                frame_id,
                int(run_detection),
                last_raw_count,
                last_candidate_count,
                len(targets),
                int(fresh_egg_present),
            )
        else:
            primary_cx, primary_cy = obj_center(primary_obj)
            track_line = "TRACK,%d,DETECT,%d,RAW,%d,CAND,%d,EGGS,%d,PRIMARY,%d,RAWID,%d,PCX,%d,PCY,%d,PRED,%d,MISSES,%d,FRESH,%d" % (
                frame_id,
                int(run_detection),
                last_raw_count,
                last_candidate_count,
                len(targets),
                primary_id,
                int(getattr(primary_obj, "track_id", 0)),
                primary_cx,
                primary_cy,
                int(getattr(primary_obj, "predicted", False)),
                int(getattr(primary_obj, "misses", 0)),
                int(fresh_egg_present),
            )
        print(track_line)
        write_telemetry(track_line)
        if control_aim_dot is None or primary_obj is None:
            aim_line = "AIMSTAT,%d,VALID,0,EX,0.000,EY,0.000,DOT,0,PRIMARY,%d" % (frame_id, primary_id)
        else:
            aim_x = aim_waypoint[0] if aim_waypoint is not None else primary_obj.x + primary_obj.w * 0.5
            aim_y = aim_waypoint[1] if aim_waypoint is not None else primary_obj.y + primary_obj.h * 0.5
            aim_ex = (aim_x - control_aim_dot.x) / max(1.0, FRAME_W * 0.5)
            aim_ey = (aim_y - control_aim_dot.y) / max(1.0, FRAME_H * 0.5)
            aim_line = "AIMSTAT,%d,VALID,1,EX,%.4f,EY,%.4f,DOT,%d,PRIMARY,%d" % (
                frame_id,
                aim_ex,
                aim_ey,
                int(getattr(control_aim_dot, "fresh", False)),
                primary_id,
            )
        print(aim_line)
        write_telemetry(aim_line)

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
        and overlay_stream_runtime
        and frame_id % DEBUG_OVERLAY_STREAM_EVERY_N_FRAMES == 0
    ):
        try:
            overlay_path = "%s/demo_overlay_%06d.jpg" % (DEBUG_DIR, frame_id)
            img.save(overlay_path)
            overlay_stream_saved += 1
            if overlay_stream_saved >= DEBUG_OVERLAY_STREAM_MAX_SAVES:
                os.remove(DEBUG_OVERLAY_STREAM_FLAG)
                overlay_stream_runtime = False
            print("DEMO_OVERLAY,%s" % overlay_path)
        except Exception as e:
            print("DEMO_OVERLAY_ERROR,%s" % e)

    # Save web snapshots after drawing detections, so remote recordings show
    # the same annotated frame that is visible on the MaixCAM display.
    profile_pre_display = profile_clock()
    if web_control is not None and web_snapshot_requested:
        try:
            img.save(web_control.snapshot_path)
        except Exception as e:
            print("WEB_SNAPSHOT_ERROR,%s" % e)
        web_control.publish_frame()

    profile_stream_start = profile_clock()
    if jpeg_streamer is not None and frame_id % WEB_STREAM_EVERY_N_FRAMES == 0:
        try:
            stream_jpeg = img.to_jpeg()
            profile_encode_done = profile_clock()
            jpeg_streamer.submit(stream_jpeg)
            stream_fps = jpeg_streamer.fps
        except Exception as e:
            print("WEB_STREAM_WRITE_ERROR,%s" % e)
            profile_encode_done = profile_clock()
    else:
        profile_encode_done = profile_clock()
    profile_stream_done = profile_clock()

    if disp:
        disp.show(img)
    profile_display_done = profile_clock()
    profile_count += 1
    profile_read_s += profile_read_done - profile_loop_start
    profile_dot_s += profile_dot_done - profile_read_done
    if run_detection:
        profile_detect_count += 1
        profile_detect_s += profile_detect_done - profile_dot_done
    profile_control_draw_s += profile_pre_display - profile_detect_done
    profile_encode_s += profile_encode_done - profile_stream_start
    profile_stream_s += profile_stream_done - profile_stream_start
    profile_display_s += profile_display_done - profile_stream_done
    if profile_count >= PROFILE_EVERY_N_FRAMES:
        prof_line = (
            "PROF,%d,FRAMES,%d,DETECTS,%d,READ_MS,%.2f,DOT_MS,%.2f,"
            "DETECT_MS,%.2f,CONTROL_DRAW_MS,%.2f,ENCODE_MS,%.2f,ENQUEUE_MS,%.2f,DISPLAY_MS,%.2f"
            % (
                frame_id,
                profile_count,
                profile_detect_count,
                1000.0 * profile_read_s / max(1, profile_count),
                1000.0 * profile_dot_s / max(1, profile_count),
                1000.0 * profile_detect_s / max(1, profile_detect_count),
                1000.0 * profile_control_draw_s / max(1, profile_count),
                1000.0 * profile_encode_s / max(1, profile_count),
                1000.0 * (profile_stream_s - profile_encode_s) / max(1, profile_count),
                1000.0 * profile_display_s / max(1, profile_count),
            )
        )
        print(prof_line)
        write_telemetry(prof_line)
        profile_count = 0
        profile_detect_count = 0
        profile_read_s = 0.0
        profile_dot_s = 0.0
        profile_detect_s = 0.0
        profile_control_draw_s = 0.0
        profile_encode_s = 0.0
        profile_stream_s = 0.0
        profile_display_s = 0.0
    frame_id += 1

if gimbal:
    gimbal.close()
if aim_relay:
    aim_relay.close()
if web_control is not None:
    web_control.stop()
if jpeg_streamer is not None:
    jpeg_streamer.close()
print("YOLO SNAIL EGG DETECTOR STOP")
