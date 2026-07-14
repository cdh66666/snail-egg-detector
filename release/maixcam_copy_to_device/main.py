import os
import time as pytime

from maix import camera, display, image, nn, app, time, pwm, pinmap


# Copy this file to MaixVision as main.py after converting the ONNX model to MUD.
# Put the MUD and model file under /root/models/ on the MaixCam.

MODEL = "/root/models/snail_eggs_yolov8n_640x480.mud"
CONF_TH = 0.38
IOU_TH = 0.35
MAX_TARGETS = 32
MIN_MODEL_CONF = 0.38
STRONG_MODEL_CONF = 0.38
LOW_CONF_MIN_PINK_RATIO = 0.035
LOW_CONF_MIN_AREA = 18

# Basic gimbal control. On the current wiring A18/PWM6 is pan and
# A19/PWM7 is tilt. Every command is clamped to center +/-45 degrees
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
GIMBAL_MAX_OFFSET_DEG = 45
GIMBAL_CENTER_SETTLE_S = 0.35
GIMBAL_PAN_PWM_ID = 6
GIMBAL_TILT_PWM_ID = 7
# Control loop is slower than the detector on purpose. The detector/display
# can stay near 19 FPS while the hobby servos receive smooth 10 Hz updates.
GIMBAL_CONTROL_HZ = 10.0
GIMBAL_DEADZONE_X = 0.040
GIMBAL_DEADZONE_Y = 0.050
GIMBAL_MAX_STEP_DEG = 0.7
GIMBAL_MAX_RATE_DEG_S = 6.0
GIMBAL_MAX_ACCEL_DEG_S2 = 12.0
GIMBAL_ERROR_FILTER_TAU_S = 0.12
# In the velocity controller KP maps normalized image error to deg/s.
GIMBAL_PAN_KP = 20.0
GIMBAL_PAN_KI = 0.0
GIMBAL_PAN_KD = 6.0
GIMBAL_TILT_KP = 16.0
GIMBAL_TILT_KI = 0.0
GIMBAL_TILT_KD = 4.0
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
TRACK_MAX_MISSES = 6
TRACK_PREDICT_MAX_MISSES = 4
TRACK_MATCH_DISTANCE_SCALE = 1.35
TRACK_KF_PROCESS_NOISE = 2.0
TRACK_KF_MEASUREMENT_NOISE = 16.0
# 0 means automatically lock the first stable track; set a positive ID to
# lock a specific track later when the gimbal controller is connected.
LOCK_TARGET_ID = 0
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
_tile_scan_index = 0
_memory = []
_last_tile_info = "0/0"


def clamp_gimbal_offset(offset):
    if offset < -GIMBAL_MAX_OFFSET_DEG:
        return -GIMBAL_MAX_OFFSET_DEG
    if offset > GIMBAL_MAX_OFFSET_DEG:
        return GIMBAL_MAX_OFFSET_DEG
    return offset


def clamp_gimbal_angle(angle):
    min_angle = GIMBAL_CENTER_DEG - GIMBAL_MAX_OFFSET_DEG
    max_angle = GIMBAL_CENTER_DEG + GIMBAL_MAX_OFFSET_DEG
    if angle < min_angle:
        return min_angle
    if angle > max_angle:
        return max_angle
    return angle


def servo_duty_from_angle(angle):
    # Standard hobby servo pulse: 0.5ms..2.5ms in a 20ms period.
    # That maps 0..180 degrees to 2.5%..12.5% duty at 50Hz.
    angle = clamp_gimbal_angle(angle)
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
        pinmap.set_pin_function("A19", "PWM7")
        pinmap.set_pin_function("A18", "PWM6")
        center_duty = servo_duty_from_angle(GIMBAL_CENTER_DEG)
        self.pan = pwm.PWM(GIMBAL_PAN_PWM_ID, freq=GIMBAL_FREQ, duty=center_duty, enable=True)
        self.tilt = pwm.PWM(GIMBAL_TILT_PWM_ID, freq=GIMBAL_FREQ, duty=center_duty, enable=True)
        safe_sleep(GIMBAL_CENTER_SETTLE_S)

    def set_offsets(self, pan_offset, tilt_offset, source="CMD"):
        self.pan_offset = clamp_gimbal_offset(pan_offset)
        self.tilt_offset = clamp_gimbal_offset(tilt_offset)
        pan_angle = clamp_gimbal_angle(GIMBAL_CENTER_DEG + self.pan_offset)
        tilt_angle = clamp_gimbal_angle(GIMBAL_CENTER_DEG + self.tilt_offset)
        self.pan.duty(servo_duty_from_angle(pan_angle))
        self.tilt.duty(servo_duty_from_angle(tilt_angle))
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
        self.pan_offset = clamp_gimbal_offset(pan_offset)
        self.tilt_offset = clamp_gimbal_offset(tilt_offset)
        print(
            "GIMBAL,DRY_RUN,%s,PAN_OFFSET,%.2f,TILT_OFFSET,%.2f"
            % (source, self.pan_offset, self.tilt_offset)
        )

    def close(self):
        print("GIMBAL,DRY_RUN,CLOSE")


class GimbalAxisController:
    """Acceleration-limited image-error to angular-velocity controller.

    The output is an angle increment, not a raw PWM duty.  This makes the
    absolute +/-45 degree mechanical limit and the per-update slew limit easy
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

    def reset(self):
        self.pan.reset()
        self.tilt.reset()

    def update(self, target, frame_w, frame_h, now):
        # Never drive from a predicted box.  A predicted box is useful for
        # stable display/ID continuity, but it is not fresh visual evidence.
        if target is None or getattr(target, "predicted", False):
            self.reset()
            return "HOLD,NO_FRESH_TARGET"
        if getattr(target, "track_id", 0) <= 0:
            self.reset()
            return "HOLD,NO_TRACK_ID"
        if getattr(target, "stable", 0) < GIMBAL_MIN_STABLE:
            self.reset()
            return "HOLD,UNSTABLE"
        if getattr(target, "misses", 1) != 0 or getattr(target, "score", 0.0) < ACTIVE_GIMBAL_MIN_SCORE:
            self.reset()
            return "HOLD,QUALITY"

        interval = 1.0 / GIMBAL_CONTROL_HZ
        if self.last_update > 0.0 and now - self.last_update < interval:
            return "HOLD,RATE"
        dt = interval if self.last_update <= 0.0 else max(0.001, now - self.last_update)
        self.last_update = now

        cx = target.x + target.w * 0.5
        cy = target.y + target.h * 0.5
        reference_x = ACTIVE_AIM_X
        reference_y = ACTIVE_AIM_Y
        error_x = (cx - reference_x) / max(1.0, frame_w * 0.5)
        error_y = (cy - reference_y) / max(1.0, frame_h * 0.5)
        if abs(error_x) < GIMBAL_DEADZONE_X:
            error_x = 0.0
        if abs(error_y) < GIMBAL_DEADZONE_Y:
            error_y = 0.0

        self.pan_offset = clamp_gimbal_offset(self.pan_offset + self.pan.step(error_x, dt))
        self.tilt_offset = clamp_gimbal_offset(self.tilt_offset + self.tilt.step(error_y, dt))
        self.gimbal.set_offsets(self.pan_offset, self.tilt_offset, "TRACK")
        return "TRACK,%d,EX,%.3f,EY,%.3f,PAN,%.2f,TILT,%.2f,REF,%d,%d" % (
            target.track_id,
            error_x,
            error_y,
            self.pan_offset,
            self.tilt_offset,
            int(reference_x),
            int(reference_y),
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
        print("GIMBAL_INIT_START,LIMIT,+-%d" % GIMBAL_MAX_OFFSET_DEG)
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
    return (
        abs(cx - ACTIVE_AIM_X) <= AIM_MARKER_REJECT_RADIUS_PX
        and abs(cy - ACTIVE_AIM_Y) <= AIM_MARKER_REJECT_RADIUS_PX
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
        if is_aim_marker_box(obj):
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
        if score_ok and obj.score >= STRONG_MODEL_CONF:
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
    scale = max(28.0, max(predicted_box[2], predicted_box[3]) * TRACK_MATCH_DISTANCE_SCALE)
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
    return [
        track["obj"]
        for track in _tracks
        if track["stable"] >= REQUIRE_STABLE_FRAMES
        and track["misses"] <= TRACK_PREDICT_MAX_MISSES
    ]


def select_primary_target(objs):
    global _locked_track_id
    ids = [getattr(obj, "track_id", 0) for obj in objs]
    if LOCK_TARGET_ID > 0:
        _locked_track_id = LOCK_TARGET_ID if LOCK_TARGET_ID in ids else 0
    elif _locked_track_id not in ids:
        if objs:
            # Lock the first stable target by score; this remains fixed while
            # the target is briefly occluded or missed by the detector.
            chosen = sorted(objs, key=lambda obj: obj.score, reverse=True)[0]
            _locked_track_id = getattr(chosen, "track_id", 0)
        else:
            _locked_track_id = 0
    for obj in objs:
        if getattr(obj, "track_id", 0) == _locked_track_id:
            return obj
    return None


def merge_overlaps(objs):
    ordered = sorted(objs, key=lambda obj: obj.score, reverse=True)
    kept = []
    for obj in ordered:
        box = (int(obj.x), int(obj.y), int(obj.w), int(obj.h))
        duplicate = False
        for other in kept:
            other_box = (int(other.x), int(other.y), int(other.w), int(other.h))
            if box_iou(box, other_box) >= MERGE_IOU:
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
print("INIT,LOOP_START")

while not app.need_exit():
    if RUN_MODE == 3 and frame_id < 3:
        print("TRACE,%d,READ_BEGIN" % frame_id)
    img = cam.read()
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
    primary_obj = select_primary_target([item[0] for item in targets])
    primary_id = getattr(primary_obj, "track_id", 0) if primary_obj else 0
    control_status = "OFF"
    if gimbal_tracker is not None:
        control_status = gimbal_tracker.update(primary_obj, FRAME_W, FRAME_H, pytime.time())
        if control_status.startswith("TRACK,"):
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
    img.draw_string(2, 50, gimbal_label, image.COLOR_BLUE if gimbal_tracker else image.COLOR_RED)

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
        target_id = getattr(obj, "track_id", 0) or (idx + 1)
        is_primary = target_id == primary_id
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

    if disp:
        disp.show(img)
    frame_id += 1

if gimbal:
    gimbal.close()
print("YOLO SNAIL EGG DETECTOR STOP")
