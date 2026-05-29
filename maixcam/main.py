import os

from maix import camera, display, image, nn, app, time


# Copy this file to MaixVision as main.py after converting the ONNX model to MUD.
# Put the MUD and model file under /root/models/ on the MaixCam.

MODEL = "/root/models/snail_eggs_yolov8n_640x480.mud"
CONF_TH = 0.25
IOU_TH = 0.35
MAX_TARGETS = 32
MIN_MODEL_CONF = 0.25
STRONG_MODEL_CONF = 0.50
LOW_CONF_MIN_PINK_RATIO = 0.03
LOW_CONF_MIN_AREA = 24

# 0 = raw YOLO debug: draw every model detection, no color filtering.
# 1 = color debug: raw boxes are yellow, pink-passed boxes are green.
# 2 = final: only draw pink-passed targets with IDs.
# 3 = auto tune: color debug + save frames + print every raw/candidate stat.
RUN_MODE = 2

# Runtime speed profiles:
# - "full_frame": 640x480 model, one inference per camera frame, no memory boxes.
# - "accuracy": legacy 320x320 model, scan all 6 tiles every frame.
# - "balanced": legacy 320x320 model, scan 3 tiles per frame with short memory.
# - "fast": legacy 320x320 model, scan 2 tiles per frame.
# Keep the camera at 640x480 so aiming coordinates stay in the real view.
SPEED_PROFILE = "full_frame"

# The current deployed model input is 640x480, so the detector sees the whole
# camera image in one pass. That avoids stale round-robin boxes for laser aiming.
FRAME_W = 640
FRAME_H = 480
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
MAX_BOX_AREA_RATIO = 0.12
MAX_BOX_SIDE_RATIO = 0.52
MIN_ASPECT = 0.18
MAX_ASPECT = 5.5
ENABLE_COLOR_GATE = True
MIN_PINK_RATIO = 0.03
MIN_PINK_PIXELS = 1
MAX_RED_BAD_RATIO = 0.55
RED_BAD_DOMINANCE = 2.2
COLOR_GRID = 6
MAX_COLOR_CHECKS = 36
REQUIRE_STABLE_FRAMES = 2
TRACK_MAX_MISSES = 1
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
DEBUG_SAVE_EVERY_N_FRAMES = 15
DEBUG_MAX_SAVES = 6
# Default to visual output for MaixVision/device use. The VSCode SSH helper
# creates /root/snail_egg/headless before running so automated tests do not
# block on display.Display().
ENABLE_DISPLAY = True
HEADLESS_FLAG = "/root/snail_egg/headless"
_tracks = []
_tile_scan_index = 0
_memory = []
_last_tile_info = "0/0"


class DetectedBox:
    def __init__(self, x, y, w, h, score):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.score = float(score)


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
        score_ok = obj.score >= MIN_MODEL_CONF
        if score_ok and obj.score >= STRONG_MODEL_CONF:
            color_ok = not red_reject
        else:
            color_ok = ok and pink_ratio >= LOW_CONF_MIN_PINK_RATIO and area >= LOW_CONF_MIN_AREA
        if not color_ok or not score_ok:
            continue
        kept.append(obj)
    return kept, best_pink_ratio, best_red_ratio, rows


def update_tracks(objs):
    global _tracks
    next_tracks = []
    used = set()
    for obj in objs:
        box = (int(obj.x), int(obj.y), int(obj.w), int(obj.h))
        best_idx = -1
        best_score = 0
        for idx, tr in enumerate(_tracks):
            if idx in used:
                continue
            score = box_iou(box, tr["box"])
            if score > best_score:
                best_score = score
                best_idx = idx
        stable = 1
        if best_idx >= 0 and best_score >= 0.25:
            used.add(best_idx)
            stable = _tracks[best_idx]["stable"] + 1
        next_tracks.append({"box": box, "obj": obj, "stable": stable, "misses": 0})

    for idx, tr in enumerate(_tracks):
        if idx in used:
            continue
        if tr["misses"] < TRACK_MAX_MISSES:
            tr["misses"] += 1
            next_tracks.append(tr)

    _tracks = next_tracks
    return [tr["obj"] for tr in _tracks if tr["stable"] >= REQUIRE_STABLE_FRAMES and tr["misses"] == 0]


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
        objs = detector.detect(tile, conf_th=CONF_TH, iou_th=IOU_TH)
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
        CONF_TH,
        MIN_MODEL_CONF,
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
cam = camera.Camera(FRAME_W, FRAME_H, detector.input_format())
print("INIT,CAMERA_OK")
if ENABLE_DISPLAY:
    disp = display.Display()
    print("INIT,DISPLAY_OK")
else:
    disp = None
    print("INIT,DISPLAY_SKIP")
frame_id = 0
debug_saved = 0
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
    if RUN_MODE == 3 and frame_id < 3:
        print("TRACE,%d,DETECT_OK,%d" % (frame_id, raw_count))
    stable = update_tracks(candidates)
    targets = [] if frame_id < WARMUP_FRAMES else sort_targets(stable, FRAME_H)
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
    img.draw_string(2, 18, "M%d RAW %d CAND %d %s %d" % (RUN_MODE, raw_count, len(candidates), status_label, len(targets)),
                    image.COLOR_GREEN if targets else image.COLOR_RED)
    img.draw_string(
        2,
        34,
        "PINK %.2f RED %.2f T %s" % (best_pink_ratio, best_red_ratio, _last_tile_info),
        image.COLOR_GREEN if candidates else image.COLOR_RED,
    )

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
        x = int(obj.x)
        y = int(obj.y)
        w = int(obj.w)
        h = int(obj.h)
        img.draw_rect(x, y, w, h, color=image.COLOR_GREEN, thickness=2)
        draw_cross(img, cx, cy, image.COLOR_RED)
        label_y = y - 14 if y >= 14 else y
        img.draw_string(x, label_y, str(target_id), image.COLOR_GREEN)

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

print("YOLO SNAIL EGG DETECTOR STOP")
