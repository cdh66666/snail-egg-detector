from __future__ import annotations

import cv2
import numpy as np


def pink_ratio_bgr(bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    h, w = bgr.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    if crop.shape[0] > 80 or crop.shape[1] > 80:
        crop = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_AREA)

    b = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    r = crop[:, :, 2].astype(np.int16)

    saturated_pink = (r >= 95) & (b >= 55) & ((r - g) >= 12) & ((b - g) >= -30) & ((r - b) <= 165)
    pale_pink = (r >= 135) & (b >= 105) & (g >= 80) & ((r - g) >= 6) & ((b - g) >= -8)
    red_only_reject = (r >= 120) & ((r - g) >= 35) & (b <= 55)
    mask = (saturated_pink | pale_pink) & ~red_only_reject
    return float(mask.mean())


def pass_laser_safe_filter(
    bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    confidence: float,
    *,
    min_conf: float = 0.62,
    min_area_ratio: float = 0.00004,
    max_area_ratio: float = 0.22,
    min_aspect: float = 0.18,
    max_aspect: float = 5.5,
    min_pink_ratio: float = 0.018,
) -> tuple[bool, float]:
    if confidence < min_conf:
        return False, 0.0
    h, w = bgr.shape[:2]
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return False, 0.0
    area_ratio = (bw * bh) / float(w * h)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False, 0.0
    aspect = bw / float(bh)
    if aspect < min_aspect or aspect > max_aspect:
        return False, 0.0
    ratio = pink_ratio_bgr(bgr, x1, y1, x2, y2)
    return ratio >= min_pink_ratio, ratio
