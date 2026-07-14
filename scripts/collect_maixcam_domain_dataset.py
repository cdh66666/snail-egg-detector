"""Collect labeled monitor-to-MaixCam domain adaptation frames.

The collector keeps every visible window compact. It calibrates a planar
screen-to-camera homography with a small green marker, then projects existing
YOLO labels into raw frames captured by the MaixCam. PPT images are collected
only as a frame-level holdout and are never emitted into train/val.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import paramiko
import tkinter as tk
from PIL import Image, ImageEnhance, ImageOps, ImageTk


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class SourceSample:
    name: str
    image: Image.Image
    box_xyxy: tuple[float, float, float, float] | None
    split: str
    kind: str


class MaixCapture:
    def __init__(self, host: str, username: str, password: str):
        last_error = None
        for attempt in range(6):
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                self.client.connect(
                    host,
                    username=username,
                    password=password,
                    timeout=8,
                    banner_timeout=10,
                    auth_timeout=8,
                )
                break
            except Exception as error:
                last_error = error
                self.client.close()
                if attempt == 5:
                    raise
                time.sleep(2.0)
        if last_error is not None:
            print("SSH connected after retry: %s" % last_error)
        self.sftp = self.client.open_sftp()
        self.debug_dir = "/root/snail_egg/debug"
        self.flag = "/root/snail_egg/capture_once"

    def close(self) -> None:
        self.sftp.close()
        self.client.close()

    def capture(self, output: Path, timeout: float = 8.0) -> Path:
        before = set(self.sftp.listdir(self.debug_dir))
        with self.sftp.open(self.flag, "wb") as handle:
            handle.write(b"")
        deadline = time.monotonic() + timeout
        remote_name = None
        while time.monotonic() < deadline:
            names = set(self.sftp.listdir(self.debug_dir))
            new_names = sorted(
                name for name in names - before if name.startswith("once_") and name.endswith(".jpg")
            )
            if new_names:
                remote_name = new_names[-1]
                break
            time.sleep(0.12)
        if remote_name is None:
            raise TimeoutError("MaixCam capture_once timed out")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(self.debug_dir + "/" + remote_name, str(output))
        return output


def yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cx, cy, bw, bh = map(float, fields[1:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def crop_around_box(
    image: Image.Image, box: tuple[float, float, float, float], margin: float
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(48.0, (x2 - x1) * margin)
    height = max(48.0, (y2 - y1) * margin)
    left = max(0, int(round(cx - width / 2.0)))
    top = max(0, int(round(cy - height / 2.0)))
    right = min(image.width, int(round(cx + width / 2.0)))
    bottom = min(image.height, int(round(cy + height / 2.0)))
    crop = image.crop((left, top, right, bottom))
    mapped = (x1 - left, y1 - top, x2 - left, y2 - top)
    return crop, mapped


def load_yolo_samples(
    dataset_root: Path,
    split: str,
    positives: bool,
    limit: int,
    seed: int,
) -> list[SourceSample]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    candidates: list[SourceSample] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / (image_path.stem + ".txt")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        boxes = yolo_boxes(label_path, image.width, image.height)
        if positives and not boxes:
            continue
        if not positives and boxes:
            continue
        if positives:
            box = max(boxes, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
            # Keep substantial context around the mass. Very tight crops let the
            # detector memorize the monitor's dark image-card boundary.
            crop, mapped = crop_around_box(image, box, 3.2)
            candidates.append(SourceSample(image_path.stem, crop, mapped, split, "positive"))
        else:
            candidates.append(SourceSample(image_path.stem, image, None, split, "negative"))
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def load_pptx_holdout(path: Path, limit: int = 32) -> list[SourceSample]:
    samples = []
    with zipfile.ZipFile(path) as archive:
        media = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ppt/media/") and Path(name).suffix.lower() in IMAGE_SUFFIXES
        )
        for index, name in enumerate(media[:limit]):
            try:
                image = Image.open(io.BytesIO(archive.read(name))).convert("RGB")
            except Exception:
                continue
            # Holdout is scored for frame-level recall. No pseudo box is used for training.
            samples.append(SourceSample("ppt_%02d" % index, image, None, "holdout", "holdout_positive"))
    return samples


def apply_variant(image: Image.Image, brightness: float, contrast: float, saturation: float) -> Image.Image:
    result = ImageEnhance.Brightness(image).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(saturation)
    return result


class CompactDisplay:
    def __init__(self, width: int, height: int):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.geometry("%dx%d+20+20" % (width, height))
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.width = width
        self.height = height
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.photo = None
        self.item = None
        self.image_origin = (0.0, 0.0)
        self.image_size = (0, 0)
        self.window_origin = (20, 20)
        self.root.update()

    def close(self) -> None:
        self.root.destroy()

    def move_center(self, nx: float, ny: float) -> None:
        x = int(max(0, min(self.screen_width - self.width, nx * self.screen_width - self.width / 2)))
        y = int(max(0, min(self.screen_height - self.height, ny * self.screen_height - self.height / 2)))
        self.window_origin = (x, y)
        self.root.geometry("+%d+%d" % (x, y))
        self.root.update()

    def show_marker(self) -> None:
        self.canvas.delete("all")
        self.canvas.configure(bg="#101010")
        half = 36
        cx = self.width // 2
        cy = self.height // 2
        self.canvas.create_rectangle(cx - half, cy - half, cx + half, cy + half, fill="#00ff00", outline="")
        self.root.update()

    def show_image(self, image: Image.Image) -> tuple[float, float, float, float]:
        self.canvas.delete("all")
        self.canvas.configure(bg="#101010")
        fitted = ImageOps.contain(image, (self.width - 8, self.height - 8), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(fitted)
        ox = (self.width - fitted.width) / 2.0
        oy = (self.height - fitted.height) / 2.0
        self.item = self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        self.image_origin = (ox, oy)
        self.image_size = fitted.size
        self.root.update()
        return ox, oy, fitted.width, fitted.height

    def image_box_to_screen(
        self, source_size: tuple[int, int], box: tuple[float, float, float, float]
    ) -> np.ndarray:
        source_w, source_h = source_size
        fitted_w, fitted_h = self.image_size
        sx = fitted_w / float(source_w)
        sy = fitted_h / float(source_h)
        ox, oy = self.image_origin
        wx, wy = self.window_origin
        x1, y1, x2, y2 = box
        return np.array(
            [
                [wx + ox + x1 * sx, wy + oy + y1 * sy],
                [wx + ox + x2 * sx, wy + oy + y1 * sy],
                [wx + ox + x2 * sx, wy + oy + y2 * sy],
                [wx + ox + x1 * sx, wy + oy + y2 * sy],
            ],
            dtype=np.float32,
        )


def find_green_marker(frame_path: Path) -> tuple[float, float] | None:
    frame = cv2.imread(str(frame_path))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([38, 80, 80]), np.array([90, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 25:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def calibrate_homography(
    display: CompactDisplay, capture: MaixCapture, work_dir: Path, settle: float
) -> np.ndarray:
    positions = ((0.30, 0.28), (0.50, 0.28), (0.70, 0.28), (0.30, 0.68), (0.50, 0.68), (0.70, 0.68))
    screen_points = []
    camera_points = []
    display.show_marker()
    for index, (nx, ny) in enumerate(positions):
        display.move_center(nx, ny)
        time.sleep(settle)
        frame_path = capture.capture(work_dir / ("cal_%02d.jpg" % index))
        point = find_green_marker(frame_path)
        if point is None:
            print("CAL_MISS", index, nx, ny, flush=True)
            continue
        wx, wy = display.window_origin
        screen_points.append((wx + display.width / 2.0, wy + display.height / 2.0))
        camera_points.append(point)
        print("CAL", index, screen_points[-1], point, flush=True)
    if len(screen_points) < 4:
        raise RuntimeError("Need at least four visible marker positions for homography")
    matrix, inliers = cv2.findHomography(
        np.asarray(screen_points, dtype=np.float32),
        np.asarray(camera_points, dtype=np.float32),
        cv2.RANSAC,
        4.0,
    )
    if matrix is None or int(inliers.sum()) < 4:
        raise RuntimeError("Homography calibration failed")
    return matrix


def projected_yolo_box(matrix: np.ndarray, points: np.ndarray, frame_w: int, frame_h: int) -> str | None:
    mapped = cv2.perspectiveTransform(points.reshape(1, -1, 2), matrix)[0]
    x1 = max(0.0, float(mapped[:, 0].min()))
    y1 = max(0.0, float(mapped[:, 1].min()))
    x2 = min(float(frame_w), float(mapped[:, 0].max()))
    y2 = min(float(frame_h), float(mapped[:, 1].max()))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    cx = (x1 + x2) / 2.0 / frame_w
    cy = (y1 + y2) / 2.0 / frame_h
    bw = (x2 - x1) / frame_w
    bh = (y2 - y1) / frame_h
    return "0 %.6f %.6f %.6f %.6f\n" % (cx, cy, bw, bh)


def collect_sample(
    sample: SourceSample,
    variant_index: int,
    variant: tuple[float, float, float],
    position: tuple[float, float],
    display: CompactDisplay,
    capture: MaixCapture,
    homography: np.ndarray,
    output_root: Path,
    manifest: csv.writer,
    settle: float,
) -> None:
    brightness, contrast, saturation = variant
    shown = apply_variant(sample.image, brightness, contrast, saturation)
    display.move_center(*position)
    display.show_image(shown)
    time.sleep(settle)
    stem = "%s_%s_v%02d" % (sample.kind, sample.name, variant_index)
    image_path = output_root / "images" / sample.split / (stem + ".jpg")
    label_path = output_root / "labels" / sample.split / (stem + ".txt")
    capture.capture(image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label = ""
    if sample.box_xyxy is not None:
        screen_box = display.image_box_to_screen(sample.image.size, sample.box_xyxy)
        label = projected_yolo_box(homography, screen_box, 640, 480) or ""
    label_path.write_text(label, encoding="ascii")
    manifest.writerow(
        [sample.split, sample.kind, stem, brightness, contrast, saturation, position[0], position[1], bool(label)]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.10.110")
    parser.add_argument("--username", default="root")
    parser.add_argument("--password", default="root")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pptx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-positive", type=int, default=36)
    parser.add_argument("--train-negative", type=int, default=48)
    parser.add_argument("--val-positive", type=int, default=12)
    parser.add_argument("--val-negative", type=int, default=18)
    parser.add_argument("--settle", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    work_dir = args.output / "calibration"
    display = CompactDisplay(420, 340)
    capture = MaixCapture(args.host, args.username, args.password)
    manifest_handle = (args.output / "manifest.csv").open("w", newline="", encoding="utf-8")
    manifest = csv.writer(manifest_handle)
    manifest.writerow(
        ["split", "kind", "stem", "brightness", "contrast", "saturation", "screen_x", "screen_y", "has_label"]
    )
    try:
        matrix = calibrate_homography(display, capture, work_dir, args.settle)
        np.savetxt(args.output / "screen_to_camera_homography.txt", matrix)
        samples = []
        samples.extend(load_yolo_samples(args.dataset, "train", True, args.train_positive, args.seed))
        samples.extend(load_yolo_samples(args.dataset, "train", False, args.train_negative, args.seed + 1))
        samples.extend(load_yolo_samples(args.dataset, "val", True, args.val_positive, args.seed + 2))
        samples.extend(load_yolo_samples(args.dataset, "val", False, args.val_negative, args.seed + 3))
        # PPT holdout remains frame-level and is never merged into train/val.
        samples.extend(load_pptx_holdout(args.pptx))
        variants = (
            (1.00, 1.00, 1.00),
            (0.72, 0.86, 0.76),
            (1.24, 0.72, 0.58),
        )
        positions = ((0.46, 0.48), (0.54, 0.48), (0.50, 0.58))
        for sample_index, sample in enumerate(samples):
            variant_count = 2 if sample.kind == "negative" else 3
            for variant_index in range(variant_count):
                variant = variants[(sample_index + variant_index) % len(variants)]
                position = positions[(sample_index * 2 + variant_index) % len(positions)]
                collect_sample(
                    sample,
                    variant_index,
                    variant,
                    position,
                    display,
                    capture,
                    matrix,
                    args.output,
                    manifest,
                    args.settle,
                )
                print("COLLECT", sample.split, sample.kind, sample.name, variant_index, flush=True)
    finally:
        manifest_handle.close()
        capture.close()
        display.close()


if __name__ == "__main__":
    main()
