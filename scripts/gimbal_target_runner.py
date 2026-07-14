"""Compact, repeatable visual target for MaixCam gimbal tuning and capture.

The runner reads raster images from a PPTX file or image directory, moves one
image over deterministic trajectories, and records the requested screen
position to CSV.
It uses only Tkinter and Pillow so it can run on a normal Windows training PC.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from PIL import Image, ImageEnhance, ImageOps, ImageTk


PHASES = (
    ("center", 10.0),
    ("horizontal", 20.0),
    ("vertical", 20.0),
    ("rectangle", 24.0),
    ("smooth_random", 60.0),
)


@dataclass
class TargetFrame:
    name: str
    image: Image.Image


def load_pptx_images(path: Path) -> list[TargetFrame]:
    frames: list[TargetFrame] = []
    with zipfile.ZipFile(path) as archive:
        media = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ppt/media/")
            and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for name in media:
            try:
                image = Image.open(io.BytesIO(archive.read(name))).convert("RGB")
                if image.width >= 320 and image.height >= 240:
                    frames.append(TargetFrame(Path(name).name, image))
            except Exception as exc:
                print("SKIP,%s,%s" % (name, exc))
    if not frames:
        raise RuntimeError("No usable raster images found in %s" % path)
    return frames


def load_image_dir(path: Path) -> list[TargetFrame]:
    frames: list[TargetFrame] = []
    for image_path in sorted(path.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            if image.width >= 64 and image.height >= 64:
                frames.append(TargetFrame(image_path.name, image))
        except Exception as exc:
            print("SKIP,%s,%s" % (image_path, exc))
    if not frames:
        raise RuntimeError("No usable raster images found in %s" % path)
    return frames


def crop_frames_to_detections(
    frames: list[TargetFrame], model_path: Path, margin: float
) -> list[TargetFrame]:
    """Use the project YOLO model once on PC to make compact pet targets."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    cropped: list[TargetFrame] = []
    for frame in frames:
        results = model.predict(
            source=np.asarray(frame.image),
            imgsz=640,
            conf=0.10,
            iou=0.35,
            device="cpu",
            verbose=False,
        )
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            cropped.append(frame)
            print("CROP_MISS,%s" % frame.name, flush=True)
            continue
        scores = boxes.conf.cpu().numpy()
        best_index = int(scores.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_index].cpu().numpy().tolist()
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        width = max(32.0, (x2 - x1) * margin)
        height = max(32.0, (y2 - y1) * margin)
        crop_x1 = max(0, int(cx - width * 0.5))
        crop_y1 = max(0, int(cy - height * 0.5))
        crop_x2 = min(frame.image.width, int(cx + width * 0.5))
        crop_y2 = min(frame.image.height, int(cy + height * 0.5))
        if crop_x2 - crop_x1 < 16 or crop_y2 - crop_y1 < 16:
            cropped.append(frame)
            continue
        image = frame.image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        cropped.append(TargetFrame(frame.name, image))
        print(
            "CROP,%s,CONF,%.3f,BOX,%d,%d,%d,%d"
            % (frame.name, float(scores[best_index]), crop_x1, crop_y1, crop_x2, crop_y2),
            flush=True,
        )
    return cropped


class TargetRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.frames = load_pptx_images(args.pptx) if args.pptx else load_image_dir(args.image_dir)
        if args.model is not None:
            self.frames = crop_frames_to_detections(self.frames, args.model, args.crop_margin)
        self.random = random.Random(args.seed)
        self.root = tk.Tk()
        self.root.title("MaixCam Gimbal Target")
        self.root.configure(bg="#202020")
        self.pet_mode = not args.fullscreen
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        if args.fullscreen:
            self.root.attributes("-fullscreen", True)
            self.root.attributes("-topmost", True)
        else:
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
            self.root.attributes("-alpha", 0.97)
            self.root.geometry("%dx%d+30+30" % (args.pet_width, args.pet_height))

        self.canvas = tk.Canvas(self.root, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()
        if self.pet_mode:
            self.width = args.pet_width
            self.height = args.pet_height
        else:
            self.width = max(640, self.root.winfo_width())
            self.height = max(480, self.root.winfo_height())
        self.margin_x = self.width * 0.10
        self.margin_y = self.height * 0.12

        self.started = time.perf_counter()
        self.last_tick = self.started
        self.paused = False
        self.pause_started = 0.0
        self.pause_total = 0.0
        self.current_frame_index = args.image_index % len(self.frames)
        self.initial_frame_index = self.current_frame_index
        self.last_image_slot = 0
        self.current_mode = "center"
        self.phase_start = 0.0
        self.random_position = [args.center_x, args.center_y]
        self.random_goal = [args.center_x, args.center_y]
        self.next_waypoint = 0.0
        self.photo = None
        self.item = None
        self.image_size = (0, 0)
        self.last_log = 0.0

        args.log.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = args.log.open("w", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_handle)
        self.log_writer.writerow(
            ("elapsed_s", "mode", "image", "center_x_norm", "center_y_norm", "screen_x", "screen_y", "image_w", "image_h")
        )

        self.status_item = self.canvas.create_text(
            self.width - 18,
            18,
            anchor="ne",
            fill="#ffffff",
            font=("Arial", 16, "bold"),
            text="",
            state="hidden" if self.pet_mode else "normal",
        )
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("<space>", lambda _event: self.toggle_pause())
        self.root.bind("1", lambda _event: self.force_mode("center"))
        self.root.bind("2", lambda _event: self.force_mode("horizontal"))
        self.root.bind("3", lambda _event: self.force_mode("vertical"))
        self.root.bind("4", lambda _event: self.force_mode("rectangle"))
        self.root.bind("5", lambda _event: self.force_mode("smooth_random"))
        self.root.bind("<Button-3>", lambda _event: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.set_frame(self.current_frame_index)

    def set_frame(self, index: int):
        self.current_frame_index = index % len(self.frames)
        frame = self.frames[self.current_frame_index]
        max_w = int(self.width * (0.98 if self.pet_mode else self.args.target_width))
        max_h = int(self.height * (0.98 if self.pet_mode else self.args.target_height))
        fitted = ImageOps.contain(frame.image, (max_w, max_h), Image.Resampling.LANCZOS)
        if self.args.saturation != 1.0:
            fitted = ImageEnhance.Color(fitted).enhance(self.args.saturation)
        if self.args.contrast != 1.0:
            fitted = ImageEnhance.Contrast(fitted).enhance(self.args.contrast)
        if self.args.brightness != 1.0:
            fitted = ImageEnhance.Brightness(fitted).enhance(self.args.brightness)
        self.photo = ImageTk.PhotoImage(fitted)
        self.image_size = fitted.size
        if self.item is None:
            self.item = self.canvas.create_image(self.width / 2, self.height / 2, image=self.photo)
        else:
            self.canvas.itemconfigure(self.item, image=self.photo)
        print("IMAGE,%s,%dx%d" % (frame.name, fitted.width, fitted.height), flush=True)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started - self.pause_total

    def phase(self, elapsed: float) -> tuple[str, float, float]:
        if self.args.mode != "sequence":
            return self.args.mode, elapsed, float("inf")
        cursor = 0.0
        for mode, duration in PHASES:
            if elapsed < cursor + duration:
                return mode, elapsed - cursor, duration
            cursor += duration
        return "smooth_random", elapsed - cursor, float("inf")

    def force_mode(self, mode: str):
        self.args.mode = mode
        self.phase_start = self.elapsed()
        self.current_mode = mode
        self.random_position = [self.args.center_x, self.args.center_y]
        self.random_goal = [self.args.center_x, self.args.center_y]
        self.next_waypoint = 0.0

    def toggle_pause(self):
        if self.paused:
            self.pause_total += time.perf_counter() - self.pause_started
            self.paused = False
        else:
            self.pause_started = time.perf_counter()
            self.paused = True

    @staticmethod
    def rectangle_position(t: float) -> tuple[float, float]:
        section = (t / 6.0) % 4.0
        edge = section - int(section)
        if section < 1.0:
            return 0.25 + 0.50 * edge, 0.30
        if section < 2.0:
            return 0.75, 0.30 + 0.40 * edge
        if section < 3.0:
            return 0.75 - 0.50 * edge, 0.70
        return 0.25, 0.70 - 0.40 * edge

    def desired_position(self, mode: str, phase_t: float, dt: float) -> tuple[float, float]:
        if mode == "center":
            return self.args.center_x, self.args.center_y
        if mode == "horizontal":
            return self.args.center_x + self.args.amplitude_x * math.sin(
                2.0 * math.pi * phase_t / self.args.period_x
            ), self.args.center_y
        if mode == "vertical":
            return self.args.center_x, self.args.center_y + self.args.amplitude_y * math.sin(
                2.0 * math.pi * phase_t / self.args.period_y
            )
        if mode == "rectangle":
            x, y = self.rectangle_position(phase_t)
            return self.args.center_x + (x - 0.5), self.args.center_y + (y - 0.5)

        now = self.elapsed()
        if now >= self.next_waypoint:
            self.random_goal = [
                self.random.uniform(self.args.center_x - 0.25, self.args.center_x + 0.25),
                self.random.uniform(self.args.center_y - 0.22, self.args.center_y + 0.22),
            ]
            self.next_waypoint = now + self.random.uniform(4.5, 7.0)
            if self.args.cycle_images:
                self.set_frame(self.current_frame_index + 1)
        alpha = 1.0 - math.exp(-max(0.0, dt) / max(0.05, self.args.smoothing))
        self.random_position[0] += (self.random_goal[0] - self.random_position[0]) * alpha
        self.random_position[1] += (self.random_goal[1] - self.random_position[1]) * alpha
        return self.random_position[0], self.random_position[1]

    def tick(self):
        if self.paused:
            self.root.after(33, self.tick)
            return
        now = time.perf_counter()
        dt = min(0.1, now - self.last_tick)
        self.last_tick = now
        elapsed = self.elapsed()
        if self.args.image_period > 0:
            image_slot = int(elapsed / self.args.image_period)
            if image_slot != self.last_image_slot:
                self.last_image_slot = image_slot
                self.set_frame(self.initial_frame_index + image_slot)
        mode, phase_t, _duration = self.phase(elapsed)
        if mode != self.current_mode:
            self.current_mode = mode
            self.random_position = [self.args.center_x, self.args.center_y]
            self.random_goal = [self.args.center_x, self.args.center_y]
            self.next_waypoint = elapsed
            print("PHASE,%s,%.3f" % (mode, elapsed), flush=True)

        nx, ny = self.desired_position(mode, phase_t, dt)
        if self.pet_mode:
            center_x = max(self.width / 2, min(self.screen_width - self.width / 2, nx * self.screen_width))
            center_y = max(self.height / 2, min(self.screen_height - self.height / 2, ny * self.screen_height))
            window_x = int(center_x - self.width / 2)
            window_y = int(center_y - self.height / 2)
            self.root.geometry("+%d+%d" % (window_x, window_y))
            self.canvas.coords(self.item, self.width / 2, self.height / 2)
            x, y = center_x, center_y
        else:
            x = self.margin_x + nx * (self.width - 2.0 * self.margin_x)
            y = self.margin_y + ny * (self.height - 2.0 * self.margin_y)
            self.canvas.coords(self.item, x, y)
            self.canvas.itemconfigure(
                self.status_item,
                text="%s  %s" % (mode.upper(), self.frames[self.current_frame_index].name),
            )

        if elapsed - self.last_log >= 0.1:
            self.last_log = elapsed
            self.log_writer.writerow(
                ("%.3f" % elapsed, mode, self.frames[self.current_frame_index].name, "%.5f" % nx, "%.5f" % ny,
                 int(x), int(y), self.image_size[0], self.image_size[1])
            )
            self.log_handle.flush()
        if self.args.duration > 0 and elapsed >= self.args.duration:
            self.close()
            return
        self.root.after(33, self.tick)

    def close(self):
        try:
            self.log_handle.close()
        finally:
            self.root.destroy()

    def run(self):
        print("TARGET_RUNNER,READY,IMAGES,%d,LOG,%s" % (len(self.frames), self.args.log), flush=True)
        self.tick()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeatable desktop-pet target for MaixCam gimbal tuning")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pptx", type=Path)
    source.add_argument("--image-dir", type=Path)
    parser.add_argument("--mode", choices=("sequence", "center", "horizontal", "vertical", "rectangle", "smooth_random"), default="sequence")
    parser.add_argument("--duration", type=float, default=0.0, help="0 keeps running after the sequence")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--cycle-images", action="store_true")
    parser.add_argument("--image-period", type=float, default=0.0, help="Cycle images at this interval; 0 disables timed cycling")
    parser.add_argument("--model", type=Path, help="Optional YOLO model used to crop each PPT image around its best egg detection")
    parser.add_argument("--crop-margin", type=float, default=1.8)
    parser.add_argument("--saturation", type=float, default=1.0, help="Display-only color compensation")
    parser.add_argument("--contrast", type=float, default=1.0, help="Display-only contrast compensation")
    parser.add_argument("--brightness", type=float, default=1.0, help="Display-only brightness compensation")
    parser.add_argument("--target-width", type=float, default=0.46)
    parser.add_argument("--target-height", type=float, default=0.52)
    parser.add_argument("--center-x", type=float, default=0.5)
    parser.add_argument("--center-y", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=1.4)
    parser.add_argument("--amplitude-x", type=float, default=0.27)
    parser.add_argument("--amplitude-y", type=float, default=0.22)
    parser.add_argument("--period-x", type=float, default=10.0)
    parser.add_argument("--period-y", type=float, default=10.0)
    parser.add_argument("--pet-width", type=int, default=520)
    parser.add_argument("--pet-height", type=int, default=420)
    parser.add_argument("--fullscreen", action="store_true", help="Use only for unattended lab tests")
    default_log = Path("runs/gimbal_target/target_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--log", type=Path, default=default_log)
    args = parser.parse_args()
    if args.pptx is not None and not args.pptx.is_file():
        parser.error("PPTX does not exist: %s" % args.pptx)
    if args.image_dir is not None and not args.image_dir.is_dir():
        parser.error("image directory does not exist: %s" % args.image_dir)
    if args.model is not None and not args.model.is_file():
        parser.error("model does not exist: %s" % args.model)
    if args.period_x <= 0.0 or args.period_y <= 0.0:
        parser.error("--period-x and --period-y must be positive")
    return args


if __name__ == "__main__":
    TargetRunner(parse_args()).run()
