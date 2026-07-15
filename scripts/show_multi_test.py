"""Show a compact rotating multi-target test window without taking the desktop."""

import argparse
import random
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--interval-ms", type=int, default=2500)
    parser.add_argument("--x", type=int, default=20)
    parser.add_argument("--y", type=int, default=20)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    random.Random(20260714).shuffle(paths)
    if not paths:
        raise SystemExit("No multi-target images found")

    root = tk.Tk()
    root.title("multi-target test")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry("420x315+%d+%d" % (args.x, args.y))
    label = tk.Label(root, bg="black", borderwidth=0)
    label.pack(fill="both", expand=True)
    state = {"index": 0, "photo": None}

    def advance():
        image = Image.open(paths[state["index"] % len(paths)]).convert("RGB")
        image.thumbnail((420, 315), Image.Resampling.LANCZOS)
        state["photo"] = ImageTk.PhotoImage(image)
        label.configure(image=state["photo"])
        state["index"] += 1
        root.after(args.interval_ms, advance)

    advance()
    root.after(args.seconds * 1000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
