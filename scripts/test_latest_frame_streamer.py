"""Verify that slow streaming drops stale frames instead of building latency."""

import ast
import threading
import time
from pathlib import Path


MAIN = Path("maixcam/main.py")


class SlowStreamer:
    def __init__(self):
        self.frames = []
        self.stopped = False

    def write(self, frame):
        time.sleep(0.02)
        self.frames.append(frame)

    def stop(self):
        self.stopped = True


def load_streamer_class():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    namespace = {"threading": threading, "pytime": time}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LatestJpegStreamer":
            exec(compile(ast.Module([node], type_ignores=[]), str(MAIN), "exec"), namespace)
            return namespace[node.name]
    raise RuntimeError("LatestJpegStreamer not found")


def main():
    latest_streamer = load_streamer_class()
    sink = SlowStreamer()
    publisher = latest_streamer(sink)
    for frame_id in range(100):
        publisher.submit(frame_id)
        time.sleep(0.001)
    time.sleep(0.08)
    publisher.close()

    assert sink.stopped
    assert sink.frames
    assert sink.frames[-1] == 99
    assert len(sink.frames) < 20, sink.frames
    assert sink.frames == sorted(set(sink.frames))
    print({
        "passed": True,
        "submitted": 100,
        "written": len(sink.frames),
        "last_frame": sink.frames[-1],
    })


if __name__ == "__main__":
    main()
