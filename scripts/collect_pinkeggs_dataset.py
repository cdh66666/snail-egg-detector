from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import cv2
import numpy as np
import requests


DATASET_PAGE = "https://datasetninja.com/pink-eggs-dataset-v1"


def find_tar_url(source: str) -> str:
    html = requests.get(DATASET_PAGE, timeout=30).text
    if source == "sample":
        pattern = r'"(https://assets\.supervisely\.com/supervisely-supervisely-assets-public/[^"]+?\.tar)"'
    else:
        pattern = r'(https://assets\.supervisely\.com/remote/[^"\\)]+)'
    match = re.search(pattern, html)
    if not match:
        raise RuntimeError(f"Could not find Dataset Ninja {source} tar URL.")
    return match.group(1)


def parse_ann(raw: bytes) -> dict:
    data = json.loads(raw.decode("utf-8"))
    boxes = []
    for obj in data.get("objects", []):
        if obj.get("classTitle") != "eggs":
            continue
        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) != 2:
            continue
        (x1, y1), (x2, y2) = exterior
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return {"size": data.get("size", {}), "boxes": boxes}


def resize_image_and_boxes(raw: bytes, boxes: list[list[int]], max_dim: int) -> tuple[np.ndarray, list[list[int]]]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image.")

    h, w = image.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 0.999:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h

    scaled_boxes = []
    for x1, y1, x2, y2 in boxes:
        scaled_boxes.append(
            [
                int(round(x1 * new_w / w)),
                int(round(y1 * new_h / h)),
                int(round(x2 * new_w / w)),
                int(round(y2 * new_h / h)),
            ]
        )
    return image, scaled_boxes


def collect(limit: int, output_dir: Path, max_dim: int, source: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations: dict[str, dict] = {}
    records = []
    tar_url = find_tar_url(source)

    with requests.get(tar_url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        tar = tarfile.open(fileobj=response.raw, mode="r|")
        for member in tar:
            if member.isdir():
                continue
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            raw = file_obj.read()

            name = member.name.replace("\\", "/")
            if "/ann/" in name and name.endswith(".json"):
                image_name = Path(name.removesuffix(".json")).name
                ann = parse_ann(raw)
                if ann["boxes"]:
                    annotations[image_name.lower()] = ann
                continue

            if "/img/" not in name:
                continue
            image_name = Path(name).name
            ann = annotations.get(image_name.lower())
            if not ann or not ann["boxes"]:
                continue

            image, boxes = resize_image_and_boxes(raw, ann["boxes"], max_dim)
            safe_name = f"{len(records):03d}_{image_name}"
            output_path = images_dir / safe_name
            if not cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                raise RuntimeError(f"Failed to write {output_path}")
            records.append(
                {
                    "file": str(output_path.relative_to(output_dir)).replace("\\", "/"),
                    "source_member": name,
                    "boxes": boxes,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
            )
            print(f"[{len(records):03d}/{limit}] {safe_name} boxes={len(boxes)}")
            if len(records) >= limit:
                break

    (output_dir / "annotations.json").write_text(
        json.dumps(
            {
                "source": DATASET_PAGE,
                "source_tar_url": tar_url,
                "source_tar_kind": source,
                "license": "GNU GPL 2.0",
                "count": len(records),
                "images": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} images to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pinkeggs_100"))
    parser.add_argument("--max-dim", type=int, default=1280)
    parser.add_argument("--source", choices=["sample", "full"], default="full")
    args = parser.parse_args()
    collect(args.limit, args.output_dir, args.max_dim, args.source)


if __name__ == "__main__":
    main()
