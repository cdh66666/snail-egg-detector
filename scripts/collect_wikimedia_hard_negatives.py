from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import requests


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "snail-egg-detector hard-negative dataset builder/1.0"

DEFAULT_QUERIES = [
    "pink flower close up",
    "red flower close up",
    "rose flower close up",
    "hibiscus flower close up",
    "bougainvillea flower",
    "pink petals",
    "red berries",
    "raspberry close up",
    "strawberry close up",
    "pomegranate seeds",
    "fish roe close up",
    "pink beads",
    "red beads",
    "red necklace beads",
    "pink plastic beads",
    "red button close up",
    "red wire",
    "red cable",
    "red motor",
    "red circuit board",
    "red led light",
    "red plastic object",
    "brick wall close up",
    "rusty metal close up",
    "concrete wall texture",
    "wet concrete wall",
    "moss concrete wall",
    "algae wall",
    "wet stone wall",
    "pond wall",
    "drainage ditch concrete",
    "muddy ground",
    "leaf litter",
    "snail shell close up",
]


def commons_search(query: str, limit: int) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "1280",
    }
    response = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    return list(pages.values())


def decode_image(raw: bytes, max_dim: int) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    if min(h, w) < 96:
        return None
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 0.999:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return image


def title_slug(title: str) -> str:
    title = unquote(title).replace("File:", "")
    keep = []
    for ch in title:
        if ch.isalnum():
            keep.append(ch.lower())
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_")[:80] or "image"


def download_one(url: str, max_dim: int) -> tuple[np.ndarray | None, str]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(15, 60))
    response.raise_for_status()
    digest = hashlib.sha1(response.content).hexdigest()
    return decode_image(response.content, max_dim), digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real-world hard negative images from Wikimedia Commons.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/hard_negatives_wikimedia"))
    parser.add_argument("--per-query", type=int, default=35)
    parser.add_argument("--target-count", type=int, default=650)
    parser.add_argument("--max-dim", type=int, default=960)
    parser.add_argument("--sleep", type=float, default=0.15)
    args = parser.parse_args()

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen_hashes = set()

    for query in DEFAULT_QUERIES:
        print(f"query: {query}")
        try:
            pages = commons_search(query, args.per_query)
        except Exception as exc:
            print(f"  search failed: {exc}")
            continue

        for page in pages:
            if len(records) >= args.target_count:
                break
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            mime = info.get("mime", "")
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            try:
                image, digest = download_one(url, args.max_dim)
            except Exception as exc:
                print(f"  download failed: {exc}")
                continue
            if image is None or digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            idx = len(records)
            name = f"{idx:04d}_{title_slug(page.get('title', 'image'))}.jpg"
            out_path = image_dir / name
            ok = cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ok:
                continue

            ext = info.get("extmetadata", {})
            records.append(
                {
                    "file": str(out_path.relative_to(args.output_dir)).replace("\\", "/"),
                    "query": query,
                    "title": page.get("title"),
                    "source_url": info.get("descriptionurl"),
                    "download_url": url,
                    "license": (ext.get("LicenseShortName") or {}).get("value"),
                    "author": (ext.get("Artist") or {}).get("value"),
                    "sha1": digest,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
            )
            print(f"  saved {len(records):04d}/{args.target_count}: {name}")
            time.sleep(args.sleep)

        if len(records) >= args.target_count:
            break

    metadata = {
        "source": "Wikimedia Commons API",
        "queries": DEFAULT_QUERIES,
        "count": len(records),
        "images": records,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} hard negatives to {args.output_dir}")


if __name__ == "__main__":
    main()
