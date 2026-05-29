from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import numpy as np
import requests


DEFAULT_QUERIES = [
    "pink flower macro",
    "red flower macro",
    "rose petals close up",
    "hibiscus flower close up",
    "bougainvillea close up",
    "pink blossom close up",
    "raspberry close up",
    "strawberry close up",
    "pomegranate seeds close up",
    "salmon roe close up",
    "fish eggs close up",
    "pink beads close up",
    "red beads close up",
    "pink plastic balls close up",
    "red button close up",
    "red wire close up",
    "red cable close up",
    "red motor close up",
    "red circuit board close up",
    "red led close up",
    "red plastic close up",
    "brick wall texture",
    "rusty metal texture",
    "concrete wall texture",
    "wet concrete wall",
    "moss concrete wall",
    "algae concrete wall",
    "wet stone wall texture",
    "pond wall concrete",
    "drainage ditch concrete",
    "muddy ground texture",
    "leaf litter close up",
    "snail shell close up",
    "small pink toy close up",
    "pink foam close up",
    "red rubber close up",
]


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}


def parse_bing_images(query: str, first: int, count: int) -> list[dict]:
    url = f"https://www.bing.com/images/async?q={quote_plus(query)}&async=1&first={first}&count={count}"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    records = []
    for raw in re.findall(r'm="([^"]+)"', response.text):
        try:
            data = json.loads(html.unescape(raw))
        except Exception:
            continue
        if data.get("murl") or data.get("turl"):
            records.append(data)
    return records


def decode(raw: bytes, max_dim: int) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    if min(h, w) < 80:
        return None
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 0.999:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return image


def download_image(record: dict, max_dim: int) -> tuple[np.ndarray | None, str, str | None]:
    last_error = None
    for key in ("murl", "turl"):
        url = record.get(key)
        if not url:
            continue
        try:
            response = requests.get(url, headers=HEADERS, timeout=(10, 40))
            response.raise_for_status()
            digest = hashlib.sha1(response.content).hexdigest()
            image = decode(response.content, max_dim)
            if image is not None:
                return image, digest, url
        except Exception as exc:
            last_error = str(exc)
    return None, "", last_error


def clean_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch.lower())
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_")[:80] or "image"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Bing image-search hard negatives for local training.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/hard_negatives_bing_v2"))
    parser.add_argument("--target-count", type=int, default=900)
    parser.add_argument("--per-query", type=int, default=80)
    parser.add_argument("--page-size", type=int, default=35)
    parser.add_argument("--max-dim", type=int, default=960)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()

    for query in DEFAULT_QUERIES:
        print(f"query: {query}")
        first = 1
        query_saved = 0
        while query_saved < args.per_query and len(records) < args.target_count:
            try:
                results = parse_bing_images(query, first=first, count=args.page_size)
            except Exception as exc:
                print(f"  search failed: {exc}")
                break
            if not results:
                break
            for result in results:
                if query_saved >= args.per_query or len(records) >= args.target_count:
                    break
                image, digest, used_url = download_image(result, args.max_dim)
                if image is None or digest in seen:
                    continue
                seen.add(digest)
                idx = len(records)
                name = f"{idx:04d}_{clean_name(query)}_{clean_name(result.get('t', 'image'))}.jpg"
                out_path = image_dir / name
                if not cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                    continue
                records.append(
                    {
                        "file": str(out_path.relative_to(args.output_dir)).replace("\\", "/"),
                        "query": query,
                        "title": result.get("t"),
                        "page_url": result.get("purl"),
                        "image_url": used_url,
                        "sha1": digest,
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                    }
                )
                query_saved += 1
                print(f"  saved {len(records):04d}/{args.target_count}: {name}")
                time.sleep(args.sleep)
            first += args.page_size
        if len(records) >= args.target_count:
            break

    metadata = {
        "source": "Bing Images search results",
        "note": "Downloaded only for local hard-negative training/evaluation; not intended for redistribution.",
        "queries": DEFAULT_QUERIES,
        "count": len(records),
        "images": records,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} hard negatives to {args.output_dir}")


if __name__ == "__main__":
    main()
