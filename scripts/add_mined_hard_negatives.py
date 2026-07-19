"""Add model false-positive images to a leak-free YOLO training split."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--mine-buckets", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--repeat", type=int, default=1, help="Explicit training weight per mined image.")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing dataset: {args.output}")
    shutil.copytree(args.base, args.output)
    payload = json.loads(args.source_json.read_text(encoding="utf-8"))
    train_images = args.output / "images" / "train"
    train_labels = args.output / "labels" / "train"
    added = []
    gate = []
    for index, row in enumerate(payload.get("examples", [])):
        source = Path(row["image"])
        if not source.is_absolute():
            source = Path.cwd() / source
        if not source.exists():
            continue
        digest = hashlib.sha1(source.read_bytes()).hexdigest()
        bucket = int(digest[:8], 16) % args.bucket_count
        item = {"source": str(source), "sha1": digest, "bucket": bucket, "detections": row.get("detections", [])}
        if bucket not in args.mine_buckets:
            gate.append(item)
            continue
        targets = []
        for repeat in range(args.repeat):
            stem = f"mined_neg_{index:05d}_r{repeat:02d}_{source.stem}"
            destination = train_images / f"{stem}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            (train_labels / f"{stem}.txt").write_text("", encoding="utf-8")
            targets.append(str(destination))
        item["targets"] = targets
        added.append(item)

    yaml_path = args.output / "pinkeggs_mined.yaml"
    yaml_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: eggs\n"
        % args.output.resolve().as_posix(),
        encoding="utf-8",
    )
    report = {
        "base": str(args.base),
        "output": str(args.output),
        "hash_partition": {"bucket_count": args.bucket_count, "mine_buckets": args.mine_buckets},
        "mined_source_images": len(added),
        "added_training_images": sum(len(item["targets"]) for item in added),
        "gate_source_images": len(gate),
        "items": added,
        "gate": gate,
    }
    (args.output / "mined_negatives.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mined": len(added), "added": report["added_training_images"], "gate": len(gate)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
