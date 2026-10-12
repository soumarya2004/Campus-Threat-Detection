import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import yaml

SPLITS=("train", "val", "test")
IMAGE_EXTENSIONS={".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def sha256_file(path: Path)->str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument(
        "--output",
        default="data/processed/dangerous_items_deduplicated",
    )
    args=parser.parse_args()
    config_path=Path(args.data).resolve()
    project_root=Path.cwd().resolve()
    output_root=(project_root/args.output).resolve()
    if not config_path.exists():
        raise SystemExit(f"Dataset config not found: {config_path}")
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root=Path(config.get("path", "."))
    if not source_root.is_absolute():
        source_root=(config_path.parent/source_root).resolve()
    names=config.get("names")
    if not isinstance(names, dict) or not names:
        raise SystemExit("Expected a non-empty class 'names' mapping in data.yaml.")
    if output_root.exists():
        raise SystemExit(
            f"Output already exists; refusing to overwrite it:\n{output_root}"
        )
    #inventory images and confirm that every image has a matching label
    records=[]
    image_hashes=defaultdict(list)
    errors=[]
    for split in SPLITS:
        image_rel=Path(config[split])
        if image_rel.is_absolute():
            image_dir=image_rel
        else:
            image_dir=source_root/image_rel
        label_dir = source_root/"labels"/split
        if not image_dir.is_dir():
            errors.append(f"Image directory missing: {image_dir}")
            continue
        if not label_dir.is_dir():
            errors.append(f"Label directory missing: {label_dir}")
            continue
        images=sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_rel_paths=set()
        for image in images:
            relative=image.relative_to(image_dir)
            image_rel_paths.add(relative.with_suffix(".txt"))
            label=label_dir/relative.with_suffix(".txt")
            if not label.is_file():
                errors.append(f"Missing label for image: {image}")
                continue
            record={
                "split": split,
                "image": image,
                "relative": relative,
                "label": label,
                "hash": sha256_file(image),
            }
            records.append(record)
            image_hashes[record["hash"]].append(record)
        #detect labels that do not correspond to an image
        for label in label_dir.rglob("*.txt"):
            relative=label.relative_to(label_dir)
            if relative not in image_rel_paths:
                errors.append(f"Orphan label: {label}")
    if errors:
        print("Preflight checks failed. No output was created.")
        for error in errors[:100]:
            print(f"[ERROR] {error}")
        print(f"Total errors: {len(errors)}")
        raise SystemExit(1)
    #exclude every image belonging to a hash group present in >1 split
    cross_split_hashes={
        digest
        for digest, group in image_hashes.items()
        if len({record["split"] for record in group}) > 1
    }
    excluded=[
        record for record in records
        if record["hash"] in cross_split_hashes
    ]
    retained=[
        record for record in records
        if record["hash"] not in cross_split_hashes
    ]
    #prepare an audit report before copying
    report={
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_images": len(records),
        "unique_image_hashes": len(image_hashes),
        "cross_split_duplicate_groups": len(cross_split_hashes),
        "excluded_images": len(excluded),
        "retained_images": len(retained),
        "splits": {},
        "excluded_files": [],
    }
    for split in SPLITS:
        split_records=[r for r in records if r["split"] == split]
        kept=[r for r in retained if r["split"] == split]
        removed=[r for r in excluded if r["split"] == split]
        class_counts=Counter()
        malformed_labels=[]
        for record in kept:
            for line_number, line in enumerate(
                record["label"].read_text(encoding="utf-8").splitlines(), 1
            ):
                line=line.strip()
                if not line:
                    continue
                fields=line.split()
                try:
                    class_id=int(fields[0])
                except (ValueError, IndexError):
                    malformed_labels.append(
                        f"{record['label']}:{line_number}"
                    )
                    continue
                class_counts[class_id]+=1
        report["splits"][split]={
            "source_images": len(split_records),
            "retained_images": len(kept),
            "excluded_images": len(removed),
            "class_instances": {
                str(class_id): {
                    "name": str(names.get(class_id, names.get(str(class_id), "unknown"))),
                    "count": count,
                }
                for class_id, count in sorted(class_counts.items())
            },
            "malformed_label_rows": malformed_labels,
        }
    report["excluded_files"]=[
        {
            "split": record["split"],
            "image": str(record["image"]),
            "label": str(record["label"]),
            "sha256": record["hash"],
        }
        for record in excluded
    ]
    #create a fresh output tree and copy retained image/label pairs
    try:
        for split in SPLITS:
            (output_root/"images"/split).mkdir(parents=True)
            (output_root/"labels"/split).mkdir(parents=True)
        for record in retained:
            image_destination=(
                output_root/"images"/record["split"]/record["relative"]
            )
            label_destination=(
                output_root/"labels"/record["split"]
                /record["relative"].with_suffix(".txt")
            )
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["image"], image_destination)
            shutil.copy2(record["label"], label_destination)
        new_config={
            "path": str(output_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        }
        (output_root/"data.yaml").write_text(
            yaml.safe_dump(new_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (output_root/"dedup_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    print("\n--- DEDUPLICATION COMPLETE ---")
    print(f"Source images: {len(records)}")
    print(f"Unique image hashes: {len(image_hashes)}")
    print(f"Cross-split duplicate groups: {len(cross_split_hashes)}")
    print(f"Images excluded: {len(excluded)}")
    print(f"Images retained: {len(retained)}")
    for split, details in report["splits"].items():
        print(
            f"{split}: {details['retained_images']} retained / "
            f"{details['source_images']} source images"
        )
        print(f"  Class instances: {details['class_instances']}")
    print(f"\nNew dataset config: {output_root / 'data.yaml'}")
    print(f"Audit report: {output_root / 'dedup_report.json'}")
    print("Original dataset and existing model weights were not modified.")


if __name__=="__main__":
    main()
