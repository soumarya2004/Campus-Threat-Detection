from __future__ import annotations
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import yaml

IMAGE_EXTENSIONS={".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def sha256_file(path: Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main()->int:
    parser=argparse.ArgumentParser(description="Read-only YOLO dataset audit")
    parser.add_argument("--data", default="data.yaml", help="Dataset YAML path")
    args=parser.parse_args()
    config_path=Path(args.data).resolve()
    with config_path.open("r", encoding="utf-8-sig") as file:
        config=yaml.safe_load(file)
    if not isinstance(config, dict) or "names" not in config:
        raise ValueError("Dataset YAML must contain a 'names' mapping or list.")
    dataset_root=Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root=(config_path.parent/dataset_root).resolve()
    names=config["names"]
    if isinstance(names, list):
        class_names={i: str(name) for i, name in enumerate(names)}
    else:
        class_names={int(key): str(value) for key, value in names.items()}
    class_counts: dict[str, Counter]=defaultdict(Counter)
    image_hashes: dict[str, list[str]]=defaultdict(list)
    problems: list[str]=[]
    totals=Counter()
    print(f"Config: {config_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {class_names}")
    for split in ("train", "val", "test"):
        if split not in config:
            print(f"\n[SKIP] Split '{split}' is not configured.")
            continue
        image_dir=dataset_root/str(config[split])
        if not image_dir.is_dir():
            problems.append(f"{split}: image directory missing: {image_dir}")
            continue
        label_dir=dataset_root/"labels"/split
        images=sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels=sorted(label_dir.rglob("*.txt")) if label_dir.is_dir() else []

        #relative stems allow labels in nested folders to match images
        image_by_key={
            path.relative_to(image_dir).with_suffix("").as_posix(): path
            for path in images
        }
        label_by_key={
            path.relative_to(label_dir).with_suffix("").as_posix(): path
            for path in labels
        } if label_dir.is_dir() else {}
        missing_labels=sorted(set(image_by_key)-set(label_by_key))
        orphan_labels=sorted(set(label_by_key)-set(image_by_key))
        totals[f"{split}_images"]=len(images)
        totals[f"{split}_label_files"]=len(labels)
        totals[f"{split}_missing_labels"]=len(missing_labels)
        totals[f"{split}_orphan_labels"]=len(orphan_labels)
        print(f"\n--- {split.upper()} ---")
        print(f"Images: {len(images)} | Label files: {len(labels)}")
        print(f"Missing labels: {len(missing_labels)}")
        print(f"Orphan labels: {len(orphan_labels)}")
        for key in missing_labels[:20]:
            problems.append(f"{split}: missing label for {key}")
        for key in orphan_labels[:20]:
            problems.append(f"{split}: label has no matching image: {key}")
        for image_path in images:
            try:
                image=cv2.imread(str(image_path))
                if image is None:
                    problems.append(f"{split}: unreadable image: {image_path}")
                    totals["unreadable_images"]+=1
                    continue
                height, width=image.shape[:2]
                if width<=0 or height<=0:
                    problems.append(f"{split}: invalid image dimensions: {image_path}")
                    continue
                digest=sha256_file(image_path)
                image_hashes[digest].append(f"{split}:{image_path}")
                label_path=label_by_key.get(
                    image_path.relative_to(image_dir).with_suffix("").as_posix()
                )
                if label_path is None:
                    continue
                lines=label_path.read_text(encoding="utf-8-sig").splitlines()
                for line_number, line in enumerate(lines, start=1):
                    if not line.strip():
                        continue
                    fields=line.split()
                    if len(fields)!=5:
                        problems.append(
                            f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}"
                        )
                        totals["malformed_annotations"]+=1
                        continue
                    try:
                        class_value=float(fields[0])
                        coords=[float(value) for value in fields[1:]]
                    except ValueError:
                        problems.append(
                            f"{label_path}:{line_number}: non-numeric annotation"
                        )
                        totals["malformed_annotations"]+=1
                        continue
                    if(
                        not class_value.is_integer()
                        or int(class_value) not in class_names
                    ):
                        problems.append(
                            f"{label_path}:{line_number}: unknown class ID {fields[0]}"
                        )
                        totals["invalid_class_ids"]+=1
                        continue
                    class_id=int(class_value)
                    x_center, y_center, box_width, box_height = coords
                    if not all(0.0<=value<=1.0 for value in coords):
                        problems.append(
                            f"{label_path}:{line_number}: coordinate outside [0, 1]"
                        )
                        totals["out_of_range_boxes"]+=1
                        continue
                    if box_width<=0 or box_height<=0:
                        problems.append(
                            f"{label_path}:{line_number}: non-positive box size"
                        )
                        totals["invalid_box_sizes"]+=1
                        continue
                    if (
                        x_center-box_width/2 <0
                        or y_center-box_height/2 <0
                        or x_center+box_width/2 >1
                        or y_center+box_height/2 >1
                    ):
                        problems.append(
                            f"{label_path}:{line_number}: box extends outside image"
                        )
                        totals["out_of_bounds_boxes"]+=1
                        continue
                    class_counts[split][class_id]+=1
                    totals[f"{split}_instances"]+=1
            except OSError as exc:
                problems.append(f"{split}: file read error: {image_path}: {exc}")
    print("\n--- CLASS INSTANCE COUNTS ---")
    for split in ("train", "val", "test"):
        print(f"\n{split}:")
        for class_id, name in sorted(class_names.items()):
            print(f"  {class_id} ({name}): {class_counts[split][class_id]}")
    duplicates=[paths for paths in image_hashes.values() if len(paths)>1]
    print("\n--- EXACT DUPLICATE IMAGES (SHA-256) ---")
    print(f"Duplicate groups: {len(duplicates)}")
    for group in duplicates[:20]:
        print("  " + " == ".join(group))
    print("\n--- AUDIT SUMMARY ---")
    for key, value in sorted(totals.items()):
        print(f"{key}: {value}")
    print(f"Problems recorded: {len(problems)}")
    for problem in problems[:100]:
        print(f"[ISSUE] {problem}")
    if len(problems) > 100:
        print(f"... {len(problems) - 100} more issues omitted")
    return 1 if problems else 0

if __name__=="__main__":
    raise SystemExit(main())
