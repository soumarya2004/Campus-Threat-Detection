"""Split image families before training; never modify the source dataset.

An A<number>_ prefix denotes an offline augmentation. Families and exact image
hashes are joined; one representative is kept per connected group. This does
not identify different frames from the same video.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_family(stem: str) -> str:
    return re.sub(r"^(?:A\d+_)+", "", stem)


def read_config(config_path: Path) -> tuple[dict, Path, dict[int, str]]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not config.get("names"):
        raise ValueError("Dataset YAML must contain class names.")
    raw_names = config["names"]
    names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else raw_names
    names = {int(key): str(value) for key, value in names.items()}
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = config_path.parent / root
    return config, root.resolve(), names


def read_labels(path: Path, names: dict[int, str]) -> list[tuple]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 5:
                raise ValueError("expected class plus four coordinates")
            class_value, cx, cy, width, height = map(float, fields)
            if not all(math.isfinite(v) for v in (class_value, cx, cy, width, height)):
                raise ValueError("non-finite annotation")
            if not class_value.is_integer() or int(class_value) not in names:
                raise ValueError("unknown class")
            if width <= 0 or height <= 0:
                raise ValueError("non-positive box size")
            if (
                min(cx - width / 2, cy - height / 2) < -1e-8
                or max(cx + width / 2, cy + height / 2) > 1 + 1e-8
            ):
                raise ValueError("box outside image")
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        rows.append((int(class_value), cx, cy, width, height))
    return rows


def inventory(
    config_path: Path, check_images: bool = True
) -> tuple[list[dict], Path, dict[int, str]]:
    config, root, names = read_config(config_path)
    records = []
    for split in SPLITS:
        if split not in config:
            raise ValueError(f"Missing configured split: {split}")
        image_dir = root / config[split]
        label_dir = root / "labels" / split
        images = sorted(
            p
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if not images or not label_dir.is_dir():
            raise ValueError(f"Missing or empty {split} images/labels: {image_dir}")
        expected_labels = set()
        for image in images:
            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            expected_labels.add(label)
            if not label.is_file():
                raise ValueError(f"Missing label for {image}")
            rows = read_labels(label, names)
            if check_images and cv2.imread(str(image)) is None:
                raise ValueError(f"Unreadable image: {image}")
            records.append(
                {
                    "image": image,
                    "label": label,
                    "source_split": split,
                    "relative_image": image.relative_to(root).as_posix(),
                    "relative_label": label.relative_to(root).as_posix(),
                    "family": source_family(image.stem),
                    "image_sha256": digest_file(image),
                    "label_sha256": digest_file(label),
                    "rows": rows,
                }
            )
        orphan_labels = set(label_dir.rglob("*.txt")) - expected_labels
        if orphan_labels:
            raise ValueError(f"Orphan label: {min(orphan_labels)}")
    return records, root, names


def make_groups(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Union families and hashes; exclude groups with ambiguous exact-copy labels."""
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[find(right)] = find(left)

    families, hashes = {}, {}
    for index, record in enumerate(records):
        for lookup, key in (
            (families, record["family"]),
            (hashes, record["image_sha256"]),
        ):
            if key in lookup:
                union(index, lookup[key])
            else:
                lookup[key] = index
    members = defaultdict(list)
    for index, record in enumerate(records):
        members[find(index)].append(record)
    groups, excluded = [], []
    for group_records in members.values():
        label_signatures = defaultdict(set)
        for record in group_records:
            signature = tuple(
                sorted(tuple(round(v, 6) for v in row) for row in record["rows"])
            )
            label_signatures[record["image_sha256"]].add(signature)
        group_id = hashlib.sha256(
            "\n".join(sorted({r["family"] for r in group_records})).encode()
        ).hexdigest()[:20]
        group = {"id": group_id, "records": group_records}
        if any(len(values) > 1 for values in label_signatures.values()):
            group["reason"] = "conflicting labels on byte-identical images"
            excluded.append(group)
            continue
        representative = min(
            group_records,
            key=lambda r: (r["image"].stem != r["family"], r["relative_image"]),
        )
        group["representative"] = representative
        group["counts"] = Counter(row[0] for row in representative["rows"])
        if not representative["rows"]:
            group["counts"]["background"] = 1
        groups.append(group)
    return sorted(groups, key=lambda g: g["id"]), sorted(
        excluded, key=lambda g: g["id"]
    )


def assign_splits(
    groups: list[dict], seed: int = 42, ratios=(0.70, 0.15, 0.15)
) -> dict[str, str]:
    """Greedily balance class instances/background while meeting image quotas."""
    if len(groups) < 3:
        raise ValueError("At least three independent image groups are needed.")
    if (
        len(ratios) != 3
        or any(not math.isfinite(v) or v <= 0 for v in ratios)
        or not math.isclose(sum(ratios), 1)
    ):
        raise ValueError("Three positive split ratios must sum to 1.")
    quotas = [math.floor(len(groups) * ratio) for ratio in ratios]
    remainder_order = sorted(
        range(3), key=lambda i: (-(len(groups) * ratios[i] - quotas[i]), i)
    )
    for i in remainder_order[: len(groups) - sum(quotas)]:
        quotas[i] += 1
    if min(quotas) == 0:
        raise ValueError("Dataset is too small for non-empty requested splits.")
    totals = sum((g["counts"] for g in groups), Counter())
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(
        key=lambda g: (min(totals[c] for c in g["counts"]), -sum(g["counts"].values()))
    )
    current = [Counter(), Counter(), Counter()]
    sizes = [0, 0, 0]
    result = {}
    for group in ordered:
        costs = []
        for i, ratio in enumerate(ratios):
            if sizes[i] >= quotas[i]:
                costs.append(float("inf"))
                continue
            cost = (
                2
                * ((sizes[i] + 1 - quotas[i]) ** 2 - (sizes[i] - quotas[i]) ** 2)
                / quotas[i]
            )
            for cls, count in group["counts"].items():
                target = max(totals[cls] * ratio, 1.0)
                before = current[i][cls] - target
                cost += ((before + count) ** 2 - before**2) / target
            costs.append(cost)
        chosen = min(range(3), key=lambda i: costs[i])
        current[chosen].update(group["counts"])
        sizes[chosen] += 1
        result[group["id"]] = SPLITS[chosen]
    return result


def render_manifest(
    groups: list[dict], excluded: list[dict], assignments: dict[str, str]
) -> bytes:
    """Prepare stable CSV bytes before any output files are created."""
    fields = [
        "source_image",
        "source_label",
        "image_sha256",
        "label_sha256",
        "source_family",
        "group_id",
        "assigned_split",
        "selected",
        "reason",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for group in sorted(groups + excluded, key=lambda g: g["id"]):
        for record in sorted(group["records"], key=lambda r: r["relative_image"]):
            selected = record is group.get("representative")
            writer.writerow(
                {
                    "source_image": record["relative_image"],
                    "source_label": record["relative_label"],
                    "image_sha256": record["image_sha256"],
                    "label_sha256": record["label_sha256"],
                    "source_family": record["family"],
                    "group_id": group["id"],
                    "assigned_split": assignments.get(group["id"], "excluded"),
                    "selected": int(selected),
                    "reason": group.get(
                        "reason",
                        "selected representative"
                        if selected
                        else "same family or exact duplicate",
                    ),
                }
            )
    return stream.getvalue().encode("utf-8")


def build_dataset(
    config_path: Path,
    output: Path,
    report_dir: Path,
    seed: int = 42,
    ratios=(0.70, 0.15, 0.15),
) -> dict:
    output, report_dir = output.resolve(), report_dir.resolve()
    manifest_path = report_dir / "grouped_v1_manifest.csv"
    summary_path = report_dir / "grouped_v1_summary.json"
    if output.exists():
        raise ValueError(
            "Output dataset already exists; choose a fresh output directory."
        )
    if report_dir.exists() and not report_dir.is_dir():
        raise ValueError("Report directory is not a directory.")
    records, source_root, names = inventory(config_path)
    groups, excluded = make_groups(records)
    assignments = assign_splits(groups, seed, ratios)
    manifest_bytes = render_manifest(groups, excluded, assignments)
    counts = {}
    for split in SPLITS:
        selected_groups = [g for g in groups if assignments[g["id"]] == split]
        class_counts = sum((g["counts"] for g in selected_groups), Counter())
        counts[split] = {
            "images": len(selected_groups),
            "background_images": class_counts["background"],
            "class_instances": {names[c]: class_counts[c] for c in sorted(names)},
        }
    summary = {
        "schema_version": 1,
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "source_dataset": source_root.relative_to(REPO_ROOT).as_posix()
        if source_root.is_relative_to(REPO_ROOT)
        else source_root.name,
        "source_images": len(records),
        "selected_images": len(groups),
        "excluded_conflicting_groups": len(excluded),
        "excluded_conflicting_images": sum(len(g["records"]) for g in excluded),
        "representatives_without_unaugmented_original": sum(
            g["representative"]["image"].stem != g["representative"]["family"]
            for g in groups
        ),
        "splits": counts,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "grouping": "Union of filename family (remove A<number>_ prefixes) and exact image SHA-256.",
        "selection": "One representative per group; prefer an unprefixed original, else deterministic filename order. Online augmentation only during training.",
        "limitation": "Different frames from the same video or scene are not identifiable from this metadata. This is image-family separation, not verified video/source separation.",
    }
    # Match the original v1 summary's native newlines. Source label bytes may
    # also vary across operating systems; a different report directory records
    # that distinct build without overwriting a frozen experiment's manifest.
    summary_bytes = (
        (json.dumps(summary, indent=2) + "\n").replace("\n", os.linesep).encode("utf-8")
    )
    artifacts = ((manifest_path, manifest_bytes), (summary_path, summary_bytes))
    for path, content in artifacts:
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise ValueError(
                f"Existing report differs: {path}. Use a fresh --report-dir; no dataset was created."
            )

    output.mkdir(parents=True)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for group in groups:
        record, split = group["representative"], assignments[group["id"]]
        shutil.copy2(record["image"], output / "images" / split / record["image"].name)
        shutil.copy2(record["label"], output / "labels" / split / record["label"].name)
    config = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts:
        if not path.exists():
            path.write_bytes(content)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data/processed/dangerous_items_cleaned_v2/data.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/processed/dangerous_items_grouped_v1",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=REPO_ROOT / "reports/dataset"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary = build_dataset(args.data, args.output, args.report_dir, args.seed)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(summary, indent=2))
    print(f"Training YAML: {args.output.resolve() / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
