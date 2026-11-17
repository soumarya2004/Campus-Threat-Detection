"""Read-only verification of exact hashes and augmentation filename families."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.build_grouped_dataset import REPO_ROOT, inventory
except ModuleNotFoundError:  # direct script execution
    from build_grouped_dataset import REPO_ROOT, inventory


def verify_dataset(data_path: Path, manifest_path: Path | None = None) -> dict:
    records, _, _ = inventory(data_path)
    by_hash, by_family = defaultdict(list), defaultdict(list)
    for record in records:
        by_hash[record["image_sha256"]].append(record)
        by_family[record["family"]].append(record)
    cross_hashes = [
        g for g in by_hash.values() if len({r["source_split"] for r in g}) > 1
    ]
    cross_families = [
        g for g in by_family.values() if len({r["source_split"] for r in g}) > 1
    ]
    problems = []
    if cross_hashes:
        problems.append(f"{len(cross_hashes)} exact image hashes cross splits")
    if cross_families:
        problems.append(
            f"{len(cross_families)} augmentation filename families cross splits"
        )
    if manifest_path is not None:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        group_splits, family_splits, hash_splits = (
            defaultdict(set),
            defaultdict(set),
            defaultdict(set),
        )
        selected_groups = Counter()
        source_images = set()
        expected = set()
        for row in rows:
            split = row["assigned_split"]
            if split not in {"train", "val", "test", "excluded"} or row[
                "selected"
            ] not in {"0", "1"}:
                problems.append("Manifest has an invalid split or selection flag")
            if row["source_image"] in source_images:
                problems.append("Manifest repeats a source image")
            source_images.add(row["source_image"])
            if split == "excluded":
                if row["selected"] != "0":
                    problems.append("Manifest selects an excluded image")
                continue
            group_splits[row["group_id"]].add(split)
            family_splits[row["source_family"]].add(split)
            hash_splits[row["image_sha256"]].add(split)
            if row["selected"] == "1":
                selected_groups[row["group_id"]] += 1
                expected.add(
                    (
                        split,
                        Path(row["source_image"]).name,
                        row["image_sha256"],
                        row["label_sha256"],
                    )
                )
        if any(selected_groups[group_id] != 1 for group_id in group_splits):
            problems.append(
                "Manifest must select exactly one representative per included group"
            )
        if any(
            len(v) > 1
            for lookup in (group_splits, family_splits, hash_splits)
            for v in lookup.values()
        ):
            problems.append(
                "Manifest assigns a group, family or exact hash to multiple splits"
            )
        actual = {
            (r["source_split"], r["image"].name, r["image_sha256"], r["label_sha256"])
            for r in records
        }
        if expected != actual:
            problems.append(
                f"Manifest mismatch: {len(expected - actual)} missing/changed and {len(actual - expected)} unexpected images/labels"
            )
    result = {
        "images": len(records),
        "unique_image_hashes": len(by_hash),
        "cross_split_exact_hash_groups": len(cross_hashes),
        "cross_split_filename_families": len(cross_families),
        "manifest_checked": manifest_path is not None,
        "problems": problems,
        "limitation": "Different source-video frames and visual near-duplicates with unrelated names are not ruled out.",
    }
    if problems:
        raise ValueError(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data/processed/dangerous_items_grouped_v1/data.yaml",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_dataset(args.data, args.manifest), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
