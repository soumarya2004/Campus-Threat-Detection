"""Create portable manifests for the official XD-Violence train/test folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.xd_violence import (
    attach_test_events,
    scan_video_directory,
    stratified_video_split,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/raw/XD-Violence")
    )
    parser.add_argument(
        "--train-dir", type=Path, default=None, help="Defaults to <dataset-root>/Train"
    )
    parser.add_argument(
        "--test-dir", type=Path, default=None, help="Defaults to <dataset-root>/Test"
    )
    parser.add_argument(
        "--test-annotations",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/annotations.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-source",
        default="unspecified",
        help="Archive or source label, such as '0001-1004'",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    dataset_root = (
        args.dataset_root
        if args.dataset_root.is_absolute()
        else project_root / args.dataset_root
    )
    train_dir = args.train_dir or (dataset_root / "Train")
    test_dir = args.test_dir or (dataset_root / "Test")
    if args.test_annotations:
        annotations = args.test_annotations
    else:
        candidates = [
            dataset_root / "annotations.txt",
            dataset_root / "test_annotations.txt",
        ]
        annotations = next(
            (path for path in candidates if path.is_file()), candidates[0]
        )
    if not train_dir.is_absolute():
        train_dir = project_root / train_dir
    if not test_dir.is_absolute():
        test_dir = project_root / test_dir
    if not annotations.is_absolute():
        annotations = project_root / annotations

    train_records, train_counts = scan_video_directory(train_dir, project_root)
    test_records, test_counts = scan_video_directory(test_dir, project_root)
    train, validation = stratified_video_split(
        train_records,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    test_records, unmatched_annotations = attach_test_events(test_records, annotations)

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else project_root / args.output_dir
    )
    write_json(output_dir / "xd_train_manifest.json", train)
    write_json(output_dir / "xd_val_manifest.json", validation)
    write_json(output_dir / "xd_test_manifest.json", test_records)
    write_json(
        output_dir / "xd_test_events.json",
        [
            {
                "source_video_id": record["source_video_id"],
                "video_path": record["video_path"],
                "label": record["label"],
                "violent_intervals_frames": record.get("violent_intervals_frames", []),
            }
            for record in test_records
        ],
    )

    train_file_count = sum(train_counts.values())
    test_file_count = sum(test_counts.values())
    summary = {
        "dataset": "XD-Violence",
        "training_source": args.training_source,
        "training_scope": (
            "full_official_training_split"
            if train_file_count == 3954
            else "training_subset"
        ),
        "test_scope": (
            "full_official_test_split" if test_file_count == 800 else "test_subset"
        ),
        "source_video_files": {"train": train_file_count, "test": test_file_count},
        "label_scope": "Fighting, Shooting, Riot, Abuse vs Normal",
        "ignored_hazard_only_videos": {
            "train": train_counts["exclude_non_human_hazard"],
            "test": test_counts["exclude_non_human_hazard"],
        },
        "split_counts": {
            "train": {
                "videos": len(train),
                "normal": sum(r["label"] == 0 for r in train),
                "violence": sum(r["label"] == 1 for r in train),
            },
            "validation": {
                "videos": len(validation),
                "normal": sum(r["label"] == 0 for r in validation),
                "violence": sum(r["label"] == 1 for r in validation),
            },
            "test": {
                "videos": len(test_records),
                "normal": sum(r["label"] == 0 for r in test_records),
                "violence": sum(r["label"] == 1 for r in test_records),
            },
        },
        "test_violent_videos_with_intervals": sum(
            "violent_intervals_frames" in r for r in test_records
        ),
        "unmatched_annotation_video_ids": unmatched_annotations,
        "seed": args.seed,
        "validation_ratio_from_official_training_videos": args.validation_ratio,
    }
    write_json(output_dir / "xd_manifest_summary.json", summary)
    print("XD-Violence manifests written to:", output_dir)
    for split, values in summary["split_counts"].items():
        print(
            f"{split:10} videos={values['videos']:4} normal={values['normal']:4} violence={values['violence']:4}"
        )
    print("Ignored hazard-only videos:", dict(summary["ignored_hazard_only_videos"]))
    if unmatched_annotations:
        print("Annotation rows with no local test video:", len(unmatched_annotations))


if __name__ == "__main__":
    main()
