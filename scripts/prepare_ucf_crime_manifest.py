"""Build a manifest from the official UCF-Crime test split and annotations."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.external_evaluation.ucf_crime import (  # noqa: E402
    build_ucf_test_manifest,
    write_manifest,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--temporal-annotations", type=Path, required=True)
    parser.add_argument(
        "--mapping", type=Path, default=ROOT / "configs/behavior/dataset_mapping.yaml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/behavior/external_ucf/manifest.csv",
    )
    args = parser.parse_args(argv)

    try:
        rows, metadata = build_ucf_test_manifest(
            args.videos_root, args.test_split, args.temporal_annotations, args.mapping
        )
        write_manifest(rows, args.output, metadata)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Manifest: {args.output}")
    print(f"Official split entries: {metadata['total_split_entries']}")
    print(f"Missing local video files: {len(metadata['missing_video_files'])}")
    print(f"Manifest checksum: {metadata['split_file_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
