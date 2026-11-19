"""Build the standalone Person 1 submission ZIP from an explicit source allowlist.

Both detector weights are included. Datasets, local videos, environments,
training runs and temporary files are excluded. Dependencies still need setup.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_model_release import check_release

REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "pytest.ini",
    "data.yaml",
    "configs/camera.yaml",
    "configs/detector.yaml",
    "src/camera_source.py",
    "src/models/object_detector.py",
    "src/models/person_detector.py",
    "scripts/check_model_release.py",
    "scripts/run_camera_demo.py",
    "scripts/check_standalone_pipeline.py",
    "models/model_card.json",
    "models/person_model_card.json",
    "models/dangerous_items_yolo11n.pt",
    "yolo11n.pt",
)
SOURCE_PATTERNS = (
    ".gitignore",
    ".gitattributes",
    "src/detector.py",
    "src/evaluation.py",
    "src/model_config.py",
    "src/models/*.py",
    "src/utils/*.py",
    "scripts/*.py",
    "tests/*.py",
    "configs/*.yaml",
    "docs/*.md",
    "reports/*.md",
    "reports/dataset/*.json",
    "reports/dataset/*.csv",
    "reports/experiments/*.json",
    "reports/experiments/*.csv",
    "reports/experiments/*.yaml",
    "reports/figures/*.png",
)
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "outputs",
    "runs",
    "tmp",
}

BUNDLE_README = """Person 1 standalone submission

Extract this ZIP to a writable folder and open a terminal there.
Follow README.md to create a Python environment and install dependencies.
Both item and person checkpoints are included; inference needs no weight download.

Verify:
  python scripts/check_model_release.py
  python -m pytest
  python -m pytest -m integration

Run a local camera demonstration:
  python scripts/run_camera_demo.py --camera 0 --with-people --device cpu

Run the standalone consumer on your own local recording:
  python scripts/check_standalone_pipeline.py --video path/to/video.mp4 --device cpu

See docs/standalone_demo.md and docs/person1_requirements.md for scope and evidence.
Raw datasets, local demo videos, Python dependencies and historical training runs
are not included. README.md explains dataset preparation for training/evaluation.
Saved reports may refer to original-machine paths; they remain historical evidence.
SOURCE_MANIFEST.json records the byte size and SHA-256 of every payload entry.
It excludes itself because a manifest cannot contain its own stable checksum.
"""


def digest(content):
    return hashlib.sha256(content).hexdigest()


def collect_sources(root):
    root = Path(root).resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required submission files are missing: {', '.join(missing)}"
        )
    selected = {root / name for name in REQUIRED_FILES}
    for pattern in SOURCE_PATTERNS:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    files = {}
    for path in sorted(selected):
        relative = path.relative_to(root)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Submission source points outside the project: {relative}"
            )
        if EXCLUDED_PARTS.intersection(resolved.relative_to(root).parts):
            raise ValueError(
                f"Submission source points into an excluded directory: {relative}"
            )
        files[relative.as_posix()] = path.read_bytes()
    return files


def package_project(root, output, *, overwrite=False):
    root, output = Path(root).resolve(), Path(output).expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Submission output must have the .zip extension")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Package already exists: {output}; use --overwrite to replace it"
        )
    release = check_release(root)
    if not release["passed"]:
        failures = [item["check"] for item in release["checks"] if not item["passed"]]
        raise ValueError(f"Item-model release checks failed: {', '.join(failures)}")
    files = collect_sources(root)
    for card_name, required_checkpoint in (
        ("models/model_card.json", "models/dangerous_items_yolo11n.pt"),
        ("models/person_model_card.json", "yolo11n.pt"),
    ):
        card = json.loads(files[card_name])
        if card["checkpoint"] != required_checkpoint:
            raise ValueError(f"Unexpected checkpoint path in {card_name}")
        if digest(files[required_checkpoint]) != card["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash does not match {card_name}")
    if output.exists() and any(output.samefile(root / name) for name in files):
        raise ValueError("Package output aliases a source file")
    files["BUNDLE_README.txt"] = BUNDLE_README.encode("utf-8")
    manifest = {
        "schema_version": 1,
        "scope": "Every archived payload except SOURCE_MANIFEST.json itself",
        "files": [
            {"path": name, "bytes": len(content), "sha256": digest(content)}
            for name, content in sorted(files.items())
        ],
    }
    files["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode(
        "utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    started = False
    try:
        with zipfile.ZipFile(
            output, "w" if overwrite else "x", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            started = True
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
    except Exception:
        if started:
            output.unlink(missing_ok=True)
        raise
    return {
        "output": str(output),
        "archived_files": len(files),
        "bytes": output.stat().st_size,
        "sha256": digest(output.read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/deliverables/person1_standalone.zip",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = package_project(REPO_ROOT, args.output, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
