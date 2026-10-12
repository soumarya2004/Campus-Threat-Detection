from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT=Path("data/raw/dangerous_items")
SPLITS=("train", "val", "test")

groups=defaultdict(list)

for split in SPLITS:
    image_dir=ROOT/"images"/split
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            digest=sha256(path.read_bytes()).hexdigest()
            groups[digest].append((split, path))
duplicates=[files for files in groups.values() if len(files)>1]
cross_split=[
    files for files in duplicates
    if len({split for split, _ in files})>1
]
print(f"Unique image hashes: {len(groups)}")
print(f"Duplicate groups: {len(duplicates)}")
print(f"Cross-split duplicate groups: {len(cross_split)}")
print(
    "Images involved in cross-split groups:",
    sum(len(files) for files in cross_split)
)
for files in cross_split[:30]:
    print("\nDuplicate group:")
    for split, path in files:
        print(f"  {split}: {path}")
