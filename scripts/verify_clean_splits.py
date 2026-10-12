from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT=Path("data/processed/dangerous_items_deduplicated")
groups=defaultdict(list)

for split in ("train", "val", "test"):
    for image in (ROOT/"images"/split).rglob("*"):
        if image.is_file() and image.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            digest=sha256(image.read_bytes()).hexdigest()
            groups[digest].append((split, image.name))
duplicates=[files for files in groups.values() if len(files)>1]
cross_split=[
    files for files in duplicates
    if len({split for split, _ in files})>1
]
print(f"Unique image hashes: {len(groups)}")
print(f"Duplicate groups within or across splits: {len(duplicates)}")
print(f"Cross-split duplicate groups: {len(cross_split)}")
for group in cross_split:
    print("\nCROSS-SPLIT LEAKAGE:")
    for split, filename in group:
        print(f"  {split}: {filename}")
