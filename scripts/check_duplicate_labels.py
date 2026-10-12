from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT=Path("data/raw/dangerous_items")
SPLITS=("train", "val", "test")

groups=defaultdict(list)

for split in SPLITS:
    for image in (ROOT/"images"/split).rglob("*"):
        if not image.is_file() or image.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            continue

        digest=sha256(image.read_bytes()).hexdigest()
        label=ROOT/"labels"/split/f"{image.stem}.txt"
        groups[digest].append((split, image, label))
cross_split=[
    files for files in groups.values()
    if len(files)>1 and len({item[0] for item in files})>1
]
same_labels=0
different_labels=0
missing_labels=0
for files in cross_split:
    contents=[]
    missing=False
    for split, image, label in files:
        if not label.exists():
            missing=True
            break
        contents.append(label.read_text(encoding="utf-8").strip())
    if missing:
        missing_labels+=1
    elif len(set(contents))==1:
        same_labels+=1
    else:
        different_labels+=1
        print("\nLABEL MISMATCH:")
        for split, image, label in files:
            print(f"  {split}: {image.name}")
            if label.exists():
                print(f"    {label.read_text(encoding='utf-8').strip()}")
            else:
                print("    LABEL MISSING")
print("\n--- DUPLICATE LABEL AUDIT ---")
print(f"Cross-split duplicate groups: {len(cross_split)}")
print(f"Identical labels: {same_labels}")
print(f"Different labels: {different_labels}")
print(f"Groups with missing labels: {missing_labels}")
