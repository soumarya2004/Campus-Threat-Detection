import argparse
import csv
import math
from pathlib import Path
import cv2
import yaml

EXTENSIONS={".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/dangerous_items_deduplicated/data.yaml",
    )
    parser.add_argument(
        "--output",
        default="runs/annotation_review",
    )
    args=parser.parse_args()

    config_path=Path(args.data).resolve()
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root=Path(config.get("path", "."))
    if not root.is_absolute():
        root=(config_path.parent/root).resolve()

    names=config["names"]
    output=Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    issues=[]

    for split in ("train", "val", "test"):
        image_dir=root/config[split]
        label_dir=root/"labels"/split

        for image_path in image_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in EXTENSIONS:
                continue

            label_path=label_dir/image_path.relative_to(image_dir).with_suffix(".txt")
            if not label_path.exists():
                continue

            image=cv2.imread(str(image_path))
            if image is None:
                continue

            height, width=image.shape[:2]

            for line_no, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                fields=line.split()
                if len(fields)!=5:
                    continue

                try:
                    cls=int(fields[0])
                    cx, cy, bw, bh=map(float, fields[1:])
                except ValueError:
                    continue

                x1, y1= cx-bw/2, cy-bh/2
                x2, y2= cx+bw/2, cy+bh/2

                if x1<0 or y1<0 or x2>1 or y2>1:
                    issues.append({
                        "split": split,
                        "image": image_path,
                        "label": label_path,
                        "line": line_no,
                        "class_id": cls,
                        "class_name": str(names.get(cls, names.get(str(cls), "unknown"))),
                        "cx": cx, "cy": cy, "w": bw, "h": bh,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "width": width, "height": height,
                    })

    csv_path=output/"out_of_bounds_annotations.csv"
    fields=[
        "split", "image", "label", "line", "class_id", "class_name",
        "cx", "cy", "w", "h", "x1", "y1", "x2", "y2", "width", "height"
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer=csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in issues:
            writer.writerow({
                **item,
                "image": str(item["image"]),
                "label": str(item["label"]),
            })

    if not issues:
        print("No out-of-bounds boxes found.")
        return

    #create contact sheets of 12 annotations per page
    thumb_w, thumb_h=320, 250
    per_page=12
    cols, rows=3, 4

    for page_index in range(math.ceil(len(issues)/per_page)):
        page=issues[page_index*per_page:(page_index+1)*per_page]
        sheet=255*__import__("numpy").ones(
            (rows*thumb_h, cols*thumb_w, 3), dtype="uint8"
        )

        for index, item in enumerate(page):
            image=cv2.imread(str(item["image"]))
            h, w=image.shape[:2]

            #draw this annotation in red, clipped only for visualization
            x1=int(round(item["x1"]*w))
            y1=int(round(item["y1"]*h))
            x2=int(round(item["x2"]*w))
            y2=int(round(item["y2"]*h))

            x1c, y1c= max(0, x1), max(0, y1)
            x2c, y2c= min(w-1, x2), min(h-1, y2)

            cv2.rectangle(image, (x1c, y1c), (x2c, y2c), (0, 0, 255), 3)

            resized=cv2.resize(image, (thumb_w-12, thumb_h-55))
            label=(
                f"{item['split']} | {item['image'].name}"
            )
            detail=(
                f"line {item['line']} | {item['class_name']} | "
                f"box: {item['x1']:.3f},{item['y1']:.3f} to "
                f"{item['x2']:.3f},{item['y2']:.3f}"
            )

            tile=255*__import__("numpy").ones(
                (thumb_h, thumb_w, 3), dtype="uint8"
            )
            tile[4:4+resized.shape[0], 6:6+resized.shape[1]]=resized
            cv2.putText(tile, label[:42], (5, thumb_h-32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1)
            cv2.putText(tile, detail[:48], (5, thumb_h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 180), 1)

            r, c=divmod(index, cols)
            sheet[r*thumb_h:(r+1)*thumb_h,
                  c*thumb_w:(c+1)*thumb_w]=tile

        path=output/f"contact_sheet_{page_index+1:02d}.jpg"
        cv2.imwrite(str(path), sheet)
        print(f"Saved: {path}")

    print(f"Problematic annotation rows: {len(issues)}")
    print(f"CSV report: {csv_path}")


if __name__=="__main__":
    main()
