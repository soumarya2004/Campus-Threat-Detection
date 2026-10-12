from pathlib import Path
import shutil
import yaml

SOURCE=Path("data/processed/dangerous_items_deduplicated").resolve()
OUTPUT=Path("data/processed/dangerous_items_cleaned_v2").resolve()
SPLITS=("train", "val", "test")

def main():
    source_yaml=SOURCE/"data.yaml"
    if not source_yaml.exists():
        raise SystemExit(f"Missing dataset config: {source_yaml}")

    if OUTPUT.exists():
        raise SystemExit(
            f"Refusing to overwrite existing folder: {OUTPUT}"
        )

    config=yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    changes=[]

    try:
        shutil.copytree(SOURCE, OUTPUT)

        for split in SPLITS:
            label_dir=OUTPUT/"labels"/split

            for label_path in label_dir.rglob("*.txt"):
                original=label_path.read_text(encoding="utf-8")
                corrected_lines=[]

                for line_no, line in enumerate(original.splitlines(), 1):
                    fields=line.split()

                    if len(fields)!=5:
                        corrected_lines.append(line)
                        continue

                    cls=fields[0]
                    cx, cy, w, h=map(float, fields[1:])

                    x1=cx-w/2
                    y1=cy-h/2
                    x2=cx+w/2
                    y2=cy+h/2

                    #preserve valid rows exactly as they were
                    if(
                        0.0<=x1<=1.0
                        and 0.0<=y1<=1.0
                        and 0.0<=x2<=1.0
                        and 0.0<=y2<=1.0
                    ):
                        corrected_lines.append(line)
                        continue

                    #clamp only edges outside the normalized image
                    x1c=min(1.0, max(0.0, x1))
                    y1c=min(1.0, max(0.0, y1))
                    x2c=min(1.0, max(0.0, x2))
                    y2c=min(1.0, max(0.0, y2))

                    new_cx=(x1c+x2c)/2
                    new_cy=(y1c+y2c)/2
                    new_w=x2c-x1c
                    new_h=y2c-y1c

                    new_line=(
                        f"{cls} {new_cx:.8f} {new_cy:.8f} "
                        f"{new_w:.8f} {new_h:.8f}"
                    )
                    corrected_lines.append(new_line)

                    changes.append(
                        f"{split} {label_path.relative_to(OUTPUT)}:"
                        f"{line_no}\n"
                        f"  before: {line}\n"
                        f"  after:  {new_line}\n"
                    )

                label_path.write_text(
                    "\n".join(corrected_lines)
                    + ("\n" if corrected_lines else ""),
                    encoding="utf-8",
                )

        #ensure the new config points to the new dataset
        new_config={
            "path": str(OUTPUT),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": config["names"],
        }
        (OUTPUT/"data.yaml").write_text(
            yaml.safe_dump(
                new_config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        report_path=OUTPUT/"rounding_corrections.txt"
        report_path.write_text(
            f"Source: {SOURCE}\n"
            f"Output: {OUTPUT}\n"
            f"Changed annotation rows: {len(changes)}\n\n"
            + "\n".join(changes),
            encoding="utf-8",
        )

        print(f"Created corrected copy: {OUTPUT}")
        print(f"Changed annotation rows: {len(changes)}")
        print(f"Correction log: {report_path}")
        print("Source dataset was not modified.")

    except Exception:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        raise


if __name__=="__main__":
    main()
