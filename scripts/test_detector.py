import cv2
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import WeaponDetector

IMAGE_PATH=(
    "data/processed/dangerous_items_cleaned_v2/"
    "images/test/0017cc01-0481.jpg"
)

OUTPUT_PATH=PROJECT_ROOT/"outputs"/"detector_test.jpg"

frame=cv2.imread(str(PROJECT_ROOT/IMAGE_PATH))

if frame is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

detector=WeaponDetector()
detections=detector.detect(frame)

print("Image:", IMAGE_PATH)
print("Device:", detector.device)
print("Detections:", len(detections))

for detection in detections:
    print(detection)

    x1, y1, x2, y2=map(int, detection["xyxy"])
    label=(
        f'{detection["class_name"]} '
        f'{detection["confidence"]:.2f}'
    )

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), frame)

print("Annotated image saved to:", OUTPUT_PATH)
