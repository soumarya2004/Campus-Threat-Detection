import cv2
from collections import Counter
from ultralytics import YOLO

VIDEO = "outputs/rifle.mp4"
MODEL = "runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt"
CONFIDENCE = 0.35

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO}")

counts = Counter()
frames_with_detections = Counter()
total_frames = 0
frames_with_any_detection = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    total_frames += 1
    result = model.predict(
        frame, conf=CONFIDENCE, device=0, verbose=False
    )[0]

    if len(result.boxes):
        frames_with_any_detection += 1

    classes_seen = set()

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        name = model.names[class_id]

        counts[name] += 1
        classes_seen.add(name)

        # Print every 50th frame's detections for inspection.
        if total_frames % 50 == 0:
            print(
                f"Frame {total_frames}: "
                f"{name}, confidence={confidence:.3f}"
            )

    for name in classes_seen:
        frames_with_detections[name] += 1

cap.release()

print("\n--- Detection inspection ---")
print(f"Frames processed: {total_frames}")
print(f"Frames with detections: {frames_with_any_detection}")
print("\nDetection count by class:")

for name, count in counts.most_common():
    print(f"{name}: {count} detections across "
          f"{frames_with_detections[name]} frames")
