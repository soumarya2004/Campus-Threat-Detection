import cv2
from ultralytics import YOLO

VIDEO = "outputs/rifle.mp4"
MODEL = "runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt"
TARGET_FRAMES = {150, 200, 350}
CONFIDENCE = 0.35

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_number += 1

    if frame_number not in TARGET_FRAMES:
        continue

    result = model.predict(
        frame, conf=CONFIDENCE, device=0, verbose=False
    )[0]

    print(f"\nFrame {frame_number}:")
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        coords = [round(float(x), 1) for x in box.xyxy[0].tolist()]
        print(
            f"{model.names[class_id]} | "
            f"confidence={confidence:.3f} | box={coords}"
        )

    annotated = result.plot()
    cv2.imwrite(
        f"outputs/inspect_frame_{frame_number}.jpg",
        annotated
    )

cap.release()
print("\nSaved inspection images in outputs/")
