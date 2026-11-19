from pathlib import Path

import torch
from ultralytics import YOLO

class WeaponDetector:
    """YOLO11n detector for the project's five weapon classes."""

    def __init__(
        self,
        model_path="runs/detect/person1_clean/yolo11n_clean_v2/weights/best.pt",
        confidence=0.35,
        device=None,
    ):
        model_path=Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        self.device=device or(
            "0" if torch.cuda.is_available() else "cpu"
        )
        self.confidence=confidence
        self.model=YOLO(str(model_path))
        self.class_names=self.model.names

    def detect(self, frame):
        """
        Accept an OpenCV BGR frame.

        Return a list of dictionaries containing:
        class_id, class_name, confidence, and xyxy.
        """
        if frame is None or frame.size == 0:
            return []

        result=self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]

        detections=[]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id=int(box.cls.item())

            detections.append({
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": float(box.conf.item()),
                "xyxy": [
                    float(value)
                    for value in box.xyxy[0].tolist()
                ],
            })

        return detections
