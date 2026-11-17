from pathlib import Path

import torch
from ultralytics import YOLO

from src.model_config import (
    load_detector_config,
    validate_confidence,
    validate_image_size,
)


class WeaponDetector:
    """Ultralytics adapter used by the weapon and person frame interfaces."""

    def __init__(
        self,
        model_path=None,
        confidence=None,
        device=None,
        image_size=None,
        classes=None,
    ):
        settings = (
            load_detector_config()
            if model_path is None or confidence is None or image_size is None
            else {}
        )
        model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else settings["model_path"]
        )
        confidence = settings["confidence"] if confidence is None else confidence
        confidence = validate_confidence(confidence)
        image_size = validate_image_size(
            settings["image_size"] if image_size is None else image_size
        )

        if not model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.device = (
            device
            if device is not None
            else ("0" if torch.cuda.is_available() else "cpu")
        )
        self.confidence = confidence
        self.image_size = image_size
        self.classes = classes
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names

    def detect(self, frame):
        """
        Accept an OpenCV BGR frame.

        Return a list of dictionaries containing:
        class_id, class_name, confidence, and xyxy.
        """
        if frame is None or frame.size == 0:
            return []

        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            imgsz=self.image_size,
            iou=0.7,
            max_det=300,
            augment=False,
            classes=self.classes,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": float(box.conf.item()),
                    "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                }
            )

        return detections
