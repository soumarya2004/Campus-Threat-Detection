"""Person boxes for the tracking teammate, separate from the five weapon classes.

The general COCO checkpoint is required locally. This module neither downloads
weights nor assigns tracking IDs; a tracker consumes the returned frame payload.
"""

from pathlib import Path

from src.detector import WeaponDetector
from src.model_config import REPO_ROOT
from src.models.object_detector import YOLOObjectDetector, _build_class_map


class _PersonBackend(WeaponDetector):
    def __init__(self, **kwargs):
        super().__init__(classes=[0], **kwargs)
        names = _build_class_map(self.class_names)
        if names.get(0) != "person":
            raise ValueError("Person checkpoint must label COCO class 0 as 'person'")
        self.class_names = {0: "person"}


class YOLOPersonDetector(YOLOObjectDetector):
    """Run the local general detector with only COCO person class 0 enabled.

    Use one instance per worker. The frame schema, validation, warmup(), close(),
    and context-manager behavior match YOLOObjectDetector. Class IDs belong to
    this person stream: do not merge them with weapon IDs without a stream tag.
    """

    _expected_classes = {0: "person"}

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.35,
        device: str | None = None,
        image_size: int = 640,
    ) -> None:
        super().__init__(
            model_path=REPO_ROOT / "yolo11n.pt" if model_path is None else model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            image_size=image_size,
        )

    def _create_detector(self) -> WeaponDetector:
        return _PersonBackend(
            model_path=str(self.model_path),
            confidence=self.confidence_threshold,
            device=self.device,
            image_size=self.image_size,
        )
