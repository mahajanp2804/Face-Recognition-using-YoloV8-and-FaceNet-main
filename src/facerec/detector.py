"""YOLOv8 face detection and cropping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# (x1, y1, x2, y2) in pixel coordinates
Box = tuple[int, int, int, int]


def clamp_box(box: Box, width: int, height: int) -> Box | None:
    """Clip a box to the image bounds; return None if nothing of it is left inside."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop(image: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


class FaceDetector:
    """Wraps the trained YOLOv8 face model (``detection/weights/best.pt``)."""

    def __init__(self, weights: Path) -> None:
        from ultralytics import YOLO  # heavy import, deferred so tests don't need it

        self.model = YOLO(str(weights))

    def detect(self, image: np.ndarray) -> list[Box]:
        """Return raw face boxes, most confident first (YOLO's own ordering)."""
        result = self.model(image, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]

    def crop_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """Return every detected face as a BGR crop, skipping boxes that fall outside the image."""
        h, w = image.shape[:2]
        faces = []
        for raw in self.detect(image):
            box = clamp_box(raw, w, h)
            if box is not None:
                faces.append(crop(image, box))
        return faces
