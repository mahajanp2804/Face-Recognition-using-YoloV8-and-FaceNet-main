"""MTCNN face alignment + FaceNet (InceptionResnetV1, VGGFace2) embeddings."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceEmbedder:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or select_device()
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def align_with_landmarks(
        self, face_bgr: np.ndarray
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        """Align a BGR face crop to a 160x160 tensor with MTCNN, and return its 5 landmarks.

        Landmarks are ``(5, 2)`` pixel coordinates in the crop: left eye, right eye, nose,
        left mouth corner, right mouth corner. Both are None if MTCNN finds no face.
        This is the same detect → select → extract sequence as ``MTCNN.forward``, kept
        separate so the landmarks it computes anyway are not thrown away.
        """
        img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
        boxes, probs, points = self.mtcnn.select_boxes(
            boxes, probs, points, img, method=self.mtcnn.selection_method
        )
        if boxes is None:
            return None, None
        return self.mtcnn.extract(img, boxes, None), points[0]

    def align(self, face_bgr: np.ndarray) -> torch.Tensor | None:
        """Align a BGR face crop to a 160x160 tensor with MTCNN, or None if it finds no face."""
        return self.align_with_landmarks(face_bgr)[0]

    def embed_aligned(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Return the 512-d FaceNet embedding of an aligned face tensor."""
        with torch.no_grad():
            return self.resnet(face_tensor.unsqueeze(0).to(self.device)).cpu().numpy().flatten()

    def embed_with_landmarks(
        self, face_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(embedding, landmarks)`` for a BGR face crop, or ``(None, None)``."""
        face_tensor, landmarks = self.align_with_landmarks(face_bgr)
        if face_tensor is None:
            return None, None
        return self.embed_aligned(face_tensor), landmarks

    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """Align a BGR face crop and return its 512-d embedding, or None if MTCNN finds no face."""
        return self.embed_with_landmarks(face_bgr)[0]
