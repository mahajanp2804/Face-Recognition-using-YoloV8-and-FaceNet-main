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

    def align(self, face_bgr: np.ndarray) -> torch.Tensor | None:
        """Align a BGR face crop to a 160x160 tensor with MTCNN, or None if it finds no face."""
        face_rgb = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        return self.mtcnn(face_rgb)

    def embed_aligned(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Return the 512-d FaceNet embedding of an aligned face tensor."""
        with torch.no_grad():
            return self.resnet(face_tensor.unsqueeze(0).to(self.device)).cpu().numpy().flatten()

    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """Align a BGR face crop and return its 512-d embedding, or None if MTCNN finds no face."""
        face_tensor = self.align(face_bgr)
        if face_tensor is None:
            return None
        return self.embed_aligned(face_tensor)
