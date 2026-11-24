"""FaceEmbedder edge cases with the real MTCNN (skipped without facenet-pytorch)."""

import numpy as np
import pytest

pytest.importorskip("cv2")
facenet_pytorch = pytest.importorskip("facenet_pytorch")

from facerec.embedder import FaceEmbedder  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    # Only MTCNN is needed here; its weights ship with facenet-pytorch, so skip loading
    # FaceNet (which would download the VGGFace2 checkpoint).
    e = FaceEmbedder.__new__(FaceEmbedder)
    e.device = "cpu"
    e.mtcnn = facenet_pytorch.MTCNN(keep_all=False, device="cpu")
    return e


@pytest.mark.parametrize("shape", [(5, 5, 3), (12, 40, 3), (19, 19, 3), (40, 8, 3)])
def test_crop_smaller_than_min_face_size_returns_none(embedder, shape):
    # Used to crash inside facenet-pytorch with "torch.cat(): expected a non-empty list"
    crop = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)

    assert embedder.align_with_landmarks(crop) == (None, None)


def test_blank_crop_without_face_returns_none(embedder):
    assert embedder.align_with_landmarks(np.zeros((80, 80, 3), np.uint8)) == (None, None)
