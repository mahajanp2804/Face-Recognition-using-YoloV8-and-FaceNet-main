"""Multi-face recognition and attendance recording with fake models."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from facerec.matcher import UNKNOWN, FaceMatcher  # noqa: E402
from facerec.recognize import draw_results, recognize_faces, record_attendance  # noqa: E402


def unit(index: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[index] = 1.0
    return v


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image):
        return self.boxes


class FakeEmbedder:
    """Embeds a crop as unit(<max pixel value>); a crop of value 0 "fails alignment"."""

    def embed(self, face_bgr):
        value = int(face_bgr.max())
        return None if value == 0 else unit(value)


def frame_with_patches():
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    frame[10:50, 10:50] = 1  # Alice
    frame[10:50, 110:150] = 2  # Bob
    frame[10:50, 210:250] = 9  # stranger
    return frame


def test_recognizes_every_face_and_drops_false_positives():
    boxes = [
        (10, 10, 50, 50),
        (110, 10, 150, 50),
        (210, 10, 250, 50),
        (60, 60, 90, 90),  # empty area: alignment fails -> dropped
        (400, 400, 500, 500),  # fully outside the frame -> dropped
    ]
    matcher = FaceMatcher({"Alice": [unit(1)], "Bob": [unit(2)]}, threshold=0.8)

    results = recognize_faces(frame_with_patches(), FakeDetector(boxes), FakeEmbedder(), matcher)

    assert [(r.name, r.box) for r in results] == [
        ("Alice", (10, 10, 50, 50)),
        ("Bob", (110, 10, 150, 50)),
        (UNKNOWN, (210, 10, 250, 50)),
    ]
    assert results[0].known and not results[2].known
    assert results[0].distance == pytest.approx(0.0)


def test_box_partly_outside_is_clamped():
    matcher = FaceMatcher({"Alice": [unit(1)]})

    results = recognize_faces(
        frame_with_patches(), FakeDetector([(-20, -5, 50, 50)]), FakeEmbedder(), matcher
    )

    assert results[0].box == (0, 0, 50, 50)


def test_draw_results_marks_the_frame():
    frame = frame_with_patches()
    matcher = FaceMatcher({"Alice": [unit(1)]})
    results = recognize_faces(
        frame, FakeDetector([(10, 10, 50, 50), (210, 10, 250, 50)]), FakeEmbedder(), matcher
    )

    out = draw_results(frame.copy(), results)

    assert out.shape == frame.shape
    assert (out != frame).any()


def test_record_attendance_once_per_day(tmp_path):
    path = tmp_path / "attendance.csv"

    first = record_attendance(["Alice", "Bob", "Alice"], path)
    second = record_attendance(["Bob", "Carol"], path)

    assert first == {"Alice": True, "Bob": True}
    assert second == {"Bob": False, "Carol": True}
    assert path.read_text().count("\n") == 4  # header + Alice, Bob, Carol
