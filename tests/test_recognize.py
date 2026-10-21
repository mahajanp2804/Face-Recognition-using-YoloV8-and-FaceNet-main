"""Multi-face recognition and attendance recording with fake models."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from facerec.liveness import LivenessSettings  # noqa: E402
from facerec.matcher import UNKNOWN, FaceMatcher  # noqa: E402
from facerec.recognize import (  # noqa: E402
    AttendanceSession,
    FaceResult,
    draw_results,
    recognize_faces,
    record_attendance,
)


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


# ---------------------------------------------------------------- liveness integration

LANDMARKS = np.array([[1, 1], [3, 1], [2, 2], [1.5, 3], [2.5, 3]], dtype=np.float32)


class LandmarkEmbedder(FakeEmbedder):
    def embed_with_landmarks(self, face_bgr):
        embedding = self.embed(face_bgr)
        return (None, None) if embedding is None else (embedding, LANDMARKS.copy())


def test_landmarks_are_returned_in_frame_coordinates():
    matcher = FaceMatcher({"Bob": [unit(2)]})

    (result,) = recognize_faces(
        frame_with_patches(),
        FakeDetector([(110, 10, 150, 50)]),
        LandmarkEmbedder(),
        matcher,
        with_landmarks=True,
    )

    assert result.name == "Bob"
    np.testing.assert_allclose(result.landmarks, LANDMARKS + [110, 10])


def face(name, cue=None):
    """A FaceResult whose landmarks produce the given depth cue (None = no landmarks)."""
    landmarks = None
    if cue is not None:
        landmarks = np.array([[-1, 0], [1, 0], [2 * cue, 1], [-0.8, 2], [0.8, 2]], float)
    return FaceResult((0, 0, 10, 10), name, 0.1, landmarks)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1 / 30
        return self.t


def test_session_without_liveness_marks_immediately(tmp_path):
    session = AttendanceSession(tmp_path / "a.csv")

    pending = session.update([face("Alice"), face(UNKNOWN)])

    assert pending == set()
    assert session.marked == {"Alice": True}


def test_session_with_liveness_waits_for_head_turn(tmp_path):
    path = tmp_path / "a.csv"
    session = AttendanceSession(path, LivenessSettings(0.3, 10), clock=Clock())

    for _ in range(30):  # a photo of Alice: recognised, but the cue never moves
        assert session.update([face("Alice", 0.05)]) == {"Alice"}
    assert session.marked == {}
    assert not path.exists()
    assert session.pending_deadline() is not None

    for cue in [-0.2] * 4 + [0.2] * 4:  # the real Alice turns her head
        session.update([face("Alice", cue)])
    assert session.marked == {"Alice": True}
    assert session.update([face("Alice", 0.0)]) == set()  # already done
    assert session.pending_deadline() is None


def test_session_tracks_people_independently(tmp_path):
    session = AttendanceSession(tmp_path / "a.csv", LivenessSettings(0.3, 10), clock=Clock())

    for cue in [-0.2] * 4 + [0.2] * 4:
        session.update([face("Alice", cue), face("Bob", 0.1), face("Carol")])

    assert session.marked == {"Alice": True}
    assert set(session.challenges) == {"Alice", "Bob", "Carol"}
