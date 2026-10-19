"""Enrollment pipeline with fake models: checks the known_embeddings.pkl format without weights."""

import pickle

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from facerec.enroll import enroll_directory  # noqa: E402


class FakeDetector:
    """Treats the whole image as one face, except images that are entirely black."""

    def crop_faces(self, image):
        return [image] if image.any() else []


class FakeEmbedder:
    """Returns a 512-d vector derived from the mean pixel value."""

    def embed(self, face_bgr):
        return np.full(512, face_bgr.mean() / 255.0, dtype=np.float32)


def write_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((32, 32, 3), value, dtype=np.uint8))


def test_enroll_writes_name_to_list_of_512d_arrays(tmp_path):
    faces = tmp_path / "known_faces"
    write_image(faces / "Alice Smith" / "01.jpg", 100)
    write_image(faces / "Alice Smith" / "02.png", 150)
    write_image(faces / "Bob Lee" / "01.jpg", 200)
    write_image(faces / "Nobody" / "blank.jpg", 0)  # no face detected -> not saved
    (faces / "Bob Lee" / "notes.txt").write_text("not an image")
    (faces / "README.md").write_text("not a person folder")
    output = tmp_path / "known_embeddings.pkl"

    enroll_directory(faces, output, FakeDetector(), FakeEmbedder())

    with output.open("rb") as f:
        db = pickle.load(f)
    assert isinstance(db, dict)
    assert sorted(db) == ["Alice Smith", "Bob Lee"]
    assert len(db["Alice Smith"]) == 2
    assert len(db["Bob Lee"]) == 1
    for embeddings in db.values():
        assert isinstance(embeddings, list)
        for e in embeddings:
            assert isinstance(e, np.ndarray)
            assert e.shape == (512,)


def test_enroll_keeps_people_not_in_folder(tmp_path):
    output = tmp_path / "known_embeddings.pkl"
    with output.open("wb") as f:
        pickle.dump({"Carol": [np.zeros(512)]}, f)
    faces = tmp_path / "known_faces"
    write_image(faces / "Bob Lee" / "01.jpg", 200)

    db = enroll_directory(faces, output, FakeDetector(), FakeEmbedder())

    assert sorted(db) == ["Bob Lee", "Carol"]
