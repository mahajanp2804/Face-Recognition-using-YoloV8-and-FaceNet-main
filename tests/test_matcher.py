import math

import numpy as np

from facerec.matcher import UNKNOWN, compare_embeddings, load_embeddings, save_embeddings


def unit(index: int, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


def test_returns_closest_person_within_threshold():
    db = {"Alice": [unit(0), unit(1)], "Bob": [unit(2)]}
    query = unit(1) + 0.01 * unit(5)

    name, dist = compare_embeddings(query, db, threshold=0.8)

    assert name == "Alice"
    assert dist == np.linalg.norm(query - unit(1))


def test_picks_nearest_across_people():
    db = {"Alice": [unit(0)], "Bob": [unit(2)]}
    query = 0.9 * unit(2) + 0.1 * unit(0)

    name, _ = compare_embeddings(query, db, threshold=0.8)

    assert name == "Bob"


def test_unknown_when_closest_is_above_threshold():
    db = {"Alice": [unit(0)]}
    query = unit(1)  # distance sqrt(2) ~ 1.41

    name, dist = compare_embeddings(query, db, threshold=0.8)

    assert name == UNKNOWN
    assert math.isclose(dist, math.sqrt(2), rel_tol=1e-6)


def test_threshold_is_exclusive():
    db = {"Alice": [np.zeros(512)]}
    query = np.full(512, 0.0)
    query[0] = 0.8  # distance exactly 0.8

    name, _ = compare_embeddings(query, db, threshold=0.8)

    assert name == UNKNOWN


def test_empty_database_is_unknown():
    name, dist = compare_embeddings(unit(0), {}, threshold=0.8)

    assert name == UNKNOWN
    assert dist == float("inf")


def test_person_with_no_embeddings_is_ignored():
    name, _ = compare_embeddings(unit(0), {"Alice": [], "Bob": [unit(0)]}, threshold=0.8)

    assert name == "Bob"


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "known_embeddings.pkl"
    db = {"Alice": [unit(0), unit(1)]}

    save_embeddings(db, path)
    loaded = load_embeddings(path)

    assert list(loaded) == ["Alice"]
    assert all(np.array_equal(a, b) for a, b in zip(loaded["Alice"], db["Alice"]))


def test_load_missing_file_returns_empty_db(tmp_path):
    assert load_embeddings(tmp_path / "missing.pkl") == {}
