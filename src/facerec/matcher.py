"""Match a face embedding against the enrolled database, and load/save that database."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from facerec.config import DEFAULT_RECOGNITION_THRESHOLD

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"

# Person name -> list of 512-d FaceNet embeddings (one per enrolled face crop).
EmbeddingDB = dict[str, list[np.ndarray]]


def compare_embeddings(
    embedding: np.ndarray,
    db: EmbeddingDB,
    threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
) -> tuple[str, float]:
    """Return ``(name, distance)`` of the closest known embedding.

    The name is ``"Unknown"`` if the closest L2 distance is not below ``threshold``
    (or the database is empty, in which case the distance is ``inf``).
    """
    best_dist = float("inf")
    best_name = UNKNOWN

    for name, emb_list in db.items():
        for known in emb_list:
            dist = float(np.linalg.norm(embedding - known))
            if dist < best_dist:
                best_dist, best_name = dist, name

    if best_dist >= threshold:
        best_name = UNKNOWN
    return best_name, best_dist


def load_embeddings(path: Path) -> EmbeddingDB:
    """Load the embedding database, or return an empty one if the file does not exist."""
    if not path.exists():
        logger.info("No existing embeddings at '%s'; starting with an empty database.", path)
        return {}
    with path.open("rb") as f:
        db: EmbeddingDB = pickle.load(f)
    logger.info("Loaded embeddings for %d people from '%s'.", len(db), path)
    return db


def save_embeddings(db: EmbeddingDB, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(db, f)
    logger.info("Known embeddings saved to '%s' (%d people).", path, len(db))
