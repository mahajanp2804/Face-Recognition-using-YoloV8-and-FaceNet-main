"""Build the embedding database from a folder of photos (one sub-folder per person)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from facerec.matcher import EmbeddingDB, load_embeddings, save_embeddings

if TYPE_CHECKING:
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def enroll_directory(
    faces_dir: Path,
    embeddings_path: Path,
    detector: FaceDetector,
    embedder: FaceEmbedder,
) -> EmbeddingDB:
    """Embed every face in ``faces_dir/<name>/*`` and save them to ``embeddings_path``.

    People already in ``embeddings_path`` are kept; anyone with a folder in ``faces_dir``
    is replaced by the newly computed embeddings. Returns the saved database.
    """
    known_embeddings = load_embeddings(embeddings_path)

    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        logger.info("Processing '%s' directory...", name)
        person_embeddings = []

        for image_path in sorted(person_dir.iterdir()):
            if not image_path.name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                logger.error("  Unable to read image '%s', skipping.", image_path.name)
                continue

            faces = detector.crop_faces(img)
            if not faces:
                logger.warning("  No faces detected in '%s', skipping.", image_path.name)
                continue

            embeddings = [e for e in (embedder.embed(f) for f in faces) if e is not None]
            person_embeddings.extend(embeddings)
            logger.info("  %d embedding(s) from '%s'.", len(embeddings), image_path.name)

        if person_embeddings:
            known_embeddings[name] = person_embeddings
            logger.info("Saved %d embedding(s) for '%s'.", len(person_embeddings), name)
        else:
            logger.warning("No usable faces found for '%s', not saved.", name)

    save_embeddings(known_embeddings, embeddings_path)
    return known_embeddings
