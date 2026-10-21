"""Live recognition loop: find a known face in a video source and mark attendance."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from facerec.attendance import attendance_exists, mark_attendance
from facerec.detector import Box, clamp_box, crop
from facerec.matcher import UNKNOWN, EmbeddingDB, compare_embeddings

if TYPE_CHECKING:
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder

logger = logging.getLogger(__name__)

WINDOW_NAME = "Camera Feed"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def detect_and_embed(
    frame: np.ndarray, detector: FaceDetector, embedder: FaceEmbedder
) -> tuple[Box | None, np.ndarray | None]:
    """Return ``(box, embedding)`` for the most confident face in the frame.

    ``box`` is None if no face was detected; ``embedding`` is None if alignment failed.
    """
    boxes = detector.detect(frame)
    if not boxes:
        return None, None

    h, w = frame.shape[:2]
    box = clamp_box(boxes[0], w, h)
    if box is None:
        return None, None

    return box, embedder.embed(crop(frame, box))


def run(
    source: int | str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    known_embeddings: EmbeddingDB,
    attendance_file: Path,
    threshold: float,
    timer_seconds: float,
) -> str | None:
    """Watch ``source`` (webcam index or video file) until a known face is seen or time runs out.

    Marks attendance for the first recognised person (once per day) and returns their
    name, or None if nobody was recognised. Press ``q`` in the window to stop early.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"Cannot open video source {source!r}.")

    start_time = time.time()
    recognized_name = None

    logger.info("Camera started. Searching for face...")

    try:
        while time.time() - start_time <= timer_seconds:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read a frame from the video source.")
                break

            box, embedding = detect_and_embed(frame, detector, embedder)

            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            if embedding is not None:
                match, dist = compare_embeddings(embedding, known_embeddings, threshold)
                logger.debug("Closest match: %s (distance %.3f)", match, dist)

                if match != UNKNOWN:
                    recognized_name = match

                    if attendance_exists(recognized_name, attendance_file):
                        logger.info("Attendance already marked for %s today.", recognized_name)
                        msg = f"Already Marked: {recognized_name}"
                    else:
                        mark_attendance(recognized_name, attendance_file)
                        msg = f"Attendance Marked: {recognized_name}"

                    # Show the result on screen for 2 s before closing
                    cv2.putText(frame, msg, (30, 40), FONT, 1, (0, 255, 0), 2)
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(2000)
                    break

                cv2.putText(frame, "Face Not Recognised", (30, 40), FONT, 1, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if recognized_name is None:
        logger.info("Face not recognised — camera off.")
    return recognized_name
