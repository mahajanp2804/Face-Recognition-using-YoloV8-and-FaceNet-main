"""Recognition: find every face in a frame, name it, and mark attendance from a video source."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from facerec.attendance import attendance_exists, mark_attendance
from facerec.detector import Box, clamp_box, crop
from facerec.matcher import UNKNOWN, FaceMatcher

if TYPE_CHECKING:
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder

logger = logging.getLogger(__name__)

WINDOW_NAME = "Camera Feed"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 200, 0)
RED = (0, 0, 255)


@dataclass(frozen=True)
class FaceResult:
    box: Box
    name: str  # UNKNOWN if not recognised
    distance: float  # L2 distance to the closest enrolled face (inf if nobody is enrolled)

    @property
    def known(self) -> bool:
        return self.name != UNKNOWN


def recognize_faces(
    frame: np.ndarray, detector: FaceDetector, embedder: FaceEmbedder, matcher: FaceMatcher
) -> list[FaceResult]:
    """Detect, align, embed and match every face in ``frame`` (most confident first).

    MTCNN doubles as a second-stage check: YOLO boxes it cannot align a face in are
    treated as false positives and dropped.
    """
    h, w = frame.shape[:2]
    results = []
    for raw in detector.detect(frame):
        box = clamp_box(raw, w, h)
        if box is None:
            continue
        embedding = embedder.embed(crop(frame, box))
        if embedding is None:
            continue
        name, dist = matcher.match(embedding)
        results.append(FaceResult(box, name, dist))
    return results


def draw_results(frame: np.ndarray, results: Iterable[FaceResult]) -> np.ndarray:
    """Draw a labelled box per face onto ``frame`` (in place) and return it."""
    for r in results:
        x1, y1, x2, y2 = r.box
        color = GREEN if r.known else RED
        label = f"{r.name} ({r.distance:.2f})" if r.known else UNKNOWN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        y_text = max(th + 6, y1)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text), color, -1)
        cv2.putText(frame, label, (x1 + 3, y_text - 4), FONT, 0.6, (255, 255, 255), 2)
    return frame


def record_attendance(names: Iterable[str], attendance_file: Path) -> dict[str, bool]:
    """Mark each distinct name once per day. Returns ``{name: newly_marked}``."""
    status = {}
    for name in dict.fromkeys(names):  # de-duplicate, keep order
        if attendance_exists(name, attendance_file):
            logger.info("Attendance already marked for %s today.", name)
            status[name] = False
        else:
            mark_attendance(name, attendance_file)
            status[name] = True
    return status


def _banner(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (20, 35), FONT, 0.9, color, 2)


def run(
    source: int | str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    attendance_file: Path,
    timer_seconds: float,
    continuous: bool = False,
    show: bool = True,
) -> list[str]:
    """Recognise faces from ``source`` (webcam index or video file) and mark attendance.

    Kiosk mode (default): stop as soon as a frame contains at least one known face, mark
    everyone known in that frame, or give up after ``timer_seconds``.
    Continuous mode: keep going until ``q`` or the end of the video, marking each new person.
    Returns the names recognised, in the order first seen.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"Cannot open video source {source!r}.")

    start_time = time.time()
    seen: dict[str, bool] = {}
    fps = 0.0

    mode = "continuous" if continuous else "kiosk"
    logger.info("Video started (%s mode). Searching for faces...", mode)

    try:
        while continuous or time.time() - start_time <= timer_seconds:
            ret, frame = cap.read()
            if not ret:
                if not continuous:
                    logger.error("Could not read a frame from the video source.")
                break

            t0 = time.perf_counter()
            results = recognize_faces(frame, detector, embedder, matcher)
            dt = max(time.perf_counter() - t0, 1e-6)
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt  # smoothed processing FPS
            for r in results:
                logger.debug("Face at %s: %s (distance %.3f)", r.box, r.name, r.distance)

            new_names = [r.name for r in results if r.known and r.name not in seen]
            if new_names:
                seen.update(record_attendance(new_names, attendance_file))

            if show:
                draw_results(frame, results)
                cv2.putText(frame, f"{fps:.1f} FPS", (20, frame.shape[0] - 15), FONT, 0.6, GREEN, 2)

            if not continuous and seen:
                if show:
                    marked = [n for n, new in seen.items() if new]
                    already = [n for n, new in seen.items() if not new]
                    text = "Marked: " + ", ".join(marked) if marked else ""
                    if already:
                        text += ("  " if text else "") + "Already: " + ", ".join(already)
                    _banner(frame, text, GREEN)
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(2000)  # show the result for 2 s before closing
                break

            if show:
                if results and not any(r.known for r in results):
                    _banner(frame, "Face Not Recognised", RED)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    if not seen:
        logger.info("No known face recognised — camera off.")
    return list(seen)
