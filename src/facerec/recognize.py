"""Recognition: find every face in a frame, name it, and mark attendance from a video source."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from facerec.attendance import attendance_exists, mark_attendance
from facerec.detector import Box, clamp_box, crop
from facerec.liveness import LivenessChallenge, LivenessSettings, depth_cue
from facerec.matcher import UNKNOWN, FaceMatcher

if TYPE_CHECKING:
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder

logger = logging.getLogger(__name__)

WINDOW_NAME = "Camera Feed"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 140, 255)


@dataclass(frozen=True)
class FaceResult:
    box: Box
    name: str  # UNKNOWN if not recognised
    distance: float  # L2 distance to the closest enrolled face (inf if nobody is enrolled)
    # MTCNN's 5 landmarks in frame coordinates, when requested (for the liveness check)
    landmarks: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def known(self) -> bool:
        return self.name != UNKNOWN


def recognize_faces(
    frame: np.ndarray,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    with_landmarks: bool = False,
) -> list[FaceResult]:
    """Detect, align, embed and match every face in ``frame`` (most confident first).

    MTCNN doubles as a second-stage check: YOLO boxes it cannot align a face in are
    treated as false positives and dropped. With ``with_landmarks``, each result also
    carries MTCNN's facial landmarks (used by the liveness check).
    """
    h, w = frame.shape[:2]
    results = []
    for raw in detector.detect(frame):
        box = clamp_box(raw, w, h)
        if box is None:
            continue
        face = crop(frame, box)
        landmarks = None
        if with_landmarks:
            embedding, landmarks = embedder.embed_with_landmarks(face)
            if landmarks is not None:
                landmarks = landmarks + np.array(box[:2], dtype=landmarks.dtype)
        else:
            embedding = embedder.embed(face)
        if embedding is None:
            continue
        name, dist = matcher.match(embedding)
        results.append(FaceResult(box, name, dist, landmarks))
    return results


def draw_results(
    frame: np.ndarray, results: Iterable[FaceResult], pending: Collection[str] = ()
) -> np.ndarray:
    """Draw a labelled box per face onto ``frame`` (in place) and return it.

    Names in ``pending`` are recognised but still doing the liveness check (orange).
    """
    for r in results:
        x1, y1, x2, y2 = r.box
        if not r.known:
            color, label = RED, UNKNOWN
        elif r.name in pending:
            color, label = ORANGE, f"{r.name}: turn head"
        else:
            color, label = GREEN, f"{r.name} ({r.distance:.2f})"
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


@dataclass
class AttendanceSession:
    """Decides, frame by frame, who gets marked present.

    Without liveness, a recognised person is marked straight away. With liveness, they
    are only marked once they pass a head-turn challenge (see :mod:`facerec.liveness`),
    so a photo of an enrolled person held up to the camera is recognised but not marked.
    """

    attendance_file: Path
    liveness: LivenessSettings | None = None
    clock: Callable[[], float] = time.monotonic
    marked: dict[str, bool] = field(default_factory=dict)  # name -> newly marked today
    challenges: dict[str, LivenessChallenge] = field(default_factory=dict)

    def update(self, results: Iterable[FaceResult]) -> set[str]:
        """Process one frame's results; returns names still pending the liveness check."""
        now = self.clock()
        ready, pending = [], set()
        for r in results:
            if not r.known or r.name in self.marked:
                continue
            if self.liveness is None:
                ready.append(r.name)
                continue
            challenge = self.challenges.setdefault(
                r.name,
                LivenessChallenge(self.liveness.required_span, self.liveness.timeout_seconds),
            )
            cue = depth_cue(r.landmarks) if r.landmarks is not None else math.nan
            if challenge.update(cue, now):
                logger.info("%s passed the liveness check.", r.name)
                ready.append(r.name)
            else:
                logger.debug(
                    "%s liveness: cue %.3f, span %.3f / %.3f",
                    r.name,
                    cue,
                    challenge.span,
                    challenge.required_span,
                )
                pending.add(r.name)
        if ready:
            self.marked.update(record_attendance(ready, self.attendance_file))
        return pending

    def pending_deadline(self) -> float | None:
        """Latest time an unfinished liveness challenge may still run until."""
        deadlines = [
            c.deadline for c in self.challenges.values() if not c.passed and c.deadline is not None
        ]
        return max(deadlines, default=None)


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
    liveness: LivenessSettings | None = None,
) -> list[str]:
    """Recognise faces from ``source`` (webcam index or video file) and mark attendance.

    Kiosk mode (default): stop as soon as someone is marked, or give up after
    ``timer_seconds`` (extended while a liveness challenge is in progress).
    Continuous mode: keep going until ``q`` or the end of the video, marking each new person.
    With ``liveness``, people must turn their head left and right before being marked.
    Returns the names marked (or found already marked), in the order first seen.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"Cannot open video source {source!r}.")

    session = AttendanceSession(attendance_file, liveness)
    deadline = time.monotonic() + timer_seconds
    fps = 0.0

    mode = "continuous" if continuous else "kiosk"
    check = "on" if liveness else "off"
    logger.info("Video started (%s mode, liveness %s). Searching for faces...", mode, check)

    try:
        while continuous or time.monotonic() <= deadline:
            ret, frame = cap.read()
            if not ret:
                if not continuous:
                    logger.error("Could not read a frame from the video source.")
                break

            t0 = time.perf_counter()
            results = recognize_faces(
                frame, detector, embedder, matcher, with_landmarks=liveness is not None
            )
            dt = max(time.perf_counter() - t0, 1e-6)
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt  # smoothed processing FPS
            for r in results:
                logger.debug("Face at %s: %s (distance %.3f)", r.box, r.name, r.distance)

            pending = session.update(results)
            extend_to = session.pending_deadline()
            if extend_to is not None:
                deadline = max(deadline, extend_to)

            if show:
                draw_results(frame, results, pending)
                cv2.putText(frame, f"{fps:.1f} FPS", (20, frame.shape[0] - 15), FONT, 0.6, GREEN, 2)

            if not continuous and session.marked:
                if show:
                    marked = [n for n, new in session.marked.items() if new]
                    already = [n for n, new in session.marked.items() if not new]
                    text = "Marked: " + ", ".join(marked) if marked else ""
                    if already:
                        text += ("  " if text else "") + "Already: " + ", ".join(already)
                    _banner(frame, text, GREEN)
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(2000)  # show the result for 2 s before closing
                break

            if show:
                if pending:
                    progress = min(session.challenges[n].progress() for n in pending)
                    _banner(frame, f"Turn your head left and right  {progress:.0%}", ORANGE)
                elif results and not any(r.known for r in results):
                    _banner(frame, "Face Not Recognised", RED)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    unverified = [n for n, c in session.challenges.items() if not c.passed]
    if unverified:
        logger.warning(
            "Recognised but did not pass the liveness check (not marked): %s",
            ", ".join(unverified),
        )
    if not session.marked:
        logger.info("Nobody was marked — camera off.")
    return list(session.marked)
