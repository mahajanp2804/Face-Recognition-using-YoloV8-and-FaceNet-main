"""Head-turn liveness check against printed-photo and screen-photo spoofing.

The cue comes from the 5 landmarks MTCNN already finds (eyes, nose, mouth corners), so it
needs no extra model.

A photo is flat. Whatever you do to a flat photo in front of the camera (move it, tilt it,
rotate it, bend the viewing angle), the image of it is, to a good approximation, an *affine*
transform of the original, and affine transforms preserve affine coordinates. So if we write
the nose position in the coordinate frame spanned by the eyes and mouth,

    nose - eye_mid = a * (right_eye - left_eye) + b * (mouth_mid - eye_mid)

then ``a`` stays constant for a photo, no matter how it is held. A real head is 3-D: the
nose sticks out in front of the eye/mouth plane, so when the person turns their head the nose
moves sideways relative to the eyes and mouth and ``a`` changes (about 0.35 · tan(yaw)
for typical face proportions, so ±30° covers a span of about 0.4).

The challenge asks the person to turn their head left and right; they pass once ``a`` has
spanned a wide enough range within the time limit. This defeats printed photos and still
images on a phone. It does NOT stop a replayed video of the person turning their head, or a
3-D mask; those need a dedicated anti-spoofing model.

Calibration (README, "Liveness check"): MTCNN landmark noise is about 0.01-0.02 per frame;
a printed photo tilted and rotated in front of the camera reached a smoothed span of at most
0.20 in simulation, while a real ±30° head turn spans about 0.4. The default threshold of
0.30 sits between the two.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DEFAULT_REQUIRED_SPAN = 0.30  # about ±25° of head turn (see module docstring)
DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class LivenessSettings:
    required_span: float = DEFAULT_REQUIRED_SPAN
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def depth_cue(landmarks: np.ndarray) -> float:
    """Affine-invariant sideways nose offset ``a`` (see module docstring); NaN if degenerate.

    ``landmarks`` is MTCNN's ``(5, 2)`` array: left eye, right eye, nose, left mouth corner,
    right mouth corner.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    left_eye, right_eye, nose, mouth_left, mouth_right = pts
    eye_mid = (left_eye + right_eye) / 2
    u = right_eye - left_eye
    v = (mouth_left + mouth_right) / 2 - eye_mid
    basis = np.column_stack([u, v])
    det = np.linalg.det(basis)
    if abs(det) < 1e-6 * (np.dot(u, u) + np.dot(v, v)):
        return math.nan  # eyes and mouth collinear: not a usable face
    a, _ = np.linalg.solve(basis, nose - eye_mid)
    return float(a)


@dataclass
class LivenessChallenge:
    """Tracks one person's head-turn challenge across frames.

    Each frame's cue is median-filtered over ``window`` frames so a single landmark glitch
    can't fake a turn. The person passes once the filtered cue has covered ``required_span``.
    An attempt that takes longer than ``timeout_seconds`` starts over.
    """

    required_span: float = DEFAULT_REQUIRED_SPAN
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    window: int = 3
    started: float | None = None
    passed: bool = False
    low: float = math.inf
    high: float = -math.inf
    _recent: deque = field(default_factory=deque, repr=False)

    def _restart(self, now: float) -> None:
        self.started = now
        self.low, self.high = math.inf, -math.inf
        self._recent.clear()

    @property
    def span(self) -> float:
        return self.high - self.low if self.high >= self.low else 0.0

    @property
    def deadline(self) -> float | None:
        return None if self.started is None else self.started + self.timeout_seconds

    def update(self, cue: float, now: float) -> bool:
        """Feed one frame's cue (NaN = no landmarks this frame). Returns True once passed."""
        if self.passed:
            return True
        if self.started is None or now - self.started > self.timeout_seconds:
            self._restart(now)
        if math.isnan(cue):
            return False

        self._recent.append(cue)
        if len(self._recent) > self.window:
            self._recent.popleft()
        if len(self._recent) < self.window:
            return False

        smoothed = float(np.median(self._recent))
        self.low, self.high = min(self.low, smoothed), max(self.high, smoothed)
        self.passed = self.span >= self.required_span
        return self.passed

    def progress(self) -> float:
        """Fraction of the required head turn done so far (0-1)."""
        return min(1.0, self.span / self.required_span) if self.required_span > 0 else 1.0
