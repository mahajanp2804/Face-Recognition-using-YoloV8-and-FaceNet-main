"""Head-turn liveness: the affine-invariant depth cue and the per-person challenge."""

import math

import numpy as np
import pytest

from facerec.liveness import (
    DEFAULT_REQUIRED_SPAN,
    LivenessChallenge,
    depth_cue,
)

# Approximate 3-D landmark positions in mm (x right, y up, z towards the camera):
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
FACE_3D = np.array(
    [[-31.5, 35, -27], [31.5, 35, -27], [0, 0, 0], [-25, -30, -15], [25, -30, -15]], float
)


def project(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Perspective projection of FACE_3D turned by ``yaw`` (and nodded by ``pitch``)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    q = FACE_3D @ (rx @ ry).T
    f, distance = 800, 500
    return np.c_[f * q[:, 0] / (distance - q[:, 2]), -f * q[:, 1] / (distance - q[:, 2])]


# ---------------------------------------------------------------- depth cue


def test_frontal_face_has_zero_cue():
    assert depth_cue(project(0)) == pytest.approx(0.0, abs=1e-9)


def test_flat_photo_cue_is_invariant_to_any_affine_transform():
    """Moving, rotating, scaling, shearing or tilting a flat photo can't change the cue."""
    photo = project(12)  # the person in the photo is slightly turned
    expected = depth_cue(photo)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-1.5, 1.5, (2, 2)) + np.eye(2) * 2  # random invertible-ish linear map
        if abs(np.linalg.det(a)) < 0.1:
            continue
        t = rng.uniform(-300, 300, 2)
        assert depth_cue(photo @ a.T + t) == pytest.approx(expected, abs=1e-9)


def test_real_head_turn_changes_cue_with_sign():
    cues = {yaw: depth_cue(project(yaw)) for yaw in (-30, -15, 0, 15, 30)}

    assert cues[-30] < cues[-15] < cues[0] < cues[15] < cues[30]
    assert cues[-30] == pytest.approx(-cues[30], rel=1e-6)
    # turning about ±30° clears the default threshold; a small ±10° wobble does not
    assert cues[30] - cues[-30] > DEFAULT_REQUIRED_SPAN
    assert depth_cue(project(10)) - depth_cue(project(-10)) < DEFAULT_REQUIRED_SPAN


def test_cue_tolerates_nodding():
    assert depth_cue(project(0, pitch_deg=20)) == pytest.approx(0.0, abs=1e-9)
    assert depth_cue(project(25, pitch_deg=15)) == pytest.approx(depth_cue(project(25)), abs=0.03)


def test_degenerate_landmarks_give_nan():
    collinear = np.array([[0, 0], [10, 0], [5, 0], [2, 0], [8, 0]], float)
    assert math.isnan(depth_cue(collinear))


# ---------------------------------------------------------------- challenge


def feed(challenge, cues, fps=30.0, start=0.0):
    passed_at = None
    for i, cue in enumerate(cues):
        if challenge.update(cue, start + i / fps) and passed_at is None:
            passed_at = i
    return passed_at


def test_head_turn_left_and_right_passes():
    yaws = list(range(0, -31, -3)) + list(range(-30, 31, 3))
    challenge = LivenessChallenge()

    passed_at = feed(challenge, [depth_cue(project(y)) for y in yaws])

    assert passed_at is not None
    assert challenge.passed and challenge.progress() == 1.0


def test_static_face_with_landmark_noise_never_passes():
    rng = np.random.default_rng(1)
    challenge = LivenessChallenge()

    assert feed(challenge, 0.05 + rng.normal(0, 0.03, 300)) is None
    assert challenge.progress() < 1.0


def test_single_frame_glitch_is_filtered_out():
    cues = [0.0] * 10 + [0.6] + [0.0] * 10  # one wild landmark estimate
    assert feed(LivenessChallenge(), cues) is None


def test_missing_landmarks_are_skipped():
    cues = [-0.2, math.nan, -0.2, -0.2, math.nan, 0.2, 0.2, 0.2]
    assert feed(LivenessChallenge(required_span=0.3), cues) is not None


def test_slow_attempt_restarts_after_timeout():
    challenge = LivenessChallenge(required_span=0.3, timeout_seconds=2.0)
    feed(challenge, [-0.2] * 5, fps=1.0)  # turned one way, then took too long
    assert not challenge.passed

    feed(challenge, [0.2] * 5, fps=10.0, start=10.0)  # other way, but a fresh attempt
    assert not challenge.passed
    assert challenge.started == 10.0
