"""Benchmark timing loop with fake models."""

import math

import numpy as np
import pytest

from facerec.benchmark import STAGES, run_benchmark, summarize
from facerec.matcher import FaceMatcher


class FakeDetector:
    def detect(self, image):
        return [(0, 0, 10, 10), (10, 10, 20, 20)]


class FakeEmbedder:
    device = "cpu"

    def align(self, face):
        return None if face.mean() == 0 else face

    def embed_aligned(self, aligned):
        return np.ones(512, dtype=np.float32)


def test_summarize():
    s = summarize([1, 2, 3, 4, 100])
    assert s["n"] == 5
    assert s["p50"] == 3
    assert s["mean"] == pytest.approx(22)
    assert math.isnan(summarize([])["mean"])


def test_run_benchmark_counts_frames_and_faces_after_warmup():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:10, :10] = 255  # only the first box contains a "face"
    matcher = FaceMatcher({"Alice": [np.ones(512)]})

    result = run_benchmark([frame] * 5, FakeDetector(), FakeEmbedder(), matcher, warmup=2)

    assert result.frames == 3
    assert result.faces == 3
    assert len(result.samples["detect"]) == 3
    assert len(result.samples["align"]) == 6  # both boxes are aligned, one fails
    assert len(result.samples["embed"]) == 3
    assert result.fps > 0
    d = result.to_dict()
    assert set(d["stages_ms"]) == set(STAGES)
    assert "FPS" in result.to_markdown()


def test_run_benchmark_needs_frames_after_warmup():
    with pytest.raises(ValueError):
        run_benchmark(
            [np.zeros((5, 5, 3), np.uint8)],
            FakeDetector(),
            FakeEmbedder(),
            FaceMatcher({}),
            warmup=1,
        )
