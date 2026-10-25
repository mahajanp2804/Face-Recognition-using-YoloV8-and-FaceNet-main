"""Per-stage latency benchmark: detection, alignment, embedding, matching and the full frame."""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from facerec.detector import clamp_box, crop

if TYPE_CHECKING:
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder
    from facerec.matcher import FaceMatcher

logger = logging.getLogger(__name__)

STAGES = ("detect", "align", "embed", "match", "frame")
STAGE_LABELS = {
    "detect": "YOLOv8n face detection (per frame)",
    "align": "MTCNN alignment (per face)",
    "embed": "FaceNet embedding (per face)",
    "match": "Nearest-neighbour match (per face)",
    "frame": "Full pipeline (per frame)",
}


def summarize(samples_ms: Sequence[float]) -> dict[str, float]:
    a = np.asarray(samples_ms, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


@dataclass
class BenchmarkResult:
    device: str
    frames: int
    faces: int
    samples: dict[str, list[float]] = field(default_factory=lambda: {s: [] for s in STAGES})

    @property
    def fps(self) -> float:
        mean = summarize(self.samples["frame"])["mean"]
        return 1000.0 / mean if mean and mean == mean else float("nan")

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "platform": f"{platform.system()} {platform.machine()}",
            "frames": self.frames,
            "faces": self.faces,
            "fps": self.fps,
            "stages_ms": {s: summarize(self.samples[s]) for s in STAGES},
        }

    def to_markdown(self) -> str:
        lines = ["| Stage | Mean (ms) | p50 (ms) | p95 (ms) |", "| --- | ---: | ---: | ---: |"]
        for s in STAGES:
            st = summarize(self.samples[s])
            lines.append(
                f"| {STAGE_LABELS[s]} | {st['mean']:.1f} | {st['p50']:.1f} | {st['p95']:.1f} |"
            )
        lines.append(
            f"\n{self.frames} frames, {self.faces} faces, device: `{self.device}` "
            f"({platform.system()} {platform.machine()}) → **{self.fps:.1f} FPS** end to end."
        )
        return "\n".join(lines)


def run_benchmark(
    frames: Sequence[np.ndarray],
    detector: FaceDetector,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    warmup: int = 3,
) -> BenchmarkResult:
    """Time each stage on ``frames``. The first ``warmup`` frames are run but not recorded."""
    result = BenchmarkResult(device=str(embedder.device), frames=0, faces=0)

    for i, frame in enumerate(frames):
        record = i >= warmup
        t = {s: 0.0 for s in STAGES}
        per_face: dict[str, list[float]] = {"align": [], "embed": [], "match": []}

        start = time.perf_counter()
        boxes = detector.detect(frame)
        t["detect"] = time.perf_counter() - start

        h, w = frame.shape[:2]
        for raw in boxes:
            box = clamp_box(raw, w, h)
            if box is None:
                continue
            t0 = time.perf_counter()
            aligned = embedder.align(crop(frame, box))
            per_face["align"].append(time.perf_counter() - t0)
            if aligned is None:
                continue
            t0 = time.perf_counter()
            embedding = embedder.embed_aligned(aligned)
            per_face["embed"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            matcher.match(embedding)
            per_face["match"].append(time.perf_counter() - t0)

        t["frame"] = time.perf_counter() - start

        if record:
            result.frames += 1
            result.faces += len(per_face["embed"])
            result.samples["detect"].append(t["detect"] * 1000)
            result.samples["frame"].append(t["frame"] * 1000)
            for s, values in per_face.items():
                result.samples[s].extend(v * 1000 for v in values)

    if result.frames == 0:
        raise ValueError(
            f"Need more than {warmup} frames to benchmark (warm-up frames are not timed)"
        )
    return result
