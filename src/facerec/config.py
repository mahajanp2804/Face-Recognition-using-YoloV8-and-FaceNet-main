"""Load settings from ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = "config.yaml"

DEFAULT_PATHS = {
    "faces_dir": "known_faces",
    "embeddings": "known_embeddings.pkl",
    "yolo_weights": "detection/weights/best.pt",
    "attendance": "attendance.csv",
}
DEFAULT_RECOGNITION_THRESHOLD = 0.8
DEFAULT_TIMER_SECONDS = 5.0
DEFAULT_CAMERA_INDEX = 0
# Mirrors facerec.liveness defaults (not imported, so config stays dependency-free)
DEFAULT_LIVENESS = {"enabled": False, "required_span": 0.30, "timeout_seconds": 10.0}

_TOP_LEVEL_KEYS = {"paths", "recognition_threshold", "timer_seconds", "camera_index", "liveness"}


@dataclass(frozen=True)
class Config:
    faces_dir: Path
    embeddings_path: Path
    yolo_weights: Path
    attendance_file: Path
    recognition_threshold: float = DEFAULT_RECOGNITION_THRESHOLD
    timer_seconds: float = DEFAULT_TIMER_SECONDS
    camera_index: int = DEFAULT_CAMERA_INDEX
    liveness_enabled: bool = DEFAULT_LIVENESS["enabled"]
    liveness_required_span: float = DEFAULT_LIVENESS["required_span"]
    liveness_timeout_seconds: float = DEFAULT_LIVENESS["timeout_seconds"]


def load_config(path: str | Path | None = None) -> Config:
    """Read settings from ``path``, falling back to defaults for anything not set.

    With no ``path``, ``config.yaml`` in the current directory is used if it exists.
    Relative paths in the file are resolved against the file's own folder (or the
    current directory when no file is used).
    """
    if path is None:
        candidate = Path.cwd() / DEFAULT_CONFIG_NAME
        path = candidate if candidate.is_file() else None

    data: dict[str, Any] = {}
    base_dir = Path.cwd()
    if path is not None:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        base_dir = path.resolve().parent

    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(sorted(unknown))}")

    paths = {**DEFAULT_PATHS, **(data.get("paths") or {})}
    unknown_paths = set(paths) - set(DEFAULT_PATHS)
    if unknown_paths:
        raise ValueError(f"Unknown key(s) under 'paths': {', '.join(sorted(unknown_paths))}")

    liveness = {**DEFAULT_LIVENESS, **(data.get("liveness") or {})}
    unknown_liveness = set(liveness) - set(DEFAULT_LIVENESS)
    if unknown_liveness:
        raise ValueError(f"Unknown key(s) under 'liveness': {', '.join(sorted(unknown_liveness))}")

    def resolve(value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else base_dir / p

    return Config(
        faces_dir=resolve(paths["faces_dir"]),
        embeddings_path=resolve(paths["embeddings"]),
        yolo_weights=resolve(paths["yolo_weights"]),
        attendance_file=resolve(paths["attendance"]),
        recognition_threshold=float(
            data.get("recognition_threshold", DEFAULT_RECOGNITION_THRESHOLD)
        ),
        timer_seconds=float(data.get("timer_seconds", DEFAULT_TIMER_SECONDS)),
        camera_index=int(data.get("camera_index", DEFAULT_CAMERA_INDEX)),
        liveness_enabled=bool(liveness["enabled"]),
        liveness_required_span=float(liveness["required_span"]),
        liveness_timeout_seconds=float(liveness["timeout_seconds"]),
    )
