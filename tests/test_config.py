from pathlib import Path

import pytest

from facerec.config import load_config


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.embeddings_path == tmp_path / "known_embeddings.pkl"
    assert config.yolo_weights == tmp_path / "detection" / "weights" / "best.pt"
    assert config.recognition_threshold == 0.8
    assert config.timer_seconds == 5
    assert config.camera_index == 0


def test_values_and_relative_paths_from_file(tmp_path):
    cfg = tmp_path / "conf" / "settings.yaml"
    cfg.parent.mkdir()
    cfg.write_text(
        "paths:\n"
        "  attendance: logs/attendance.csv\n"
        "  embeddings: /abs/db.pkl\n"
        "recognition_threshold: 0.6\n"
        "timer_seconds: 10\n"
        "camera_index: 1\n"
    )

    config = load_config(cfg)

    assert config.attendance_file == cfg.parent / "logs" / "attendance.csv"
    assert str(config.embeddings_path) == "/abs/db.pkl"
    assert config.faces_dir == cfg.parent / "known_faces"  # default, relative to the file
    assert config.recognition_threshold == 0.6
    assert config.timer_seconds == 10
    assert config.camera_index == 1


def test_repo_config_matches_defaults(monkeypatch, tmp_path):
    repo_config = Path(__file__).resolve().parents[1] / "config.yaml"
    monkeypatch.chdir(tmp_path)

    from_file = load_config(repo_config)
    defaults = load_config()

    assert from_file.recognition_threshold == defaults.recognition_threshold
    assert from_file.timer_seconds == defaults.timer_seconds
    assert from_file.camera_index == defaults.camera_index
    assert from_file.yolo_weights == repo_config.parent / "detection" / "weights" / "best.pt"


def test_unknown_key_is_rejected(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("recognition_treshold: 0.6\n")

    with pytest.raises(ValueError, match="recognition_treshold"):
        load_config(cfg)
