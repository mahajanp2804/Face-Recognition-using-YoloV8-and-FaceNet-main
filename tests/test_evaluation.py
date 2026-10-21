"""Verification metrics (ROC, AUC, EER, TAR@FAR, k-fold accuracy) and pair lists, without models."""

import math

import numpy as np
import pytest

pytest.importorskip("cv2")  # facerec.evaluation imports facerec.enroll, which needs OpenCV

from facerec.evaluation import (  # noqa: E402
    accuracy_at,
    auc,
    best_threshold,
    equal_error_rate,
    kfold_accuracy,
    make_pairs,
    parse_lfw_pairs,
    roc_curve,
    tar_at_far,
    verification_report,
)


def separable(n=50):
    """Genuine pairs at distance 0.3-0.6, impostors at 1.0-1.4: perfectly separable."""
    d = np.r_[np.linspace(0.3, 0.6, n), np.linspace(1.0, 1.4, n)]
    same = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    return d, same


def test_roc_perfect_separation():
    d, same = separable()
    fpr, tpr, thresholds = roc_curve(d, same)

    assert fpr[0] == 0 and tpr[0] == 0
    assert fpr[-1] == 1 and tpr[-1] == 1
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    assert auc(fpr, tpr) == pytest.approx(1.0)
    assert equal_error_rate(fpr, tpr) == pytest.approx(0.0)
    assert tar_at_far(fpr, tpr, 1e-3) == pytest.approx(1.0)


def test_roc_handles_ties_and_reversed_scores():
    # Worst case: all impostors closer than all genuine pairs.
    d = np.array([0.1, 0.1, 0.2, 0.9, 0.9, 1.0])
    same = np.array([False, False, False, True, True, True])
    fpr, tpr, _ = roc_curve(d, same)

    assert auc(fpr, tpr) == pytest.approx(0.0)
    assert equal_error_rate(fpr, tpr) == pytest.approx(1.0)
    assert tar_at_far(fpr, tpr, 0.01) == 0.0
    # tied distances collapse into one ROC point
    assert len(fpr) == 1 + len(np.unique(d))


def test_eer_for_overlapping_distributions():
    rng = np.random.default_rng(0)
    d = np.r_[rng.normal(0.6, 0.15, 5000), rng.normal(1.2, 0.15, 5000)]
    same = np.r_[np.ones(5000, bool), np.zeros(5000, bool)]
    fpr, tpr, _ = roc_curve(d, same)

    # Two equal-variance Gaussians 4 sigma apart: EER = Phi(-2) ~ 2.3%
    assert equal_error_rate(fpr, tpr) == pytest.approx(0.0228, abs=0.006)
    assert 0.99 < auc(fpr, tpr) < 1.0


def test_roc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_curve(np.array([0.1, 0.2]), np.array([True, True]))


def test_accuracy_and_best_threshold():
    d, same = separable()

    assert accuracy_at(d, same, 0.8) == 1.0
    assert accuracy_at(d, same, 0.0) == 0.5
    t, acc = best_threshold(d, same)
    assert 0.6 < t < 1.0
    assert acc == 1.0


def test_kfold_accuracy_on_separable_data():
    d, same = separable(300)
    order = np.random.default_rng(1).permutation(len(d))
    mean, std = kfold_accuracy(d[order], same[order], folds=10)

    assert mean == 1.0
    assert std == 0.0


def test_verification_report_fields_and_markdown():
    d, same = separable()
    report = verification_report(d, same, configured_threshold=0.8, pairs_skipped=2, folds=5)

    assert report.pairs == 100 and report.genuine == 50 and report.impostor == 50
    assert report.pairs_skipped == 2
    assert report.folds == 5
    assert report.accuracy_at_configured_threshold == 1.0
    assert math.isclose(report.roc_auc, 1.0)
    md = report.to_markdown()
    assert md.startswith("| Metric | Value |")
    assert "TAR @ FAR = 0.1%" in md
    assert "5-fold" in md
    assert set(report.to_dict()) >= {"accuracy_kfold", "eer", "tar_at_far_1e3"}


def test_parse_lfw_pairs(tmp_path):
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("1\t2\nAlice_Smith\t1\t3\nAlice_Smith\t2\tBob_Lee\t11\n")

    pairs = parse_lfw_pairs(pairs_file, tmp_path / "lfw")

    assert pairs == [
        (
            tmp_path / "lfw/Alice_Smith/Alice_Smith_0001.jpg",
            tmp_path / "lfw/Alice_Smith/Alice_Smith_0003.jpg",
            True,
        ),
        (
            tmp_path / "lfw/Alice_Smith/Alice_Smith_0002.jpg",
            tmp_path / "lfw/Bob_Lee/Bob_Lee_0011.jpg",
            False,
        ),
    ]


def test_parse_lfw_pairs_rejects_bad_lines(tmp_path):
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("1 1\nonly two\n")
    with pytest.raises(ValueError):
        parse_lfw_pairs(pairs_file, tmp_path)


def test_make_pairs_is_balanced_and_reproducible(tmp_path):
    for person, n in {"Alice": 4, "Bob": 3, "Carol": 1}.items():
        (tmp_path / person).mkdir()
        for i in range(n):
            (tmp_path / person / f"{i}.jpg").write_bytes(b"")
    (tmp_path / "Bob" / "notes.txt").write_text("ignored")

    pairs = make_pairs(tmp_path, max_pairs=100, seed=3)

    genuine = [p for p in pairs if p[2]]
    impostor = [p for p in pairs if not p[2]]
    assert len(genuine) == 6 + 3  # C(4,2) + C(3,2)
    assert len(impostor) == len(genuine)
    assert all(a.parent == b.parent for a, b, _ in genuine)
    assert all(a.parent != b.parent for a, b, _ in impostor)
    assert make_pairs(tmp_path, max_pairs=100, seed=3) == pairs


def test_make_pairs_needs_two_people(tmp_path):
    (tmp_path / "Alice").mkdir()
    (tmp_path / "Alice" / "1.jpg").write_bytes(b"")
    (tmp_path / "Alice" / "2.jpg").write_bytes(b"")
    with pytest.raises(ValueError):
        make_pairs(tmp_path)
