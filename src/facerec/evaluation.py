"""Face verification evaluation: pair lists, ROC, EER, TAR@FAR and k-fold accuracy.

Pairs are ``(image_a, image_b, same_person)``. A pair is predicted "same" when the L2
distance between the two embeddings is below a threshold, so lower distance = more similar.
The metric functions only need NumPy and are unit-tested without any model.
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from facerec.enroll import IMAGE_EXTENSIONS

if TYPE_CHECKING:
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder

logger = logging.getLogger(__name__)

Pair = tuple[Path, Path, bool]


# ----------------------------------------------------------------- pair lists


def parse_lfw_pairs(pairs_file: Path, dataset_dir: Path) -> list[Pair]:
    """Read the official LFW ``pairs.txt`` (10 folds of 300 matched + 300 mismatched pairs).

    Matched lines are ``name n1 n2``; mismatched lines are ``name1 n1 name2 n2``.
    Images live at ``dataset_dir/name/name_0001.jpg``.
    """

    def image(name: str, n: str) -> Path:
        return dataset_dir / name / f"{name}_{int(n):04d}.jpg"

    pairs: list[Pair] = []
    lines = pairs_file.read_text().split("\n")
    for line in lines[1:]:  # first line is "<folds> <pairs per fold>"
        parts = line.split()
        if len(parts) == 3:
            pairs.append((image(parts[0], parts[1]), image(parts[0], parts[2]), True))
        elif len(parts) == 4:
            pairs.append((image(parts[0], parts[1]), image(parts[2], parts[3]), False))
        elif parts:
            raise ValueError(f"{pairs_file}: unexpected line {line!r}")
    return pairs


def make_pairs(dataset_dir: Path, max_pairs: int = 3000, seed: int = 0) -> list[Pair]:
    """Build a balanced pair list from a ``dataset_dir/<person>/*.jpg`` folder.

    Up to ``max_pairs // 2`` genuine pairs (two photos of the same person) and the same
    number of impostor pairs (photos of two different people), sampled with a fixed seed.
    """
    rng = random.Random(seed)
    people = {
        d.name: sorted(p for p in d.iterdir() if p.name.lower().endswith(IMAGE_EXTENSIONS))
        for d in sorted(dataset_dir.iterdir())
        if d.is_dir()
    }
    people = {name: imgs for name, imgs in people.items() if imgs}
    if len(people) < 2:
        raise ValueError(f"{dataset_dir}: need photos of at least 2 people to make pairs")

    genuine = [(a, b, True) for imgs in people.values() for a, b in itertools.combinations(imgs, 2)]
    rng.shuffle(genuine)
    genuine = genuine[: max_pairs // 2]
    if not genuine:
        raise ValueError(f"{dataset_dir}: need at least one person with 2+ photos")

    names = list(people)
    impostor: list[Pair] = []
    seen = set()
    attempts = 0
    while len(impostor) < len(genuine) and attempts < 100 * len(genuine):
        attempts += 1
        n1, n2 = rng.sample(names, 2)
        a, b = rng.choice(people[n1]), rng.choice(people[n2])
        if (a, b) not in seen:
            seen.add((a, b))
            impostor.append((a, b, False))

    pairs = genuine + impostor
    rng.shuffle(pairs)
    return pairs


# ----------------------------------------------------------------- metrics


def roc_curve(distances: np.ndarray, same: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr, thresholds)``, predicting "same" when distance <= threshold.

    Points run from the strictest threshold (nothing accepted) to the loosest (everything).
    """
    distances = np.asarray(distances, dtype=np.float64)
    same = np.asarray(same, dtype=bool)
    n_pos, n_neg = same.sum(), (~same).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both genuine and impostor pairs for a ROC curve")

    order = np.argsort(distances, kind="mergesort")
    d, s = distances[order], same[order]
    tp, fp = np.cumsum(s), np.cumsum(~s)
    # keep the last index of each run of equal distances
    last = np.r_[np.diff(d) != 0, True]
    tpr = np.r_[0.0, tp[last] / n_pos]
    fpr = np.r_[0.0, fp[last] / n_neg]
    thresholds = np.r_[-np.inf, d[last]]
    return fpr, tpr, thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def equal_error_rate(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Rate where false accepts equal false rejects (FPR == 1 - TPR), linearly interpolated."""
    fnr = 1 - tpr
    i = int(np.argmax(fpr >= fnr))
    if i == 0:
        return float(fpr[0])
    # interpolate between point i-1 (fpr < fnr) and i (fpr >= fnr)
    x0, x1 = fpr[i - 1] - fnr[i - 1], fpr[i] - fnr[i]
    w = x0 / (x0 - x1) if x0 != x1 else 0.0
    return float(fpr[i - 1] + w * (fpr[i] - fpr[i - 1]))


def tar_at_far(fpr: np.ndarray, tpr: np.ndarray, far: float) -> float:
    """Highest true-accept rate achievable with false-accept rate <= ``far``."""
    ok = fpr <= far
    return float(tpr[ok].max()) if ok.any() else 0.0


def accuracy_at(distances: np.ndarray, same: np.ndarray, threshold: float) -> float:
    return float(np.mean((np.asarray(distances) < threshold) == np.asarray(same, dtype=bool)))


def best_threshold(distances: np.ndarray, same: np.ndarray) -> tuple[float, float]:
    """Threshold (midpoint between neighbouring distances) that maximises accuracy."""
    d = np.unique(distances)
    candidates = np.r_[d[0] - 1e-6, (d[:-1] + d[1:]) / 2, d[-1] + 1e-6]
    accs = [accuracy_at(distances, same, t) for t in candidates]
    i = int(np.argmax(accs))
    return float(candidates[i]), float(accs[i])


def kfold_accuracy(distances: np.ndarray, same: np.ndarray, folds: int = 10) -> tuple[float, float]:
    """LFW-style protocol: pick the threshold on k-1 folds, measure accuracy on the held-out one.

    Folds are consecutive blocks, matching the order of the official ``pairs.txt``.
    Returns ``(mean accuracy, std)``.
    """
    distances = np.asarray(distances)
    same = np.asarray(same, dtype=bool)
    idx = np.array_split(np.arange(len(distances)), folds)
    accs = []
    for k in range(folds):
        train = np.concatenate([idx[j] for j in range(folds) if j != k])
        t, _ = best_threshold(distances[train], same[train])
        accs.append(accuracy_at(distances[idx[k]], same[idx[k]], t))
    return float(np.mean(accs)), float(np.std(accs))


@dataclass
class VerificationReport:
    pairs: int
    pairs_skipped: int
    genuine: int
    impostor: int
    folds: int
    accuracy_kfold: float
    accuracy_kfold_std: float
    best_threshold: float
    accuracy_at_best_threshold: float
    configured_threshold: float
    accuracy_at_configured_threshold: float
    roc_auc: float
    eer: float
    tar_at_far_1e2: float
    tar_at_far_1e3: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        rows = [
            (
                "Pairs evaluated",
                f"{self.pairs} ({self.genuine} genuine / {self.impostor} impostor)",
            ),
            ("Pairs skipped (no face found)", str(self.pairs_skipped)),
            (
                f"Accuracy ({self.folds}-fold cross-validation)",
                f"{self.accuracy_kfold:.2%} ± {self.accuracy_kfold_std:.2%}",
            ),
            ("ROC AUC", f"{self.roc_auc:.4f}"),
            ("Equal error rate", f"{self.eer:.2%}"),
            ("TAR @ FAR = 1%", f"{self.tar_at_far_1e2:.2%}"),
            ("TAR @ FAR = 0.1%", f"{self.tar_at_far_1e3:.2%}"),
            (
                "Best threshold (L2)",
                f"{self.best_threshold:.3f} → {self.accuracy_at_best_threshold:.2%} accuracy",
            ),
            (
                "Configured threshold (L2)",
                f"{self.configured_threshold:.3f} → "
                f"{self.accuracy_at_configured_threshold:.2%} accuracy",
            ),
        ]
        lines = ["| Metric | Value |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in rows]
        return "\n".join(lines)


def verification_report(
    distances: np.ndarray,
    same: np.ndarray,
    configured_threshold: float,
    pairs_skipped: int = 0,
    folds: int = 10,
) -> VerificationReport:
    distances = np.asarray(distances, dtype=np.float64)
    same = np.asarray(same, dtype=bool)
    fpr, tpr, _ = roc_curve(distances, same)
    acc, std = kfold_accuracy(distances, same, folds)
    t_best, acc_best = best_threshold(distances, same)
    return VerificationReport(
        pairs=len(distances),
        pairs_skipped=pairs_skipped,
        genuine=int(same.sum()),
        impostor=int((~same).sum()),
        folds=folds,
        accuracy_kfold=acc,
        accuracy_kfold_std=std,
        best_threshold=t_best,
        accuracy_at_best_threshold=acc_best,
        configured_threshold=configured_threshold,
        accuracy_at_configured_threshold=accuracy_at(distances, same, configured_threshold),
        roc_auc=auc(fpr, tpr),
        eer=equal_error_rate(fpr, tpr),
        tar_at_far_1e2=tar_at_far(fpr, tpr, 1e-2),
        tar_at_far_1e3=tar_at_far(fpr, tpr, 1e-3),
    )


# ----------------------------------------------------------------- running the models


def central_face_embedding(
    image: np.ndarray, detector: FaceDetector, embedder: FaceEmbedder
) -> np.ndarray | None:
    """Embed the detected face closest to the image centre (the subject in LFW-style crops).

    Falls back to aligning the whole image with MTCNN when YOLO finds no face.
    """
    from facerec.detector import clamp_box, crop

    h, w = image.shape[:2]
    boxes = [b for b in (clamp_box(raw, w, h) for raw in detector.detect(image)) if b]
    if boxes:
        cx, cy = w / 2, h / 2
        box = min(
            boxes, key=lambda b: ((b[0] + b[2]) / 2 - cx) ** 2 + ((b[1] + b[3]) / 2 - cy) ** 2
        )
        embedding = embedder.embed(crop(image, box))
        if embedding is not None:
            return embedding
    return embedder.embed(image)


def evaluate_pairs(
    pairs: list[Pair],
    detector: FaceDetector,
    embedder: FaceEmbedder,
    configured_threshold: float,
    folds: int = 10,
) -> tuple[VerificationReport, np.ndarray, np.ndarray]:
    """Embed every image once, score every pair, and return ``(report, distances, same)``."""
    import cv2

    images = sorted({p for a, b, _ in pairs for p in (a, b)})
    embeddings: dict[Path, np.ndarray | None] = {}
    for i, path in enumerate(images, 1):
        img = cv2.imread(str(path))
        embeddings[path] = None if img is None else central_face_embedding(img, detector, embedder)
        if i % 250 == 0 or i == len(images):
            logger.info("Embedded %d / %d images", i, len(images))

    distances, same, skipped = [], [], 0
    for a, b, is_same in pairs:
        ea, eb = embeddings[a], embeddings[b]
        if ea is None or eb is None:
            skipped += 1
            continue
        distances.append(float(np.linalg.norm(ea - eb)))
        same.append(is_same)

    if skipped:
        logger.warning("Skipped %d pair(s) where no face could be found.", skipped)
    d, s = np.array(distances), np.array(same, dtype=bool)
    report = verification_report(d, s, configured_threshold, skipped, folds)
    return report, d, s


def plot_roc(
    distances: np.ndarray, same: np.ndarray, report: VerificationReport, out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(distances, same)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(fpr, tpr, lw=2, color="#2f6fdf", label=f"AUC = {report.roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color="#999")
    ax1.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.01))
    ax1.set(xlabel="False accept rate (log scale)", ylabel="True accept rate", title="ROC curve")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    bins = np.linspace(0, max(1.6, float(distances.max())), 60)
    ax2.hist(distances[same], bins=bins, alpha=0.7, color="#2f9e44", label="Same person")
    ax2.hist(distances[~same], bins=bins, alpha=0.7, color="#d9480f", label="Different people")
    ax2.axvline(report.configured_threshold, color="#333", ls="--", lw=1.5, label="Threshold")
    ax2.set(xlabel="L2 distance between embeddings", ylabel="Pairs", title="Distance distribution")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
