"""Command-line interface: ``python -m facerec {enroll,run,benchmark,evaluate}``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from facerec.config import Config, load_config

if TYPE_CHECKING:
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder

logger = logging.getLogger("facerec")


def _parse_source(value: str) -> int | str:
    """A bare integer is a webcam index; anything else is a video file path or URL."""
    return int(value) if value.isdigit() else value


def _load_models(config: Config) -> tuple[FaceDetector, FaceEmbedder]:
    # Imported here so `--help` and argument errors don't wait for torch/ultralytics.
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder

    detector = FaceDetector(config.yolo_weights)
    embedder = FaceEmbedder()
    logger.info("YOLOv8, MTCNN and InceptionResnetV1 models loaded (device: %s).", embedder.device)
    return detector, embedder


def cmd_enroll(args: argparse.Namespace, config: Config) -> int:
    from facerec.enroll import enroll_directory

    faces_dir = Path(args.dir) if args.dir else config.faces_dir
    output = Path(args.output) if args.output else config.embeddings_path

    if not faces_dir.is_dir():
        logger.error("'%s' is not a valid directory.", faces_dir)
        return 1

    detector, embedder = _load_models(config)
    enroll_directory(faces_dir, output, detector, embedder)
    return 0


def _load_matcher(config: Config, threshold: float | None = None):
    from facerec.matcher import FaceMatcher, load_embeddings

    db = load_embeddings(config.embeddings_path)
    return FaceMatcher(db, threshold if threshold is not None else config.recognition_threshold)


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    from facerec.recognize import run

    if not config.embeddings_path.exists():
        logger.error(
            "No embeddings found at '%s'. Run `python -m facerec enroll` first.",
            config.embeddings_path,
        )
        return 1

    matcher = _load_matcher(config, args.threshold)
    detector, embedder = _load_models(config)
    source = args.source if args.source is not None else config.camera_index

    try:
        run(
            source,
            detector,
            embedder,
            matcher,
            attendance_file=config.attendance_file,
            timer_seconds=config.timer_seconds,
            continuous=args.continuous,
            show=not args.no_window,
        )
    except OSError as e:
        logger.error("%s", e)
        return 1
    return 0


def _read_frames(args: argparse.Namespace) -> list:
    """Benchmark frames: ``--frames`` images cycled from ``--images``, or read from ``--video``."""
    import itertools

    import cv2

    from facerec.enroll import IMAGE_EXTENSIONS

    if args.video:
        cap = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(_resize_to_width(frame, args.width))
        cap.release()
        return frames

    paths = sorted(
        p for p in Path(args.images).rglob("*") if p.name.lower().endswith(IMAGE_EXTENSIONS)
    )
    images = [img for img in (cv2.imread(str(p)) for p in paths) if img is not None]
    frames = list(itertools.islice(itertools.cycle(images), args.frames))
    return [_resize_to_width(f, args.width) for f in frames]


def _resize_to_width(frame, width: int | None):
    import cv2

    h, w = frame.shape[:2]
    if not width or w == width:
        return frame
    return cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def cmd_benchmark(args: argparse.Namespace, config: Config) -> int:
    import json

    from facerec.benchmark import run_benchmark
    from facerec.matcher import FaceMatcher, load_embeddings

    frames = _read_frames(args)
    if len(frames) <= args.warmup:
        logger.error("Found %d frame(s); need more than --warmup=%d.", len(frames), args.warmup)
        return 1

    detector, embedder = _load_models(config)
    db = load_embeddings(config.embeddings_path)
    if not db:  # time matching against a database of realistic size even without enrolment
        import numpy as np

        rng = np.random.default_rng(0)
        db = {f"person_{i}": [rng.standard_normal(512).astype(np.float32)] for i in range(100)}
    matcher = FaceMatcher(db, config.recognition_threshold)

    logger.info("Benchmarking %d frames (%d warm-up)...", len(frames), args.warmup)
    result = run_benchmark(frames, detector, embedder, matcher, warmup=args.warmup)
    print(result.to_markdown())
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("Saved results to %s", args.json)
    return 0


def cmd_evaluate(args: argparse.Namespace, config: Config) -> int:
    import json

    from facerec.evaluation import evaluate_pairs, make_pairs, parse_lfw_pairs, plot_roc

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        logger.error("'%s' is not a valid directory.", dataset)
        return 1

    try:
        if args.pairs:
            pairs = parse_lfw_pairs(Path(args.pairs), dataset)
        else:
            pairs = make_pairs(dataset, max_pairs=args.max_pairs, seed=args.seed)
    except (OSError, ValueError) as e:
        logger.error("%s", e)
        return 1
    missing = sorted({str(p) for a, b, _ in pairs for p in (a, b) if not p.exists()})
    if missing:
        logger.error("%d image(s) in the pair list are missing, e.g. %s", len(missing), missing[0])
        return 1

    detector, embedder = _load_models(config)
    logger.info("Evaluating %d pairs from '%s'...", len(pairs), dataset)
    report, distances, same = evaluate_pairs(
        pairs, detector, embedder, config.recognition_threshold, folds=args.folds
    )
    print(report.to_markdown())

    if args.roc:
        plot_roc(distances, same, report, Path(args.roc))
        logger.info("Saved ROC curve to %s", args.roc)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Saved report to %s", args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        help="Path to config.yaml (default: ./config.yaml if present, else built-in defaults)",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")

    parser = argparse.ArgumentParser(
        prog="facerec",
        description="Face recognition attendance with YOLOv8 + FaceNet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enroll = sub.add_parser(
        "enroll",
        parents=[common],
        help="Generate embeddings from a folder of photos",
        description="Embed every face in DIR/<person name>/*.jpg and save them to the database.",
    )
    enroll.add_argument(
        "--dir", help="Folder with one sub-folder of photos per person (default: from config)"
    )
    enroll.add_argument("--output", help="Embeddings file to write (default: from config)")
    enroll.set_defaults(func=cmd_enroll)

    run = sub.add_parser(
        "run",
        parents=[common],
        help="Recognise a face and mark attendance",
        description="Watch a webcam or video until a known face is seen, then mark attendance.",
    )
    run.add_argument(
        "--source",
        type=_parse_source,
        help="Webcam index (e.g. 0) or path to a video file (default: camera_index from config)",
    )
    run.add_argument(
        "--continuous",
        action="store_true",
        help="Keep running and mark everyone who appears, until q or the end of the video",
    )
    run.add_argument(
        "--threshold", type=float, help="Override recognition_threshold from config (L2 distance)"
    )
    run.add_argument(
        "--no-window", action="store_true", help="Don't open a display window (headless)"
    )
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser(
        "benchmark",
        parents=[common],
        help="Time each pipeline stage and report FPS",
        description="Time detection, alignment, embedding and matching on images or a video.",
    )
    src = bench.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="Folder of images (searched recursively, cycled if needed)")
    src.add_argument("--video", help="Video file to read frames from")
    bench.add_argument("--frames", type=int, default=100, help="Frames to time (default: 100)")
    bench.add_argument(
        "--width",
        type=int,
        default=640,
        help="Resize frames to this width, like a webcam feed (default: 640; 0 = keep size)",
    )
    bench.add_argument("--warmup", type=int, default=5, help="Untimed warm-up frames (default: 5)")
    bench.add_argument("--json", help="Also save the results as JSON")
    bench.set_defaults(func=cmd_benchmark)

    evaluate = sub.add_parser(
        "evaluate",
        parents=[common],
        help="Measure verification accuracy (ROC, EER, TAR@FAR) on a labelled dataset",
        description=(
            "Score pairs of photos as same/different person. DATASET is a folder with one "
            "sub-folder per person (e.g. LFW). Use --pairs with LFW's pairs.txt for the "
            "standard 6,000-pair protocol; otherwise balanced pairs are sampled."
        ),
    )
    evaluate.add_argument("--dataset", required=True, help="Folder with one sub-folder per person")
    evaluate.add_argument("--pairs", help="LFW-format pairs.txt (default: sample pairs)")
    evaluate.add_argument(
        "--max-pairs", type=int, default=3000, help="Pairs to sample without --pairs"
    )
    evaluate.add_argument("--seed", type=int, default=0, help="Random seed for sampled pairs")
    evaluate.add_argument("--folds", type=int, default=10, help="Cross-validation folds")
    evaluate.add_argument("--roc", help="Save a ROC curve + distance histogram PNG here")
    evaluate.add_argument("--json", help="Also save the metrics as JSON")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as e:
        logger.error("Could not load config: %s", e)
        return 1

    return args.func(args, config)


if __name__ == "__main__":
    sys.exit(main())
