"""Command-line interface: ``python -m facerec {enroll,run}``."""

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


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    from facerec.matcher import load_embeddings
    from facerec.recognize import run

    if not config.embeddings_path.exists():
        logger.error(
            "No embeddings found at '%s'. Run `python -m facerec enroll` first.",
            config.embeddings_path,
        )
        return 1

    known_embeddings = load_embeddings(config.embeddings_path)
    detector, embedder = _load_models(config)
    source = args.source if args.source is not None else config.camera_index

    try:
        run(
            source,
            detector,
            embedder,
            known_embeddings,
            attendance_file=config.attendance_file,
            threshold=config.recognition_threshold,
            timer_seconds=config.timer_seconds,
        )
    except OSError as e:
        logger.error("%s", e)
        return 1
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
    run.set_defaults(func=cmd_run)

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
