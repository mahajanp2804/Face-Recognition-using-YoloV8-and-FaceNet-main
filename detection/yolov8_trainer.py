import argparse
import os

from ultralytics import YOLO

# Fine-tunes YOLOv8n for single-class face detection.
# Set `path` in config.yaml to your dataset root (YOLO format: train/images, valid/images, ...).

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Train the YOLOv8 face detector.")
parser.add_argument(
    "--data", default=os.path.join(BASE_DIR, "config.yaml"), help="Dataset config YAML"
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument(
    "--resume", metavar="LAST_PT", help="Resume an interrupted run from its last.pt checkpoint"
)
args = parser.parse_args()

if args.resume:
    model = YOLO(args.resume)
    model.train(resume=True)
else:
    # Start from COCO-pretrained weights rather than training from scratch
    model = YOLO("yolov8n.pt")
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
