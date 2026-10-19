# Face Recognition Attendance System — YOLOv8 + FaceNet

[![CI](https://github.com/mahajanp2804/Face-Recognition-using-YoloV8-and-FaceNet-main/actions/workflows/ci.yml/badge.svg)](https://github.com/mahajanp2804/Face-Recognition-using-YoloV8-and-FaceNet-main/actions/workflows/ci.yml)

A real-time face recognition system that marks attendance from a webcam. YOLOv8 detects faces, MTCNN aligns them, and FaceNet (InceptionResnetV1, VGGFace2 weights) turns each face into a 512-d embedding that is matched against enrolled people.

## How it works

```
Webcam / video frame
   │
   ▼
YOLOv8n face detector ──► crop most confident face
   │
   ▼
MTCNN alignment (160×160)
   │
   ▼
FaceNet embedding (512-d)
   │
   ▼
Nearest neighbour vs. enrolled embeddings (L2 distance < 0.8)
   │
   ├── match   ──► log Name + Timestamp to attendance.csv (once per day)
   └── no match ──► "Face Not Recognised" (camera stops after 5 s)
```

## Project structure

```
├── src/facerec/
│   ├── cli.py            # `python -m facerec enroll | run`
│   ├── config.py         # loads config.yaml
│   ├── detector.py       # YOLOv8 face detection, crop + box clamping
│   ├── embedder.py       # MTCNN alignment + FaceNet 512-d embeddings
│   ├── matcher.py        # nearest-neighbour matching, embeddings .pkl load/save
│   ├── attendance.py     # once-per-day CSV attendance log
│   ├── enroll.py         # known_faces/ → known_embeddings.pkl
│   └── recognize.py      # live recognition loop
├── tests/                # pytest suite (no webcam or model weights needed)
├── detection/
│   ├── weights/best.pt   # trained YOLOv8n face detector
│   ├── yolov8_detector.py  # standalone face-detection webcam demo
│   ├── yolov8_trainer.py   # train / fine-tune the detector
│   └── config.yaml       # dataset config for training
├── known_faces/          # your enrollment photos (git-ignored)
├── config.yaml           # paths, threshold, timer, camera index
└── pyproject.toml        # package metadata, dependencies, ruff + pytest config
```

## Installation

Requires Python 3.9–3.12.

```bash
git clone https://github.com/mahajanp2804/Face-Recognition-using-YoloV8-and-FaceNet-main.git
cd Face-Recognition-using-YoloV8-and-FaceNet-main
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .                # or: pip install -r requirements.txt
```

This installs the `facerec` package and its dependencies. Run the commands below from the repository folder so `config.yaml` is picked up.

## Usage

**1. Add photos of each person** — one folder per person inside `known_faces/` (see [known_faces/README.md](known_faces/README.md)):

```
known_faces/
├── Alice Smith/  01.jpg  02.jpg
└── Bob Lee/      01.jpg
```

**2. Enroll — generate embeddings**

```bash
python -m facerec enroll --dir known_faces
```

This writes `known_embeddings.pkl` (a dict of name → list of 512-d embeddings). Use `--output <file.pkl>` to write somewhere else. Re-running updates the people found in the folder and keeps everyone else already in the file.

**3. Run face recognition and attendance**

```bash
python -m facerec run --source 0
```

`--source` is a webcam index (`0`, `1`, …) or a path to a video file; it defaults to `camera_index` from `config.yaml`. The window shows the feed, recognises the first known face and writes it to `attendance.csv`. A person is only marked once per day. Press `q` to quit early.

Both commands accept `--config <file>` to use a different settings file and `-v` for debug logging (including match distances). `facerec` also works as a shorter command name after installation, e.g. `facerec run --source 0`.

### Configuration

[`config.yaml`](config.yaml) holds the settings; relative paths are resolved against the file's folder.

| Key | Default | Meaning |
| --- | --- | --- |
| `paths.faces_dir` | `known_faces` | Enrollment photos, one sub-folder per person |
| `paths.embeddings` | `known_embeddings.pkl` | Embeddings written by `enroll`, read by `run` |
| `paths.yolo_weights` | `detection/weights/best.pt` | YOLOv8 face detector |
| `paths.attendance` | `attendance.csv` | Attendance log |
| `recognition_threshold` | `0.8` | Max L2 distance to count as a match |
| `timer_seconds` | `5` | Stop searching after this many seconds |
| `camera_index` | `0` | Default webcam for `run` |

**Optional — face detection demo / training**

```bash
python detection/yolov8_detector.py                 # live face boxes only
python detection/yolov8_trainer.py --epochs 100     # set the dataset path in detection/config.yaml first
```

## Development

```bash
pip install -e ".[dev]"
pytest          # unit tests: matcher, attendance, config, box clamping, enrollment
ruff check .    # lint
ruff format .   # format
```

The tests use fake detector/embedder objects, so they run without a webcam, GPU or model weights. CI runs ruff and pytest on Python 3.11 for every push and pull request.

## Privacy

Face photos, embeddings and attendance logs are personal biometric data. `known_faces/`, `*.pkl` and `attendance*.csv` are git-ignored so they stay on your machine. Only enroll people who have agreed to it.

## Credits

This project builds on [todap/Face-Recognition-using-YoloV8-and-FaceNet](https://github.com/todap/Face-Recognition-using-YoloV8-and-FaceNet), which provided the YOLOv8 face-detection weights, the detector/trainer scripts and the original detection → FaceNet recognition pipeline.

**Added in this version:**
- Attendance logging to CSV with once-per-day duplicate check
- Multiple embeddings per person, matched by nearest neighbour
- Automatic GPU/CPU selection and auto-shutoff timer
- Command-line options, portable paths and bounding-box clamping
- Installable `facerec` package with an `enroll` / `run` CLI, YAML config, logging and type hints
- pytest suite and GitHub Actions CI (ruff + pytest)
- `pyproject.toml`, `.gitignore` for personal data, and bug fixes across the scripts

Models: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MTCNN + InceptionResnetV1)
