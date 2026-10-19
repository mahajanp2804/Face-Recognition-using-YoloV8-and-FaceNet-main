# Face Recognition Attendance System — YOLOv8 + FaceNet

A real-time face recognition system that marks attendance from a webcam. YOLOv8 detects faces, MTCNN aligns them, and FaceNet (InceptionResnetV1, VGGFace2 weights) turns each face into a 512-d embedding that is matched against enrolled people.

## How it works

```
Webcam frame
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
├── generate_face_embeddings.py   # enroll people from known_faces/ → known_embeddings.pkl
├── face_recognition.py           # live recognition + attendance logging
├── detection/
│   ├── weights/best.pt           # trained YOLOv8n face detector
│   ├── yolov8_detector.py        # standalone face-detection webcam demo
│   ├── yolov8_trainer.py         # train / fine-tune the detector
│   └── config.yaml               # dataset config for training
├── known_faces/                  # your enrollment photos (git-ignored)
└── requirements.txt
```

## Installation

Requires Python 3.9–3.12.

```bash
git clone https://github.com/mahajanp2804/Face-Recognition-using-YoloV8-and-FaceNet-main.git
cd Face-Recognition-using-YoloV8-and-FaceNet-main
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Add photos of each person** — one folder per person inside `known_faces/` (see [known_faces/README.md](known_faces/README.md)):

```
known_faces/
├── Alice Smith/  01.jpg  02.jpg
└── Bob Lee/      01.jpg
```

**2. Generate embeddings**

```bash
python generate_face_embeddings.py
```

Options: `--faces-dir <folder>` and `--output <file.pkl>`. Re-running updates the people found in the folder and keeps everyone else already in the file.

**3. Run face recognition and attendance**

```bash
python face_recognition.py
```

The webcam opens, recognises the first known face and writes it to `attendance.csv`. A person is only marked once per day. Press `q` to quit early. Settings such as `RECOGNITION_THRESHOLD`, `TIMER_SECONDS` and `CAMERA_INDEX` are at the top of the file.

**Optional — face detection demo / training**

```bash
python detection/yolov8_detector.py                 # live face boxes only
python detection/yolov8_trainer.py --epochs 100     # set the dataset path in detection/config.yaml first
```

## Privacy

Face photos, embeddings and attendance logs are personal biometric data. `known_faces/`, `*.pkl` and `attendance*.csv` are git-ignored so they stay on your machine. Only enroll people who have agreed to it.

## Credits

This project builds on [todap/Face-Recognition-using-YoloV8-and-FaceNet](https://github.com/todap/Face-Recognition-using-YoloV8-and-FaceNet), which provided the YOLOv8 face-detection weights, the detector/trainer scripts and the original detection → FaceNet recognition pipeline.

**Added in this version:**
- Attendance logging to CSV with once-per-day duplicate check
- Multiple embeddings per person, matched by nearest neighbour
- Automatic GPU/CPU selection and auto-shutoff timer
- Command-line options, portable paths and bounding-box clamping
- `requirements.txt`, `.gitignore` for personal data, and bug fixes across the scripts

Models: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MTCNN + InceptionResnetV1)
