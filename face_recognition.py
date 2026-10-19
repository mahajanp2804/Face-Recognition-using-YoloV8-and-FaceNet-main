import os
import sys
import csv
import time
import pickle
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# ------------------- CONFIG -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "known_embeddings.pkl")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "detection", "weights", "best.pt")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
CAMERA_INDEX = 0
RECOGNITION_THRESHOLD = 0.8
TIMER_SECONDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------


# ------------------------------------------------
# FUNCTION: COMPARE EMBEDDINGS
# ------------------------------------------------
def compare_embeddings(embedding, db, threshold=RECOGNITION_THRESHOLD):
    """
    Returns (name, distance) of the closest known embedding,
    or ("Unknown", distance) if it is not within the threshold.
    """
    best_dist = float('inf')
    best_name = "Unknown"

    for name, emb_list in db.items():
        for known in emb_list:
            dist = np.linalg.norm(embedding - known)
            if dist < best_dist:
                best_dist, best_name = dist, name

    if best_dist >= threshold:
        best_name = "Unknown"
    return best_name, best_dist


# ------------------------------------------------
# FUNCTION: CHECK IF ATTENDANCE ALREADY MARKED TODAY
# ------------------------------------------------
def attendance_exists(name):
    """
    Returns True if attendance for this person is already marked today.
    """
    if not os.path.exists(ATTENDANCE_FILE):
        return False

    today = datetime.now().strftime("%Y-%m-%d")

    with open(ATTENDANCE_FILE, "r", newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) < 2:
                continue

            stored_name, timestamp = row[0], row[1]
            if stored_name == name and timestamp.startswith(today):
                return True

    return False


# ------------------------------------------------
# FUNCTION: MARK ATTENDANCE
# ------------------------------------------------
def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.exists(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Timestamp"])
        writer.writerow([name, now])

    print(f"Attendance Marked for {name} at {now}")


# ------------------------------------------------
# FUNCTION: DETECT + EMBED THE MOST CONFIDENT FACE
# ------------------------------------------------
def detect_and_embed(frame, model, mtcnn, resnet):
    """
    Returns (box, embedding) for the most confident face in the frame.
    box is None if no face was detected; embedding is None if alignment failed.
    """
    r = model(frame, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None

    # YOLO sorts boxes by confidence, so index 0 is the most confident face
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[0].astype(int)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    box = (x1, y1, x2, y2)

    face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(Image.fromarray(face_rgb))
    if face_tensor is None:
        return box, None

    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy().flatten()
    return box, embedding


def main():
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"No embeddings found at '{EMBEDDINGS_PATH}'. Run generate_face_embeddings.py first.")
        sys.exit(1)

    with open(EMBEDDINGS_PATH, 'rb') as f:
        known_embeddings = pickle.load(f)

    model = YOLO(YOLO_WEIGHTS)
    mtcnn = MTCNN(keep_all=False, device=DEVICE)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    # --------------------- START CAMERA ---------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    start_time = time.time()
    recognized_name = None

    print("Camera started. Searching for face...")

    while time.time() - start_time <= TIMER_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        box, embedding = detect_and_embed(frame, model, mtcnn, resnet)

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if embedding is not None:
            match, dist = compare_embeddings(embedding, known_embeddings)

            # ------------------- IF MATCH FOUND -------------------
            if match != "Unknown":
                recognized_name = match

                # Check duplicate attendance
                if attendance_exists(recognized_name):
                    print(f"Attendance already marked for {recognized_name} today.")
                    msg = f"Already Marked: {recognized_name}"
                else:
                    mark_attendance(recognized_name)
                    msg = f"Attendance Marked: {recognized_name}"

                # Show message on screen for 2 sec
                cv2.putText(frame, msg, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera Feed", frame)
                cv2.waitKey(2000)
                break

            cv2.putText(frame, "Face Not Recognised", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if recognized_name is None:
        print("Face not recognised — Camera off.")


if __name__ == "__main__":
    main()
