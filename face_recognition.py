import os
import cv2
import pickle
import numpy as np
import time
import csv
from datetime import datetime
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# ------------------- CONFIG -------------------
EMBEDDINGS_PATH = "known_embeddings.pkl"
YOLO_WEIGHTS = os.path.join("detection", "weights", "best.pt")
ATTENDANCE_FILE = "attendance.csv"
RECOGNITION_THRESHOLD = 0.8
TIMER_SECONDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------

device = torch.device(DEVICE)

# Load YOLO
model = YOLO(YOLO_WEIGHTS)

# Load MTCNN + ResNet
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load embeddings
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, 'rb') as f:
        known_embeddings = pickle.load(f)
else:
    print("No embeddings found.")
    exit()

# ------------------------------------------------
# FUNCTION: COMPARE EMBEDDINGS
# ------------------------------------------------
def compare_embeddings(embedding, db, threshold=RECOGNITION_THRESHOLD):
    best_dist = float('inf')
    best_match = "Unknown"

    for name, emb_list in db.items():
        for known in emb_list:
            dist = np.linalg.norm(embedding - known)
            if dist < best_dist:
                best_dist = dist
                best_match = name if dist < threshold else "Unknown"

    return best_match, best_dist


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

    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if exists

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


# --------------------- START CAMERA ---------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

start_time = time.time()
face_recognized = False
recognized_name = None

print("Camera started. Searching for face...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    elapsed = time.time() - start_time

    if elapsed > TIMER_SECONDS and not face_recognized:
        print("Face not recognised — Camera shutting down")
        break

    # YOLO detect
    results = model(frame)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Use first detected face
    x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[0].astype(int)
    face_bgr = frame[y1:y2, x1:x2]

    if face_bgr.size == 0:
        continue

    # Convert for MTCNN
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(face_rgb)

    face_tensor = mtcnn(pil_face)
    if face_tensor is None:
        continue

    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)

    face_tensor = face_tensor.to(device)

    with torch.no_grad():
        embedding = resnet(face_tensor).cpu().numpy().flatten()

    match, dist = compare_embeddings(embedding, known_embeddings)

    # ------------------- IF MATCH FOUND -------------------
    if match != "Unknown":
        recognized_name = match
        face_recognized = True

        # Check duplicate attendance
        if attendance_exists(recognized_name):
            print(f"Attendance already marked for {recognized_name} today.")
            msg = f"Already Marked: {recognized_name}"
        else:
            mark_attendance(recognized_name)
            msg = f"Attendance Marked: {recognized_name}"

        # Show message on screen
        cv2.putText(frame, msg, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(2000)   # show message for 2 sec
        break

    else:
        cv2.putText(frame, "Face Not Recognised", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

if not face_recognized:
    print("Face not recognised — Camera off.")
