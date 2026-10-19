import os

import cv2
from ultralytics import YOLO

# Standalone face-detection demo: draws YOLOv8 face boxes on the webcam feed.
# Press 'q' to quit.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE_DIR, "weights", "best.pt")

# start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

# model
model = YOLO(WEIGHTS)

# object classes
classNames = model.names

while True:
    success, img = cap.read()
    if not success:
        print("Camera error.")
        break

    results = model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # class name + confidence
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{classNames[cls]} ({confidence * 100:.1f}%)"
            cv2.putText(
                img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
