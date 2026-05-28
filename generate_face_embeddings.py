import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle

# Initialize YOLOv8 model for face detection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "detection", "weights", "best.pt")
model = YOLO(weights_path)
print("YOLOv8 model loaded successfully.")

# Initialize MTCNN for face detection and InceptionResnetV1 for face recognition
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
print("MTCNN and InceptionResnetV1 models loaded successfully.")

# Load known embeddings
try:
    with open('known_embeddings_claude.pkl', 'rb') as f:
        known_embeddings = pickle.load(f)
        print("Known embeddings loaded successfully.")
except FileNotFoundError:
    known_embeddings = {}
    print("No known embeddings found. Starting with an empty dictionary.")

# Function to save embeddings for images in a directory structure
def save_embeddings_from_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    for person_dir in os.listdir(directory_path):
        person_path = os.path.join(directory_path, person_dir)
        if os.path.isdir(person_path):
            name = person_dir
            person_embeddings = []
            print(f"Processing '{name}' directory...")

            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):  # Add any other supported image extensions
                    image_path = os.path.join(person_path, filename)
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error: Unable to read image '{image_path}'.")
                        continue

                    print(f"Processing '{filename}'...")
                    results = model(img)

                    # Make sure we have results and boxes
                    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                        print(f"No faces detected in '{filename}', skipping.")
                        continue

                    # Get xyxy boxes as numpy array
                    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

                    faces = []
                    for box in boxes_xyxy:
                        x1, y1, x2, y2 = box.astype(int)

                        # Clamp coordinates to image bounds
                        h, w = img.shape[:2]
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(0, min(x2, w))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(0, min(y2, h))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        face = img[y1:y2, x1:x2]
                        if face.size == 0:
                            continue
                        faces.append(face)

                    embeddings = []
                    for face in faces:
                        # Convert BGR to RGB for MTCNN
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                        # Get aligned face tensor from MTCNN
                        face_tensor = mtcnn(face_rgb)

                        if face_tensor is None:
                            continue

                        # If keep_all=True, mtcnn can return a batch
                        if face_tensor.ndim == 4:
                            face_tensor = face_tensor[0]

                        face_embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy().flatten()
                        embeddings.append(face_embedding)

                    person_embeddings.extend(embeddings)
                    print(f"Embeddings saved for '{filename}'.")

            known_embeddings[name] = person_embeddings
            print(f"Embeddings saved for '{name}'.")

    # Save the updated embeddings dictionary
    with open('known_embeddings.pkl', 'wb') as f:
        pickle.dump(known_embeddings, f)
        print("Known embeddings saved successfully.")


directory_path = "/Users/pranav/Desktop/PC/Pranav/Deep Learning Lab/Project/Face-Recognition-using-YoloV8-and-FaceNet-main/known_faces"  # Enter your directory path here
save_embeddings_from_directory(directory_path)
