import os
import argparse
import pickle

import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# ------------------- CONFIG -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(BASE_DIR, "detection", "weights", "best.pt")
DEFAULT_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
DEFAULT_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "known_embeddings.pkl")
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------


def load_models():
    model = YOLO(YOLO_WEIGHTS)
    mtcnn = MTCNN(keep_all=False, device=DEVICE)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    print("YOLOv8, MTCNN and InceptionResnetV1 models loaded successfully.")
    return model, mtcnn, resnet


def load_embeddings(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            print(f"Loaded existing embeddings from '{path}'.")
            return pickle.load(f)
    print("No existing embeddings found. Starting with an empty dictionary.")
    return {}


def crop_faces(img, model):
    """Detect faces with YOLO and return the cropped BGR face regions."""
    results = model(img, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    h, w = img.shape[:2]
    faces = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)

        # Clamp coordinates to image bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        faces.append(img[y1:y2, x1:x2])
    return faces


def embed_face(face_bgr, mtcnn, resnet):
    """Align a face crop with MTCNN and return its 512-d FaceNet embedding, or None."""
    face_rgb = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    face_tensor = mtcnn(face_rgb)
    if face_tensor is None:
        return None

    with torch.no_grad():
        return resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy().flatten()


def save_embeddings_from_directory(directory_path, embeddings_path):
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    model, mtcnn, resnet = load_models()
    known_embeddings = load_embeddings(embeddings_path)

    for name in sorted(os.listdir(directory_path)):
        person_path = os.path.join(directory_path, name)
        if not os.path.isdir(person_path):
            continue

        print(f"Processing '{name}' directory...")
        person_embeddings = []

        for filename in sorted(os.listdir(person_path)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(person_path, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"  Error: Unable to read image '{filename}', skipping.")
                continue

            faces = crop_faces(img, model)
            if not faces:
                print(f"  No faces detected in '{filename}', skipping.")
                continue

            embeddings = [e for e in (embed_face(f, mtcnn, resnet) for f in faces) if e is not None]
            person_embeddings.extend(embeddings)
            print(f"  {len(embeddings)} embedding(s) from '{filename}'.")

        if person_embeddings:
            known_embeddings[name] = person_embeddings
            print(f"Saved {len(person_embeddings)} embedding(s) for '{name}'.")
        else:
            print(f"Warning: no usable faces found for '{name}', not saved.")

    with open(embeddings_path, 'wb') as f:
        pickle.dump(known_embeddings, f)
    print(f"Known embeddings saved to '{embeddings_path}' ({len(known_embeddings)} people).")


def main():
    parser = argparse.ArgumentParser(description="Generate FaceNet embeddings for known faces.")
    parser.add_argument("--faces-dir", default=DEFAULT_FACES_DIR,
                        help="Folder with one sub-folder of photos per person (default: known_faces/)")
    parser.add_argument("--output", default=DEFAULT_EMBEDDINGS_PATH,
                        help="Where to save the embeddings (default: known_embeddings.pkl)")
    args = parser.parse_args()

    save_embeddings_from_directory(args.faces_dir, args.output)


if __name__ == "__main__":
    main()
