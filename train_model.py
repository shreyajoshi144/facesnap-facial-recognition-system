import cv2
import os
import numpy as np
import pickle

DATASET_PATH = "dataset"
MODEL_PATH = "model/trained_model.yml"
LABELS_PATH = "model/labels.pkl"


def train_model():
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset folder not found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    print("[INFO] Scanning dataset...")

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        images = os.listdir(person_path)

        if len(images) == 0:
            print(f"⚠️ No images found for {person_name}. Skipping.")
            continue

        label_map[current_label] = person_name

        for image_name in images:
            image_path = os.path.join(person_path, image_name)

            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue

                faces.append(image)
                labels.append(current_label)

            except Exception as e:
                print(f"⚠️ Skipping image {image_name}: {e}")

        current_label += 1

    if len(faces) == 0:
        print("❌ No training data found.")
        return

    print("[INFO] Training model...")
    recognizer.train(faces, np.array(labels))

    os.makedirs("model", exist_ok=True)

    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print("✅ Model trained successfully!")
    print(f"📦 Total faces used: {len(faces)}")


if __name__ == "__main__":
    train_model()