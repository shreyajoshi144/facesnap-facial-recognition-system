import cv2
import os

MAX_IMAGES = 30  # 🔥 Hard limit


def capture_faces(person_name, dataset_path="dataset"):
    """
    Captures face images of a new person using webcam and stores them.
    Auto-stops after MAX_IMAGES images.
    """

    person_dir = os.path.join(dataset_path, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    print(f"[INFO] Capturing faces for {person_name}.")
    print(f"[INFO] Maximum images allowed: {MAX_IMAGES}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5
        )

        for (x, y, w, h) in faces:
            if count >= MAX_IMAGES:
                break

            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            count += 1

            file_path = os.path.join(
                person_dir, f"{person_name}_{count}.jpg"
            )

            cv2.imwrite(file_path, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            print(f"[INFO] Captured {count}/{MAX_IMAGES}")

        cv2.imshow("Face Capture", frame)

        # Auto-stop when max reached
        if count >= MAX_IMAGES:
            print("[INFO] Reached maximum image limit.")
            break

        # Manual exit still allowed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Capture stopped manually.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ Saved {count} face images for {person_name}.")


if __name__ == "__main__":
    name = input("Enter the name of the person: ")
    capture_faces(name)