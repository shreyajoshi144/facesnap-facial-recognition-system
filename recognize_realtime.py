import cv2
import pickle
import os
import csv
from datetime import datetime

def recognize_realtime(model_path="model/trained_model.yml",
                       labels_path="model/labels.pkl",
                       attendance_path="logs/attendance.csv",
                       history_log_path="logs/history.log"):

    """
    Real-time face recognition with:
    - Stable confidence threshold
    - Professional authentication display
    - Attendance marking
    - Unknown alert trigger
    - History logging
    """

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("⚠️ No trained model found. Please run train_model.py first.")
        return

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("[INFO] Recognition started. Press 'q' to quit.")

    CONF_THRESHOLD = 75   # tuned for stability
    UNKNOWN_TRIGGER_LIMIT = 15  # frames before alert

    attendance_marked = set()
    unknown_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5
        )

        for (x, y, w, h) in faces:

            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))

            id_, conf = recognizer.predict(roi_gray)

            if conf < CONF_THRESHOLD:
                person_name = labels.get(id_, "Unknown")

                display_text = f"Authenticated: {person_name}"
                color = (0, 255, 0)

                unknown_counter = 0  # reset unknown counter

                # ---- Attendance Marking ----
                if person_name not in attendance_marked:
                    now = datetime.now()
                    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

                    file_exists = os.path.isfile(attendance_path)

                    with open(attendance_path, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["Name", "Timestamp"])
                        writer.writerow([person_name, time_str])

                    # ---- History Logging ----
                    with open(history_log_path, "a") as log:
                        log.write(f"[{time_str}] AUTHENTICATED: {person_name} | Confidence: {conf:.2f}\n")

                    attendance_marked.add(person_name)

            else:
                display_text = "Unknown"
                color = (0, 0, 255)

                unknown_counter += 1

                # ---- Unknown Alert Trigger ----
                if unknown_counter > UNKNOWN_TRIGGER_LIMIT:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(history_log_path, "a") as log:
                        log.write(f"[{now}] ALERT: Unknown face detected repeatedly.\n")

                    unknown_counter = 0  # reset after logging

            cv2.putText(frame, display_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Recognition System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recognition stopped.")


if __name__ == "__main__":
    recognize_realtime()