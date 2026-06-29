 # FaceSnap – Real-Time Face Recognition System

FaceSnap is a real-time face recognition application built with Python and OpenCV that captures face images, trains a recognition model, and identifies individuals through a live webcam feed. The system also maintains recognition logs, making it suitable for learning computer vision concepts and serving as a foundation for security, attendance, or access control applications.

---

## Features

* Capture face images for new users through a webcam.
* Automatically organize captured images for training.
* Train a face recognition model using the collected dataset.
* Perform real-time face detection and recognition.
* Detect and flag unknown individuals.
* Store recognition history for future review.
* Simple menu-driven interface for managing the complete workflow.

---

![img.png](img.png)

## Project Structure

```text
FaceSnap/
│── FaceSnap.py              # Main application launcher
│── capture_faces.py         # Capture face images
│── train_model.py           # Train the face recognition model
│── recognize_realtime.py    # Real-time face recognition
│── search_history.py        # View recognition history
│── faces/                   # Dataset of captured face images
│── model/                   # Trained recognition model
│── history/                 # Recognition logs
│── requirements.txt         # Project dependencies
│── README.md                # Project documentation
```

---

## System Workflow

### 1. Face Registration

Users register by capturing multiple face images through a webcam. The images are stored in the dataset directory for training.

### 2. Model Training

The collected face images are processed to train a face recognition model capable of identifying registered users.

### 3. Real-Time Recognition

The webcam continuously detects faces and compares them with the trained model.

* Recognized users are identified by name.
* Unknown faces are flagged automatically.

### 4. Recognition Logging

Each detection event is stored in the history directory, allowing previous recognitions to be reviewed later.

---

## Technologies Used

* Python
* OpenCV
* NumPy
* OS Module
* Webcam

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/FaceSnap.git
cd FaceSnap
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python FaceSnap.py
```

---

## Usage

Launch the application and choose from the menu:

1. Capture face images
2. Train the recognition model
3. Start real-time recognition
4. View recognition history
5. Exit

---

## Future Enhancements

* Improve recognition accuracy with deep learning models.
* Add face mask detection.
* Integrate emotion recognition.
* Export recognition logs to CSV or a database.
* Build a desktop or web-based user interface.
* Add user management features.
* Support multiple cameras.
---

