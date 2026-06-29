"""
services/recognition_service.py
=================================
Wraps your existing trained model (model/) with:
  - OpenCV LBPH recognition  (INTEGRATION: your original pipeline)
  - DeepFace ArcFace embeddings for improved accuracy (optional layer)
  - DeepFace age/gender analysis

INTEGRATION NOTES:
  - Loads the model trained by your train_model.py from model/trained_model.yml
  - Loads label map from model/labels.pkl  (created by your train_model.py)
  - Does NOT alter your training pipeline in any way
  - ArcFace embeddings are stored separately in embeddings.pkl
"""

import os
import cv2
import pickle
import numpy as np
from typing import Optional

# ── DeepFace is an optional enhancement; graceful fallback if not installed ──
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
class RecognitionService:
    """
    Thin service layer over your existing OpenCV LBPH recognizer.
    Adds DeepFace-based attribute analysis and ArcFace embedding support.
    """

    MODEL_PATH      = os.path.join("model", "trained_model.yml")
    LABELS_PATH     = os.path.join("model", "labels.pkl")
    EMBEDDINGS_PATH = "embeddings.pkl"

    def __init__(self):
        self._recognizer   = self._load_lbph_model()
        self._label_map    = self._load_labels()
        self._embeddings   = self._load_embeddings()
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ── Model loaders ────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Safe file checks — surfaces clear messages in Streamlit instead of crashes
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def model_ready(self) -> bool:
        return self._model_ready

    def _load_lbph_model(self):
        """
        INTEGRATION: loads the model saved by your train_model.py.
        Expected path: model/trained_model.yml

        Safe: if the file doesn't exist, the recognizer is created but not
        loaded. Callers should check self.model_ready before predicting.
        """
        self._model_ready = False
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        if not os.path.exists(self.MODEL_PATH):
            print(
                f"[RecognitionService] ⚠ Model not found: {self.MODEL_PATH}\n"
                "  → Run: python train_model.py"
            )
            return recognizer

        try:
            recognizer.read(self.MODEL_PATH)
            self._model_ready = True
            print(f"[RecognitionService] ✓ Model loaded from {self.MODEL_PATH}")
        except Exception as e:
            print(f"[RecognitionService] ✗ Failed to load model: {e}")

        return recognizer

    def _load_labels(self) -> dict:
        """
        INTEGRATION: loads the label → name mapping saved by your train_model.py.
        Expected path: model/labels.pkl  |  Falls back to empty dict if not found.
        """
        if not os.path.exists(self.LABELS_PATH):
            print(f"[RecognitionService] ⚠ Labels not found: {self.LABELS_PATH}")
            return {}
        try:
            with open(self.LABELS_PATH, "rb") as f:
                labels = pickle.load(f)
            print(f"[RecognitionService] ✓ Labels loaded — {len(labels)} known person(s)")
            return labels
        except Exception as e:
            print(f"[RecognitionService] ✗ Failed to load labels: {e}")
            return {}

    def _load_embeddings(self) -> dict:
        """
        Loads ArcFace embeddings — separate from your original pipeline.
        Path: embeddings.pkl  (created by build_embeddings() — optional)
        Returns empty dict silently if not present; this is expected on first run.
        """
        if not os.path.exists(self.EMBEDDINGS_PATH):
            return {}
        try:
            with open(self.EMBEDDINGS_PATH, "rb") as f:
                emb = pickle.load(f)
            print(f"[RecognitionService] ✓ ArcFace embeddings loaded — {len(emb)} person(s)")
            return emb
        except Exception as e:
            print(f"[RecognitionService] ⚠ Could not load embeddings: {e}")
            return {}

    # ── Core detection / recognition ─────────────────────────────────────────

    def detect_and_recognize(self, frame: np.ndarray, conf_threshold: float = 0.6) -> list[dict]:
        """
        Detects faces in `frame` and returns a list of detection dicts:
          {bbox: (x,y,w,h), name: str, confidence: float}

        INTEGRATION: uses your LBPH model for recognition. If ArcFace
        embeddings exist, cosine similarity is used as a fallback/override.
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        results = []
        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]

            name, confidence = self._recognize_face(roi, frame[y : y + h, x : x + w])

            if confidence < conf_threshold:
                name = "Unknown"

            results.append({"bbox": (x, y, w, h), "name": name, "confidence": confidence})

        return results

    def _recognize_face(self, gray_roi: np.ndarray, color_roi: np.ndarray) -> tuple[str, float]:
        """
        Tries ArcFace embeddings first; falls back to LBPH.
        Returns (name, confidence_0_to_1).
        Safely returns ("Unknown", 0.0) if no model is loaded yet.
        """
        # ── Option A: ArcFace cosine similarity (better accuracy) ────────────
        if self._embeddings and DEEPFACE_AVAILABLE:
            name, score = self._cosine_match(color_roi)
            if name:
                return name, score

        # ── Option B: LBPH (your original logic) ─────────────────────────────
        if not self._model_ready:
            # Model not loaded — don't crash, return Unknown gracefully
            return "Unknown", 0.0

        try:
            label, lbph_dist = self._recognizer.predict(gray_roi)
            # Convert LBPH distance to a 0–1 confidence (lower dist = better)
            confidence = max(0.0, 1.0 - lbph_dist / 100.0)
            name = self._label_map.get(label, "Unknown")
            return name, confidence
        except Exception:
            return "Unknown", 0.0

    def _cosine_match(self, color_roi: np.ndarray, threshold: float = 0.68) -> tuple[Optional[str], float]:
        """Uses DeepFace ArcFace embeddings + cosine similarity to match a face."""
        try:
            embedding = DeepFace.represent(
                color_roi, model_name="ArcFace", enforce_detection=False
            )[0]["embedding"]
            embedding = np.array(embedding)

            best_name, best_score = None, 0.0
            for name, stored_emb in self._embeddings.items():
                score = self._cosine_similarity(embedding, np.array(stored_emb))
                if score > best_score:
                    best_score = score
                    best_name  = name

            if best_score >= threshold:
                return best_name, best_score
        except Exception:
            pass
        return None, 0.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    # ── Attribute analysis ───────────────────────────────────────────────────

    def get_attributes(self, face_roi: np.ndarray) -> Optional[dict]:
        """
        Runs DeepFace age & gender analysis on a face crop.
        Returns {'age': int, 'gender': str} or None on failure.
        Throttle calls from the caller to avoid slowdowns.
        """
        if not DEEPFACE_AVAILABLE or face_roi.size == 0:
            return None
        try:
            result = DeepFace.analyze(
                face_roi,
                actions=["age", "gender"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            return {
                "age":    result.get("age", "–"),
                "gender": result.get("dominant_gender", "–"),
            }
        except Exception:
            return None

    # ── Embedding management ─────────────────────────────────────────────────

    def build_embeddings(self, faces_dir: str = "faces"):
        """
        Builds ArcFace embeddings for all persons in faces/ directory.
        Stores in embeddings.pkl — separate from your model/ files.

        Call this after running train_model.py to add the ArcFace layer.
        """
        if not DEEPFACE_AVAILABLE:
            print("[RecognitionService] DeepFace not available; skipping embedding build.")
            return

        embeddings = {}
        for person in os.listdir(faces_dir):
            person_dir = os.path.join(faces_dir, person)
            if not os.path.isdir(person_dir):
                continue
            vecs = []
            for img_file in os.listdir(person_dir)[:20]:   # cap at 20 per person
                img_path = os.path.join(person_dir, img_file)
                try:
                    rep = DeepFace.represent(
                        img_path, model_name="ArcFace", enforce_detection=False
                    )
                    vecs.append(rep[0]["embedding"])
                except Exception:
                    continue
            if vecs:
                embeddings[person] = np.mean(vecs, axis=0).tolist()
                print(f"[RecognitionService] Embedded: {person} ({len(vecs)} images)")

        with open(self.EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)
        self._embeddings = embeddings
        print(f"[RecognitionService] Saved embeddings → {self.EMBEDDINGS_PATH}")
