"""
services/liveness_service.py
==============================
Blink-based liveness detection using MediaPipe Face Mesh + Eye Aspect Ratio.

Python 3.13 note: MediaPipe imports fine but FaceMesh() may crash at runtime.
The constructor is wrapped in try/except — if it fails, all faces are marked
LIVE (non-blocking fallback) and a fix message is printed to the terminal.
"""

import time
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 2
MIN_BLINKS    = 2
WINDOW_SECS   = 5.0


class LivenessService:
    """Stateful EAR-based blink detector. Call check_liveness(frame) per frame."""

    def __init__(self):
        self._blink_times:   deque = deque()
        self._frame_counter: int   = 0
        self._ear_history:   deque = deque(maxlen=5)
        self._face_mesh = None

        if MEDIAPIPE_AVAILABLE:
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                print(
                    f"[LivenessService] ⚠ MediaPipe FaceMesh failed: {e}\n"
                    "  → Liveness skipped; all faces marked LIVE.\n"
                    "  → Fix: switch to Python 3.11/3.12 and reinstall mediapipe."
                )

    def check_liveness(self, frame: np.ndarray) -> dict:
        if self._face_mesh is None:
            return self._fallback()

        import cv2
        results = self._face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return self._build(0.0)

        lm  = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        ear = (self._ear([pt(i) for i in LEFT_EYE_IDX]) +
               self._ear([pt(i) for i in RIGHT_EYE_IDX])) / 2.0
        self._ear_history.append(ear)

        if ear < EAR_THRESHOLD:
            self._frame_counter += 1
        else:
            if self._frame_counter >= CONSEC_FRAMES:
                self._blink_times.append(time.time())
            self._frame_counter = 0

        return self._build(ear)

    def _build(self, ear: float) -> dict:
        now = time.time()
        while self._blink_times and self._blink_times[0] < now - WINDOW_SECS:
            self._blink_times.popleft()
        n = len(self._blink_times)
        return {
            "is_live":     n >= MIN_BLINKS,
            "blink_count": n,
            "ear":         round(ear, 3),
            "reason":      f"{n} blink(s) — LIVE" if n >= MIN_BLINKS
                           else f"{n}/{MIN_BLINKS} blink(s) — checking…",
        }

    @staticmethod
    def _ear(pts: list) -> float:
        p1, p2, p3, p4, p5, p6 = pts
        return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (
            2.0 * np.linalg.norm(p1 - p4) + 1e-6)

    def _fallback(self) -> dict:
        return {"is_live": True, "blink_count": 0, "ear": 0.0,
                "reason": "Liveness skipped (MediaPipe unavailable)"}

    def reset(self):
        self._blink_times.clear()
        self._frame_counter = 0
        self._ear_history.clear()
