"""
services/analytics_service.py
================================
Handles all detection logging, history search, and export.
"""

import os
import csv
import pandas as pd
from datetime import datetime
from threading import Lock

LOGS_FILE   = "logs.csv"
HISTORY_DIR = "history"

LOG_FIELDS = ["name", "timestamp", "age", "gender", "liveness", "confidence"]


class AnalyticsService:
    """
    Writes detection events to logs.csv and provides search + export.
    Thread-safe via a lock (important for streamlit-webrtc callbacks).
    """

    def __init__(self):
        self._lock = Lock()
        self._history_available = self._prepare_history_dir()
        self._ensure_csv_header()

    # ── Directory setup ───────────────────────────────────────────────────────

    def _prepare_history_dir(self) -> bool:
        """
        Safely handles the history/ path.
        Your repo has 'history' as a plain file (binary/text blob), not a folder.
        We detect this and gracefully skip legacy log loading instead of crashing.
        Returns True if history/ is a usable directory, False otherwise.
        """
        if os.path.isdir(HISTORY_DIR):
            return True                          # already a proper directory

        if os.path.exists(HISTORY_DIR):
            # 'history' exists but is a FILE (your original repo artifact)
            # We cannot and should not delete or rename your file.
            # Legacy history loading will simply be skipped.
            print(
                f"[AnalyticsService] ⚠ '{HISTORY_DIR}' exists as a file, not a directory. "
                "Legacy history loading is disabled. All new detections log to logs.csv."
            )
            return False

        # Neither file nor directory — safe to create
        os.makedirs(HISTORY_DIR, exist_ok=True)
        return True

    # ── Logging ──────────────────────────────────────────────────────────────

    def log_detection(self, name: str, age, gender: str, liveness: str, confidence: float):
        """Appends one detection record to logs.csv. Thread-safe."""
        row = {
            "name":       name,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "age":        str(age),
            "gender":     gender,
            "liveness":   liveness,
            "confidence": str(confidence),
        }
        with self._lock:
            with open(LOGS_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                writer.writerow(row)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, name_query: str = None) -> pd.DataFrame:
        """Returns detection history as a DataFrame, optionally filtered by name."""
        df = self._load_logs_csv()

        legacy_df = self._load_legacy_history()
        if not legacy_df.empty:
            df = pd.concat([df, legacy_df], ignore_index=True)

        if name_query:
            df = df[df["name"].str.contains(name_query, case=False, na=False)]

        if not df.empty:
            df = df.sort_values("timestamp", ascending=False)

        return df

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        df = self._load_logs_csv()
        return {
            "total":        len(df),
            "unique_names": df["name"].nunique() if not df.empty else 0,
            "unknowns":     int((df["name"] == "Unknown").sum()) if not df.empty else 0,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _ensure_csv_header(self):
        if not os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                writer.writeheader()

    def _load_logs_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(LOGS_FILE)
            for col in LOG_FIELDS:
                if col not in df.columns:
                    df[col] = "–"
            return df[LOG_FIELDS]
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=LOG_FIELDS)

    def _load_legacy_history(self) -> pd.DataFrame:
        """
        INTEGRATION: reads logs from history/ folder written by search_history.py.
        Safely skipped if history/ is not a usable directory.
        """
        if not self._history_available or not os.path.isdir(HISTORY_DIR):
            return pd.DataFrame(columns=LOG_FIELDS)

        records = []
        for fname in os.listdir(HISTORY_DIR):
            fpath = os.path.join(HISTORY_DIR, fname)
            try:
                if fname.endswith(".csv"):
                    records.append(pd.read_csv(fpath))
                elif fname.endswith(".txt"):
                    with open(fpath) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                records.append(pd.DataFrame([{
                                    "name": line, "timestamp": "",
                                    "age": "–", "gender": "–",
                                    "liveness": "–", "confidence": "–",
                                }]))
            except Exception:
                continue

        if not records:
            return pd.DataFrame(columns=LOG_FIELDS)

        combined = pd.concat(records, ignore_index=True)
        for col in LOG_FIELDS:
            if col not in combined.columns:
                combined[col] = "–"
        return combined[LOG_FIELDS]
