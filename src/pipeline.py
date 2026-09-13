"""
Shared per-frame pipeline used by both the Streamlit app and the local
webcam/video script, so the two entry points can't drift out of sync.

Flow per frame:
  1. detect_faces()                    -- cheap, every frame
  2. tracker.update()                  -- assign/maintain stable IDs
  3. for each track: run heavy age/gender ensemble inference only every
     INFER_EVERY_N_FRAMES frames (or on a brand-new track) -- this is
     the real-time performance optimization described in detector.py
  4. smoother.update() / get_smoothed() -- reduce flicker
  5. fairness_audit.flag_prediction()   -- honest uncertainty flag,
     informed by both frame-to-frame agreement AND cross-model agreement
  6. session_log.record()               -- for the dashboard / CSV export
  7. draw overlay box + label on the frame
"""

from __future__ import annotations
import cv2

from .detector import detect_faces, infer_age_gender_ensemble
from .tracker import CentroidTracker
from .smoothing import TrackSmoother
from .analytics import SessionLog
from .fairness_audit import flag_prediction

INFER_EVERY_N_FRAMES = 8  # re-run heavy model this often per track


class RealtimePipeline:
    def __init__(self):
        self.tracker = CentroidTracker()
        self.smoother = TrackSmoother()
        self.session_log = SessionLog()
        self._frame_counter_per_track = {}
        self._last_models_agreed = {}  # track_id -> bool | None, for UI/debugging

    def process_frame(self, frame):
        boxes = detect_faces(frame)
        tracks = self.tracker.update(boxes)

        for track_id, (x, y, w, h) in tracks.items():
            if w <= 0 or h <= 0:
                continue

            self._frame_counter_per_track.setdefault(track_id, 0)
            due_for_inference = (
                self._frame_counter_per_track[track_id] % INFER_EVERY_N_FRAMES == 0
            )
            self._frame_counter_per_track[track_id] += 1

            if due_for_inference:
                pad = int(0.15 * max(w, h))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                face_crop = frame[y0:y1, x0:x1]
                if face_crop.size > 0:
                    result = infer_age_gender_ensemble(face_crop)
                    if result:
                        self.smoother.update(
                            track_id, result["age"], result["gender"], result["gender_confidence"]
                        )
                        self._last_models_agreed[track_id] = result["models_agreed"]

            smoothed = self.smoother.get_smoothed(track_id)
            if smoothed is None:
                label = "Analyzing..."
                color = (160, 160, 160)
            else:
                age, gender, confidence, agreement, age_std = smoothed
                flag = flag_prediction(
                    gender_confidence=confidence, agreement_ratio=agreement, age_std=age_std
                )
                # Cross-model disagreement on the most recent inference is
                # an additional, independent reason to flag uncertainty.
                if self._last_models_agreed.get(track_id) is False:
                    flag["uncertain"] = True
                    flag["reasons"].append("models disagreed")

                label = f"ID{track_id} {gender}, ~{age}y"
                if flag["uncertain"]:
                    label += " (uncertain)"
                    color = (0, 165, 255)
                else:
                    color = (60, 200, 60)
                self.session_log.record(track_id, age, gender, confidence, agreement)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, tracks
