"""
Session analytics: turns raw predictions into something a viewer of the
demo can actually read -- a running log, age-group distribution, and
gender ratio -- and lets the session be exported as CSV evidence.
"""

from __future__ import annotations
import csv
import io
import time
from collections import Counter


def age_to_bucket(age: int) -> str:
    buckets = [(0, 12, "Child (0-12)"), (13, 19, "Teen (13-19)"),
               (20, 32, "Young Adult (20-32)"), (33, 45, "Adult (33-45)"),
               (46, 60, "Middle Age (46-60)"), (61, 200, "Senior (61+)")]
    for lo, hi, label in buckets:
        if lo <= age <= hi:
            return label
    return "Unknown"


class SessionLog:
    def __init__(self):
        self.entries = []  # each: dict(timestamp, track_id, age, gender, confidence, agreement)

    def record(self, track_id: int, age: int, gender: str, confidence: float, agreement: float):
        self.entries.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": track_id,
            "age": age,
            "age_group": age_to_bucket(age),
            "gender": gender,
            "gender_confidence": confidence,
            "prediction_agreement": agreement,
        })

    def summary(self):
        if not self.entries:
            return {"total_events": 0, "unique_faces": 0, "gender_counts": {}, "age_group_counts": {}}
        genders = Counter(e["gender"] for e in self.entries)
        age_groups = Counter(e["age_group"] for e in self.entries)
        unique_faces = len({e["track_id"] for e in self.entries})
        low_conf = sum(1 for e in self.entries if e["prediction_agreement"] < 0.6)
        return {
            "total_events": len(self.entries),
            "unique_faces": unique_faces,
            "gender_counts": dict(genders),
            "age_group_counts": dict(age_groups),
            "low_confidence_events": low_conf,
        }

    def to_csv_bytes(self) -> bytes:
        buf = io.StringIO()
        if self.entries:
            writer = csv.DictWriter(buf, fieldnames=list(self.entries[0].keys()))
            writer.writeheader()
            writer.writerows(self.entries)
        return buf.getvalue().encode("utf-8")
