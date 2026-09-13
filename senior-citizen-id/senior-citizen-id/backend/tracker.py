"""
Lightweight centroid tracker so the same person isn't logged as a new
"visit" on every single frame of a live feed or video. Not a full
re-identification system (it has no memory of appearance) — it's a
positional tracker good enough for a single fixed camera watching an
entrance, which is the brief's scenario.

Logic: a detection matches an existing track if its centroid is close
enough (relative to face size) to that track's last known position.
A track is logged as a "visit" the moment it's created, and again if it
re-matches after sitting unlogged for longer than REVISIT_COOLDOWN_S
(handles someone leaving and re-entering frame later). Tracks not seen
for STALE_AFTER_S are dropped so ids don't leak memory forever.
"""
import csv
import os
import threading
import time
from datetime import datetime

REVISIT_COOLDOWN_S = 20
STALE_AFTER_S = 5
MATCH_DIST_FACTOR = 0.8  # centroid must be within this * face_width to match

LOG_PATH = os.path.join(os.path.dirname(__file__), "visitor_log.csv")
LOG_FIELDS = ["timestamp", "track_id", "gender", "gender_confidence", "estimated_age", "senior_citizen", "source"]

_lock = threading.Lock()


def ensure_log_file():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def append_log_row(row: dict):
    ensure_log_file()
    with _lock:
        with open(LOG_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def read_log_rows():
    ensure_log_file()
    with _lock:
        with open(LOG_PATH, "r", newline="") as f:
            return list(csv.DictReader(f))


def clear_log():
    with _lock:
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


class Track:
    __slots__ = ("id", "cx", "cy", "face_w", "last_seen", "last_logged_at", "gender", "age")

    def __init__(self, track_id, cx, cy, face_w, gender, age):
        self.id = track_id
        self.cx, self.cy, self.face_w = cx, cy, face_w
        self.last_seen = time.time()
        self.last_logged_at = 0.0
        self.gender = gender
        self.age = age


class VisitorTracker:
    """One instance per video stream / session (live feed or one uploaded video)."""

    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def _find_match(self, cx, cy, face_w):
        best, best_dist = None, None
        for t in self.tracks:
            dist = ((t.cx - cx) ** 2 + (t.cy - cy) ** 2) ** 0.5
            if dist < face_w * MATCH_DIST_FACTOR and (best_dist is None or dist < best_dist):
                best, best_dist = t, dist
        return best

    def gc(self):
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t.last_seen < STALE_AFTER_S]

    def update(self, cx, cy, face_w, gender, gender_conf, age, source, now=None):
        """Returns (track_id, was_logged: bool)."""
        now = now or time.time()
        match = self._find_match(cx, cy, face_w)

        if match is None:
            track = Track(self._next_id, cx, cy, face_w, gender, age)
            self._next_id += 1
            self.tracks.append(track)
            self._log(track, gender, gender_conf, age, source, now)
            return track.id, True

        match.cx, match.cy, match.face_w = cx, cy, face_w
        match.last_seen = now
        match.gender, match.age = gender, age
        if now - match.last_logged_at > REVISIT_COOLDOWN_S:
            self._log(match, gender, gender_conf, age, source, now)
            return match.id, True
        return match.id, False

    def _log(self, track, gender, gender_conf, age, source, now):
        track.last_logged_at = now
        append_log_row({
            "timestamp": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": track.id,
            "gender": gender,
            "gender_confidence": round(gender_conf, 3),
            "estimated_age": round(age, 1),
            "senior_citizen": "Yes" if age > 60 else "No",
            "source": source,
        })
