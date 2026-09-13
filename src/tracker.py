"""
Lightweight multi-object centroid tracker.

Most beginner age/gender-detector projects re-run classification on every
detected face in every frame, with no notion of "this is the same person
I saw 3 frames ago." That causes two visible problems on a live demo:

1. Predictions flicker (Male -> Female -> Male) frame to frame.
2. It's wasteful: the expensive DeepFace age/gender model gets called on
   every face, every frame, even though a face rarely changes age/gender
   between two frames 30ms apart.

This tracker assigns a stable integer ID to each face using a simple
centroid-distance matching scheme (no external dependency, no GPU, cheap
enough to run at real-time frame rates). Downstream code uses the ID to:
  - keep a rolling prediction history per face -> temporal smoothing
  - decide when to re-run the heavy model on a track (every N frames)
    instead of every frame -> real-time performance optimization
"""

from __future__ import annotations
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared: int = 15, max_distance: int = 90):
        """
        max_disappeared: how many consecutive frames a track may go
                          undetected before it is dropped.
        max_distance:    max pixel distance between centroids to be
                          considered the same face.
        """
        self.next_object_id = 0
        self.objects: "OrderedDict[int, tuple]" = OrderedDict()   # id -> centroid
        self.bboxes: "OrderedDict[int, tuple]" = OrderedDict()    # id -> (x,y,w,h)
        self.disappeared: "OrderedDict[int, int]" = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        rects: list of (x, y, w, h) face boxes detected in the current frame.
        returns: dict {object_id: (x, y, w, h)} of currently tracked faces.
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return dict(self.bboxes)

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        if len(self.objects) == 0:
            for i, rect in enumerate(rects):
                self._register(tuple(input_centroids[i]), rect)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            D = np.linalg.norm(
                object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = tuple(input_centroids[col])
                self.bboxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            for col in unused_cols:
                self._register(tuple(input_centroids[col]), rects[col])

        return dict(self.bboxes)
