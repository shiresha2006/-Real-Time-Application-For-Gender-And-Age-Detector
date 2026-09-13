"""
Temporal smoothing of per-face predictions.

A raw per-frame prediction is noisy: pose, blur, and lighting cause a
single frame's classification to disagree with the frame before and
after it, even though the person obviously hasn't changed. Rather than
display raw noisy output, each tracked face accumulates a short rolling
history and reports a smoothed, more trustworthy estimate:

  - Age: running average over the last N successful inferences.
  - Gender: majority vote over the last N inferences, weighted by each
    inference's own confidence -- a single low-confidence flip doesn't
    override several confident agreeing predictions.

This also gives a natural, honest notion of "uncertainty": if recent
predictions disagree a lot, that track is flagged as low-confidence in
the UI and in the fairness/uncertainty audit panel, instead of silently
picking one label and asserting it as fact.
"""

from __future__ import annotations
from collections import deque, defaultdict


class TrackSmoother:
    def __init__(self, history_len: int = 8):
        self.history_len = history_len
        self._ages = defaultdict(lambda: deque(maxlen=history_len))
        self._genders = defaultdict(lambda: deque(maxlen=history_len))  # (label, conf)

    def update(self, track_id: int, age: int, gender: str, gender_confidence: float):
        self._ages[track_id].append(age)
        self._genders[track_id].append((gender, gender_confidence))

    def get_smoothed(self, track_id: int):
        """Returns (smoothed_age, smoothed_gender, avg_confidence, agreement_ratio, age_std)
        or None if no data yet.

        avg_confidence: mean of the model's own reported confidence, over
                        only the frames that agreed with the winning label.
                        This is the real signal for "was the model actually
                        sure" -- distinct from agreement_ratio below.
        agreement_ratio: fraction of frames whose label matches the winning
                        label. A model that outputs the same label every
                        frame at 51% confidence scores 1.0 here despite
                        being weak -- that's exactly why avg_confidence is
                        tracked separately instead of reusing this value.
        age_std:        standard deviation of the raw age readings in the
                        window -- a large spread (e.g. predictions swinging
                        between 25 and 45) is itself a sign of an unreliable
                        read, even if a single average age looks plausible.
        """
        ages = self._ages.get(track_id)
        genders = self._genders.get(track_id)
        if not ages or not genders:
            return None

        smoothed_age = round(sum(ages) / len(ages))
        if len(ages) > 1:
            mean_age = sum(ages) / len(ages)
            age_std = (sum((a - mean_age) ** 2 for a in ages) / len(ages)) ** 0.5
        else:
            age_std = 0.0

        weighted = defaultdict(float)
        for label, conf in genders:
            weighted[label] += conf
        smoothed_gender = max(weighted, key=weighted.get)

        matching_confidences = [conf for label, conf in genders if label == smoothed_gender]
        avg_confidence = sum(matching_confidences) / len(matching_confidences)

        agree = len(matching_confidences)
        agreement_ratio = agree / len(genders)

        return (
            smoothed_age,
            smoothed_gender,
            round(avg_confidence, 2),
            round(agreement_ratio, 2),
            round(age_std, 1),
        )

    def forget(self, track_id: int):
        self._ages.pop(track_id, None)
        self._genders.pop(track_id, None)

    def active_ids(self):
        return set(self._ages.keys())
