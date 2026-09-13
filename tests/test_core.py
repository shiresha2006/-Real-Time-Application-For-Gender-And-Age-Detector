"""
Basic unit tests for the non-ML core logic (tracker, smoothing, analytics,
fairness audit). These don't require a webcam or the DeepFace model, so
they run fast and are safe to include as CI-style evidence in the repo.

Run: python -m pytest tests/ -v   (or: python tests/test_core.py)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracker import CentroidTracker
from src.smoothing import TrackSmoother
from src.analytics import SessionLog, age_to_bucket
from src.fairness_audit import flag_prediction, session_fairness_report


def test_tracker_assigns_stable_id_across_frames():
    tracker = CentroidTracker()
    ids_frame1 = tracker.update([(10, 10, 50, 50)])
    ids_frame2 = tracker.update([(12, 11, 50, 50)])  # face moved slightly
    assert list(ids_frame1.keys()) == list(ids_frame2.keys()), "Same face should keep the same ID"


def test_tracker_drops_after_max_disappeared():
    tracker = CentroidTracker(max_disappeared=2)
    tracker.update([(10, 10, 50, 50)])
    tracker.update([])
    tracker.update([])
    result = tracker.update([])
    assert len(result) == 0, "Track should be dropped after exceeding max_disappeared"


def test_smoother_averages_age_and_votes_gender():
    smoother = TrackSmoother(history_len=5)
    smoother.update(0, age=30, gender="Man", gender_confidence=0.9)
    smoother.update(0, age=32, gender="Man", gender_confidence=0.8)
    smoother.update(0, age=31, gender="Woman", gender_confidence=0.3)  # noisy outlier
    age, gender, confidence, agreement, age_std = smoother.get_smoothed(0)
    assert 29 <= age <= 33
    assert gender == "Man", "Majority + confidence-weighted vote should reject the low-confidence outlier"
    assert 0 <= agreement <= 1
    assert 0 <= confidence <= 1


def test_smoother_flags_low_confidence_even_with_full_agreement():
    # Model repeats the same label every frame, but at weak confidence each time.
    # agreement_ratio would be 1.0; avg_confidence must still be low so the
    # fairness flag can catch this case (this is the bug fix under test).
    smoother = TrackSmoother(history_len=5)
    for _ in range(5):
        smoother.update(0, age=30, gender="Man", gender_confidence=0.52)
    _, gender, confidence, agreement, _ = smoother.get_smoothed(0)
    assert agreement == 1.0
    assert confidence < 0.6, "avg_confidence must reflect real model confidence, not agreement"


def test_age_to_bucket():
    assert age_to_bucket(8) == "Child (0-12)"
    assert age_to_bucket(25) == "Young Adult (20-32)"
    assert age_to_bucket(70) == "Senior (61+)"


def test_session_log_summary_and_csv():
    log = SessionLog()
    log.record(track_id=0, age=25, gender="Woman", confidence=0.9, agreement=0.9)
    log.record(track_id=1, age=40, gender="Man", confidence=0.5, agreement=0.5)
    summary = log.summary()
    assert summary["total_events"] == 2
    assert summary["unique_faces"] == 2
    csv_bytes = log.to_csv_bytes()
    assert b"track_id" in csv_bytes


def test_fairness_flagging():
    confident = flag_prediction(gender_confidence=0.9, agreement_ratio=0.9)
    uncertain = flag_prediction(gender_confidence=0.4, agreement_ratio=0.4)
    assert confident["uncertain"] is False
    assert uncertain["uncertain"] is True
    assert len(uncertain["reasons"]) > 0


def test_session_fairness_report_text():
    log = SessionLog()
    log.record(track_id=0, age=25, gender="Woman", confidence=0.9, agreement=0.9)
    report = session_fairness_report(log.summary())
    assert "predictions" in report


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
