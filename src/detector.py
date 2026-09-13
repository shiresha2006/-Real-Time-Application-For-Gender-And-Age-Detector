"""
Face detection + age/gender inference.

Face detection: OpenCV's DNN SSD detector (res10-based, 300x300 input).
It's far more robust than the Haar cascades used in most tutorial clones
(handles angled faces, partial occlusion, varied lighting) and still runs
comfortably in real time on CPU.

Age/Gender inference uses TWO independent models, combined as an ensemble:

1. Primary: DeepFace (Serengil, github.com/serengil/deepface), which
   wraps a VGG-Face-based age/gender model.
2. Secondary: a Levi & Hassner-style CNN loaded via OpenCV DNN (Caffe),
   trained separately from the primary model.

Running two independently-trained models and only trusting the result
when they agree is a meaningfully different (and more honest) design
than the typical tutorial, which reports a single model's raw output as
fact. Disagreement between the two is itself used as an uncertainty
signal downstream (see pipeline.py / fairness_audit.py) rather than
silently averaged away.

Performance note: both age/gender calls are comparatively expensive next
to face detection. Running them on every detected face on every frame
will not sustain real-time frame rates once there is more than one face
on screen. This module exposes them separately from `detect_faces`
specifically so the calling code (see pipeline.py) can schedule
inference per tracked face every N frames instead of every frame -- the
tracker supplies the temporal identity that makes that scheduling
possible.
"""

from __future__ import annotations
import os
import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROTO_PATH = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")

AGE_PROTO_PATH = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_WEIGHTS_PATH = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO_PATH = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "gender_net.caffemodel")

_AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
_GENDER_LIST = ['Man', 'Woman']
_CAFFE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

_face_net = None


def _get_face_net():
    global _face_net
    if _face_net is None:
        if os.path.exists(PROTO_PATH) and os.path.exists(WEIGHTS_PATH):
            _face_net = cv2.dnn.readNetFromTensorflow(WEIGHTS_PATH, PROTO_PATH)
        else:
            _face_net = None  # caller falls back to Haar cascade
    return _face_net


_haar = None


def _get_haar():
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _haar


def detect_faces(frame: np.ndarray, conf_threshold: float = 0.6):
    """
    Returns a list of (x, y, w, h) boxes in pixel coordinates.
    Uses the DNN detector when model files are present; otherwise falls
    back to a Haar cascade so the app never hard-fails on a missing model.
    """
    h, w = frame.shape[:2]
    net = _get_face_net()

    boxes = []
    if net is not None:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = _get_haar().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        boxes = [tuple(r) for r in rects]

    return boxes


def infer_age_gender(face_bgr: np.ndarray):
    """
    Runs DeepFace age/gender analysis on a single cropped face image (BGR).
    Returns dict: {"age": int, "gender": "Man"|"Woman", "gender_confidence": float 0-1}
    or None if the model could not process the crop (e.g. too small/blank).
    """
    from deepface import DeepFace  # imported lazily -- heavy import

    try:
        result = DeepFace.analyze(
            face_bgr,
            actions=["age", "gender"],
            enforce_detection=False,
            detector_backend="skip",  # we already cropped the face ourselves
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        gender_scores = result["gender"]  # {'Woman': %, 'Man': %}
        dominant = result["dominant_gender"]
        confidence = float(gender_scores[dominant]) / 100.0
        return {
            "age": int(result["age"]),
            "gender": dominant,
            "gender_confidence": confidence,
        }
    except Exception:
        return None


_age_net = None
_gender_net = None


def _get_secondary_nets():
    global _age_net, _gender_net
    if _age_net is None:
        if not (os.path.exists(AGE_PROTO_PATH) and os.path.exists(AGE_WEIGHTS_PATH)
                and os.path.exists(GENDER_PROTO_PATH) and os.path.exists(GENDER_WEIGHTS_PATH)):
            return None, None
        _age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO_PATH, AGE_WEIGHTS_PATH)
        _gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO_PATH, GENDER_WEIGHTS_PATH)
    return _age_net, _gender_net


def infer_age_gender_secondary(face_bgr: np.ndarray):
    """
    Independent second opinion using a Levi & Hassner-style CNN (OpenCV DNN,
    Caffe weights). Used only to cross-check the primary DeepFace result --
    see infer_age_gender_ensemble() -- not as a standalone replacement.
    Returns None if the model files aren't present in models/, so the app
    degrades gracefully to single-model mode instead of crashing.
    """
    age_net, gender_net = _get_secondary_nets()
    if age_net is None:
        return None

    try:
        blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), _CAFFE_MEAN, swapRB=False)

        gender_net.setInput(blob)
        g_pred = gender_net.forward()[0]
        gender = _GENDER_LIST[int(g_pred.argmax())]
        gender_confidence = float(g_pred.max())

        age_net.setInput(blob)
        a_pred = age_net.forward()[0]
        age_bracket = _AGE_LIST[int(a_pred.argmax())]
        lo, hi = age_bracket.strip("()").split("-")
        age_mid = int((int(lo) + int(hi)) / 2)

        return {"age": age_mid, "gender": gender, "gender_confidence": gender_confidence}
    except Exception:
        return None


def infer_age_gender_ensemble(face_bgr: np.ndarray):
    """
    Combines the primary (DeepFace) and secondary (Caffe CNN) models.

    - If only one model is available, returns that model's result as-is.
    - If both are available and agree on gender, confidence is the
      primary model's own confidence (the second model corroborates it).
    - If both are available and disagree, confidence is heavily
      downweighted -- disagreement between two independently-trained
      models is a stronger uncertainty signal than either model's own
      self-reported confidence, and should visibly lower trust rather
      than being silently averaged away.

    Returns dict: {"age": int, "gender": str, "gender_confidence": float,
    "models_agreed": bool} or None if no model could process the crop.
    """
    primary = infer_age_gender(face_bgr)
    secondary = infer_age_gender_secondary(face_bgr)

    if primary and not secondary:
        return {**primary, "models_agreed": None}
    if secondary and not primary:
        return {**secondary, "models_agreed": None}
    if not primary and not secondary:
        return None

    models_agreed = primary["gender"] == secondary["gender"]
    avg_age = round((primary["age"] + secondary["age"]) / 2)

    if models_agreed:
        combined_confidence = primary["gender_confidence"]
    else:
        combined_confidence = min(primary["gender_confidence"], secondary["gender_confidence"]) * 0.5

    return {
        "age": avg_age,
        "gender": primary["gender"],
        "gender_confidence": combined_confidence,
        "models_agreed": models_agreed,
    }



def _get_face_net():
    global _face_net
    if _face_net is None:
        if os.path.exists(PROTO_PATH) and os.path.exists(WEIGHTS_PATH):
            _face_net = cv2.dnn.readNetFromTensorflow(WEIGHTS_PATH, PROTO_PATH)
        else:
            _face_net = None  # caller falls back to Haar cascade
    return _face_net


_haar = None


def _get_haar():
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _haar


def detect_faces(frame: np.ndarray, conf_threshold: float = 0.6):
    """
    Returns a list of (x, y, w, h) boxes in pixel coordinates.
    Uses the DNN detector when model files are present; otherwise falls
    back to a Haar cascade so the app never hard-fails on a missing model.
    """
    h, w = frame.shape[:2]
    net = _get_face_net()

    boxes = []
    if net is not None:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = _get_haar().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        boxes = [tuple(r) for r in rects]

    return boxes


def infer_age_gender(face_bgr: np.ndarray):
    """
    Runs DeepFace age/gender analysis on a single cropped face image (BGR).
    Returns dict: {"age": int, "gender": "Man"|"Woman", "gender_confidence": float 0-1}
    or None if the model could not process the crop (e.g. too small/blank).
    """
    from deepface import DeepFace  # imported lazily -- heavy import

    try:
        result = DeepFace.analyze(
            face_bgr,
            actions=["age", "gender"],
            enforce_detection=False,
            detector_backend="skip",  # we already cropped the face ourselves
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        gender_scores = result["gender"]  # {'Woman': %, 'Man': %}
        dominant = result["dominant_gender"]
        confidence = float(gender_scores[dominant]) / 100.0
        return {
            "age": int(result["age"]),
            "gender": dominant,
            "gender_confidence": confidence,
        }
    except Exception:
        return None
