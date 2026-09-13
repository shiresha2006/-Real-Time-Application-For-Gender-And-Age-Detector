import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
M = lambda f: os.path.join(BASE, "models", f)

face_net = cv2.dnn.readNet(M("opencv_face_detector_uint8.pb"), M("opencv_face_detector.pbtxt"))
age_net = cv2.dnn.readNet(M("age_net.caffemodel"), M("age_deploy.prototxt"))
gender_net = cv2.dnn.readNet(M("gender_net.caffemodel"), M("gender_deploy.prototxt"))

MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
AGE_REP = [1, 5, 10, 18, 28, 40, 50, 70]  # representative age used for the 20-30 range check
GENDER_LIST = ["Male", "Female"]


def detect_faces(frame, conf_threshold=0.6):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def predict_age_gender(frame, box, pad=20):
    x, y, w, h = box
    H, W = frame.shape[:2]
    face = frame[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    if face.size == 0:
        return "Unknown", 0, "Unknown"
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN, swapRB=False)

    gender_net.setInput(blob)
    g_preds = gender_net.forward()
    gender = GENDER_LIST[int(np.argmax(g_preds))]

    age_net.setInput(blob)
    a_preds = age_net.forward()
    idx = int(np.argmax(a_preds))
    age_bucket = AGE_BUCKETS[idx]
    age_rep = AGE_REP[idx]
    return gender, age_rep, age_bucket


def estimate_hair_length(frame, box):
    """Heuristic hair-length estimate: looks at the region beside/below the face
    (jaw -> shoulders) and measures how much of it is covered by dark, non-skin
    'hair-like' pixels. Returns ('Long'|'Short', coverage_ratio)."""
    x, y, w, h = box
    H, W = frame.shape[:2]

    rx1 = max(0, x - int(0.35 * w))
    rx2 = min(W, x + w + int(0.35 * w))
    ry1 = max(0, y + int(0.55 * h))          # start around chin level
    ry2 = min(H, y + int(2.1 * h))           # extend down toward shoulders

    region = frame[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return "Short", 0.0

    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array((0, 133, 77), dtype=np.uint8)
    upper_skin = np.array((255, 173, 127), dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark_mask = (v < 100).astype(np.uint8) * 255

    hair_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(skin_mask))
    coverage = float(np.mean(hair_mask > 0))

    label = "Long" if coverage > 0.22 else "Short"
    return label, coverage


def apply_task_logic(age_rep, gender_raw, hair_label):
    """Deliberate rule required by the task spec:
    - Age NOT in [20,30]: report the raw predicted gender, ignore hair.
    - Age IN [20,30]: override gender purely from hair length —
      long hair -> Female, short hair -> Male — regardless of true gender.
    """
    in_range = 20 <= age_rep <= 30
    if in_range:
        final_gender = "Female" if hair_label == "Long" else "Male"
        rule = f"Age {age_rep} is within 20-30 -> gender forced from hair length ({hair_label})"
    else:
        final_gender = gender_raw
        rule = f"Age {age_rep} is outside 20-30 -> using model's raw gender prediction"
    return final_gender, in_range, rule


def process(frame):
    boxes = detect_faces(frame)
    out = frame.copy()
    results = []

    for box in boxes:
        x, y, w, h = box
        gender_raw, age_rep, age_bucket = predict_age_gender(frame, box)
        hair_label, coverage = estimate_hair_length(frame, box)
        final_gender, in_range, rule = apply_task_logic(age_rep, gender_raw, hair_label)

        color = (0, 200, 255) if in_range else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text1 = f"{final_gender}  age~{age_bucket}"
        text2 = f"hair:{hair_label}"
        cv2.putText(out, text1, (x, max(y - 26, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(out, text2, (x, max(y - 6, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        results.append({
            "box": {"x": x, "y": y, "w": w, "h": h},
            "raw_gender": gender_raw,
            "age_bucket": age_bucket,
            "age_representative": age_rep,
            "hair_length": hair_label,
            "hair_coverage": round(coverage, 3),
            "in_target_age_range": in_range,
            "final_gender": final_gender,
            "rule_applied": rule,
        })

    return out, results
