import cv2
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
car_cascade = cv2.CascadeClassifier(os.path.join(BASE, "models", "cars.xml"))
body_cascade = cv2.CascadeClassifier(os.path.join(BASE, "models", "haarcascade_fullbody.xml"))

# HSV colour ranges -> label
COLOR_RANGES = [
    ("Red",    (0, 70, 50),   (10, 255, 255)),
    ("Red",    (170, 70, 50), (180, 255, 255)),
    ("Orange", (11, 70, 50),  (20, 255, 255)),
    ("Yellow", (21, 70, 50),  (34, 255, 255)),
    ("Green",  (35, 40, 40),  (85, 255, 255)),
    ("Blue",   (86, 40, 40),  (135, 255, 255)),
    ("Purple", (136, 40, 40), (160, 255, 255)),
]

def classify_color(bgr_roi):
    """Return a colour name for a cropped car region using HSV histogram voting,
    with separate low-saturation handling for black/white/gray/silver."""
    if bgr_roi.size == 0:
        return "Unknown"
    h, w = bgr_roi.shape[:2]
    # sample the central 60% to avoid background/shadow pixels at edges
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    roi = bgr_roi[y0:y1, x0:x1]
    if roi.size == 0:
        roi = bgr_roi
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    s = pixels[:, 1].astype(np.float32)
    v = pixels[:, 2].astype(np.float32)
    low_sat = s < 45
    frac_low_sat = np.mean(low_sat)

    if frac_low_sat > 0.55:
        mean_v = np.mean(v[low_sat])
        if mean_v < 60:
            return "Black"
        elif mean_v > 175:
            return "White"
        else:
            return "Silver/Gray"

    votes = {}
    colored = pixels[~low_sat]
    if colored.shape[0] == 0:
        colored = pixels
    for name, lo, hi in COLOR_RANGES:
        lo = np.array(lo); hi = np.array(hi)
        mask = np.all((colored >= lo) & (colored <= hi), axis=1)
        count = int(np.sum(mask))
        if count:
            votes[name] = votes.get(name, 0) + count
    if not votes:
        return "Silver/Gray"
    return max(votes, key=votes.get)


def detect(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40))
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 60))

    out = image_bgr.copy()
    car_results = []

    for (x, y, w, h) in cars:
        roi = image_bgr[y:y + h, x:x + w]
        color_name = classify_color(roi)
        is_blue = color_name == "Blue"
        box_color = (0, 0, 255) if is_blue else (255, 0, 0)  # BGR: red for blue cars, blue for others
        cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 2)
        label = f"{color_name}"
        cv2.putText(out, label, (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
        car_results.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "color": color_name})

    for (x, y, w, h) in bodies:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(out, f"Cars: {len(cars)}  People: {len(bodies)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, f"Cars: {len(cars)}  People: {len(bodies)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return out, {
        "car_count": len(cars),
        "people_count": len(bodies),
        "cars": car_results,
    }
