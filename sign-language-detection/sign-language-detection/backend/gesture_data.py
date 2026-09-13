"""
Synthetic landmark generator.

No hand-gesture image dataset is bundled with this project (keeps the repo
small and license-free), so we generate training data procedurally: each
of the 8 known words is defined as a target "curl" (0 = straight,
1 = fully curled) for each of the 5 fingers, plus a splay factor. A simple
2-segment forward-kinematics model turns those parameters into a full
21-point MediaPipe-style hand skeleton, and we sample many noisy /
rotated / scaled variations per class to train a robust classifier.

Swap this out for a real recorded dataset later by replacing
`build_dataset()` with landmarks captured from your own webcam via
/collect (see app.py) — the feature extraction and model code do not
change.
"""
import numpy as np

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# base angle (degrees, 0=+x axis, 90=straight up/away from wrist) and
# (mcp_radius, seg2, seg3, seg4) bone lengths, tuned to loosely resemble
# real hand proportions
FINGER_GEOMETRY = {
    "thumb":  {"angle": 55,  "lengths": (0.20, 0.20, 0.16, 0.14)},
    "index":  {"angle": 70,  "lengths": (0.38, 0.30, 0.18, 0.14)},
    "middle": {"angle": 90,  "lengths": (0.40, 0.33, 0.20, 0.16)},
    "ring":   {"angle": 110, "lengths": (0.38, 0.30, 0.19, 0.15)},
    "pinky":  {"angle": 128, "lengths": (0.34, 0.22, 0.15, 0.13)},
}

# GESTURES[label] = {finger: curl in [0,1]}, curl 0=straight, 1=fully curled
GESTURES = {
    "Hello":     {"thumb": 0.1, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "splay": 1.3},
    "Yes":       {"thumb": 0.9, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0, "splay": 0.6},
    "Peace":     {"thumb": 0.9, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0, "splay": 1.1},
    "Good":      {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0, "splay": 0.7, "thumb_up": True},
    "Stop":      {"thumb": 0.15, "index": 0.05, "middle": 0.05, "ring": 0.05, "pinky": 0.05, "splay": 0.75},
    "ILoveYou":  {"thumb": 0.0, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0, "splay": 1.15},
    "OK":        {"thumb": 0.55, "index": 0.55, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "splay": 1.0, "pinch": True},
    "Point":     {"thumb": 0.9, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0, "splay": 0.9},
}

LABELS = list(GESTURES.keys())


def _rot2d(v, deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _build_finger(base_angle, lengths, curl, thumb=False):
    """Return the 4 landmark positions (mcp, pip/ip, dip, tip) for one finger."""
    mcp_r, l2, l3, l4 = lengths
    dir0 = _rot2d(np.array([0.0, 1.0]), base_angle - 90)  # base_angle measured from +x
    mcp = dir0 * mcp_r

    # bending increases with curl, each successive joint bends further "inward"
    max_bend1 = 55 if thumb else 100
    max_bend2 = 40 if thumb else 80
    bend1 = curl * max_bend1
    bend2 = curl * max_bend2

    dir1 = _rot2d(dir0, -bend1)
    pip = mcp + dir1 * l2

    dir2 = _rot2d(dir1, -bend2)
    dip = pip + dir2 * l3

    dir3 = _rot2d(dir2, -bend2 * 0.6)
    tip = dip + dir3 * l4

    return [mcp, pip, dip, tip]


def generate_landmarks(label, rng, noise=0.02, rotation_jitter=12, scale_jitter=0.08):
    """Generate one noisy 21x3 landmark sample for the given gesture label."""
    spec = GESTURES[label]
    splay = spec.get("splay", 1.0)

    points_2d = [np.array([0.0, 0.0])]  # wrist
    for finger in FINGER_ORDER:
        geo = FINGER_GEOMETRY[finger]
        base_angle = 90 + (geo["angle"] - 90) * splay
        curl = spec[finger]
        pts = _build_finger(base_angle, geo["lengths"], curl, thumb=(finger == "thumb"))
        points_2d.extend(pts)

    points_2d = np.array(points_2d)  # (21, 2)

    # OK-sign / pinch gestures: pull thumb tip toward index tip
    if spec.get("pinch"):
        thumb_tip_idx, index_tip_idx = 4, 8
        mid = (points_2d[thumb_tip_idx] + points_2d[index_tip_idx]) / 2
        points_2d[thumb_tip_idx] = mid + (points_2d[thumb_tip_idx] - mid) * 0.3
        points_2d[index_tip_idx] = mid + (points_2d[index_tip_idx] - mid) * 0.3

    if spec.get("thumb_up"):
        # rotate thumb to point straight up rather than sideways
        points_2d[1:5] = [_rot2d(p, 35) for p in points_2d[1:5]]

    # global rotation + uniform scale jitter (simulates hand tilt / camera distance)
    rot = rng.uniform(-rotation_jitter, rotation_jitter)
    scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    points_2d = np.array([_rot2d(p, rot) * scale for p in points_2d])

    # small per-point noise (simulates landmark-detector jitter)
    points_2d = points_2d + rng.normal(0, noise, points_2d.shape)

    # small synthetic z (depth) — near-zero with slight per-finger variation
    z = rng.normal(0, 0.015, (21, 1))
    points_3d = np.hstack([points_2d, z])
    return points_3d


def build_dataset(samples_per_class=300, seed=42):
    rng = np.random.default_rng(seed)
    X_landmarks, y = [], []
    for label in LABELS:
        for _ in range(samples_per_class):
            X_landmarks.append(generate_landmarks(label, rng))
            y.append(label)
    return X_landmarks, y
