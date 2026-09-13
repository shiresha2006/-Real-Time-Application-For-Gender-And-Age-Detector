"""
Landmark -> feature vector conversion, shared by training and inference.

MediaPipe Hands gives 21 (x, y, z) landmarks per hand. Raw pixel/normalized
coordinates are not translation/scale invariant, so we re-express every
landmark relative to the wrist (landmark 0) and scale by the hand's own
size (wrist -> middle-finger-MCP distance). On top of that we add a small
set of hand-crafted geometric features (per-finger "curl" angles and
fingertip-to-wrist distances) that are the actual signal classic sign-
language-recognition systems use, since they are invariant to hand size,
camera distance and, to a good extent, in-plane rotation.
"""
import numpy as np

WRIST = 0
FINGER_JOINTS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def _angle(a, b, c):
    """Angle at point b, formed by a-b-c, in radians."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def landmarks_to_vector(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples (MediaPipe normalized image
    coordinates, z relative depth).
    Returns a fixed-length 1D numpy feature vector.
    """
    pts = np.array(landmarks, dtype=np.float32)
    wrist = pts[WRIST]

    # scale reference: wrist -> middle finger MCP (landmark 9)
    scale = np.linalg.norm(pts[9] - wrist) + 1e-6

    centered = (pts - wrist) / scale
    flat_coords = centered.flatten()  # 21*3 = 63 values

    # per-finger curl angle at the PIP/first-knuckle joint
    curl_angles = []
    fingertip_dists = []
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        curl_angles.append(_angle(pts[mcp], pts[pip], pts[dip]))
        fingertip_dists.append(np.linalg.norm(pts[tip] - wrist) / scale)

    # thumb-to-each-fingertip distances (captures pinch / crossed-thumb shapes)
    thumb_tip = pts[4]
    cross_dists = [
        np.linalg.norm(thumb_tip - pts[tip]) / scale for tip in (8, 12, 16, 20)
    ]

    vec = np.concatenate(
        [flat_coords, np.array(curl_angles), np.array(fingertip_dists), np.array(cross_dists)]
    )
    return vec.astype(np.float32)


FEATURE_LENGTH = 63 + 5 + 5 + 4  # = 77
