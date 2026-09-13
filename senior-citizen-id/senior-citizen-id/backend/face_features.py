"""
Per-face feature extraction, used by both training (synthetic features)
and inference (real MediaPipe FaceMesh landmarks + pixel crops).

Two feature groups, kept separate because they come from genuinely
different signals and drive two different predictions:

  GENDER features - geometric ratios from face-mesh landmarks (jaw width,
  brow-eye distance, lip thickness, etc.) normalized by face size, so
  they're roughly invariant to how close the person is to the camera.

  AGE features - pixel-level cues in the forehead / eye-corner / cheek
  regions (edge density as a wrinkle proxy, grey-hair ratio near the
  hairline, skin texture variance) that tend to increase with age.

This is a classical (pre-deep-learning) computer-vision approach:
real, meaningful signals, but noisier than a CNN trained on a large
labelled face dataset. See README for that trade-off.
"""
import numpy as np
import cv2

# stable MediaPipe FaceMesh landmark indices used as anchor points
FOREHEAD, CHIN = 10, 152
LEFT_FACE, RIGHT_FACE = 234, 454
LEFT_EYE_INNER, RIGHT_EYE_INNER = 133, 362
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263
LEFT_EYE_UPPER, RIGHT_EYE_UPPER = 159, 386
LEFT_BROW, RIGHT_BROW = 105, 334
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP_OUTER, LIP_BOTTOM_OUTER = 0, 17

GENDER_FEATURE_NAMES = [
    "face_h_w_ratio", "jaw_to_face_ratio", "brow_eye_ratio",
    "lip_thickness_ratio", "eye_spacing_ratio", "mouth_width_ratio",
]
AGE_FEATURE_NAMES = [
    "edge_density_forehead", "edge_density_eyecorner",
    "grey_hair_ratio", "skin_texture_variance",
]


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def landmarks_to_gender_features(px):
    """px: dict index -> (x, y) pixel coords."""
    face_w = _dist(px[LEFT_FACE], px[RIGHT_FACE]) + 1e-6
    face_h = _dist(px[FOREHEAD], px[CHIN]) + 1e-6
    jaw_w = _dist(px[LEFT_FACE], px[RIGHT_FACE])  # approx jaw span at cheek level
    brow_eye = (_dist(px[LEFT_BROW], px[LEFT_EYE_UPPER]) + _dist(px[RIGHT_BROW], px[RIGHT_EYE_UPPER])) / 2
    lip_thickness = _dist(px[LIP_TOP_OUTER], px[LIP_BOTTOM_OUTER])
    eye_spacing = _dist(px[LEFT_EYE_INNER], px[RIGHT_EYE_INNER])
    mouth_w = _dist(px[MOUTH_LEFT], px[MOUTH_RIGHT])

    return np.array([
        face_h / face_w,
        jaw_w / face_w,
        brow_eye / face_h,
        lip_thickness / face_h,
        eye_spacing / face_w,
        mouth_w / face_w,
    ], dtype=np.float32)


def _safe_crop(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0, x1 = sorted((max(0, min(int(x0), w)), max(0, min(int(x1), w))))
    y0, y1 = sorted((max(0, min(int(y0), h)), max(0, min(int(y1), h))))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return img[y0:y1, x0:x1]


def _edge_density(crop):
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    edges = cv2.Canny(gray, 60, 140)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _grey_ratio(crop):
    if crop is None or crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    grey_mask = (sat < 0.25) & (val > 0.35)
    return float(np.count_nonzero(grey_mask)) / float(grey_mask.size)


def _texture_variance(crop):
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.clip(var / 800.0, 0.0, 1.0))  # empirical normalization


def landmarks_to_age_features(bgr_image, px):
    face_w = _dist(px[LEFT_FACE], px[RIGHT_FACE]) + 1e-6
    face_h = _dist(px[FOREHEAD], px[CHIN]) + 1e-6

    fx, fy = px[FOREHEAD]
    lx, ly = px[LEFT_FACE]
    rx, ry = px[RIGHT_FACE]

    forehead_crop = _safe_crop(bgr_image, min(lx, rx), fy - 0.28 * face_h, max(lx, rx), fy + 0.12 * face_h)

    eye_patch_size = 0.14 * face_w
    left_eye_crop = _safe_crop(
        bgr_image, px[LEFT_EYE_OUTER][0] - eye_patch_size, px[LEFT_EYE_OUTER][1] - eye_patch_size / 2,
        px[LEFT_EYE_OUTER][0] + eye_patch_size * 0.3, px[LEFT_EYE_OUTER][1] + eye_patch_size / 2,
    )
    right_eye_crop = _safe_crop(
        bgr_image, px[RIGHT_EYE_OUTER][0] - eye_patch_size * 0.3, px[RIGHT_EYE_OUTER][1] - eye_patch_size / 2,
        px[RIGHT_EYE_OUTER][0] + eye_patch_size, px[RIGHT_EYE_OUTER][1] + eye_patch_size / 2,
    )
    eye_edge_density = (_edge_density(left_eye_crop) + _edge_density(right_eye_crop)) / 2

    hair_crop = _safe_crop(bgr_image, min(lx, rx), fy - 0.55 * face_h, max(lx, rx), fy - 0.10 * face_h)
    grey_ratio = _grey_ratio(hair_crop)

    cheek_y = fy + 0.55 * face_h
    cheek_crop = _safe_crop(bgr_image, lx + 0.1 * face_w, cheek_y - 0.1 * face_h, lx + 0.4 * face_w, cheek_y + 0.1 * face_h)
    skin_var = _texture_variance(cheek_crop)

    return np.array([
        _edge_density(forehead_crop),
        eye_edge_density,
        grey_ratio,
        skin_var,
    ], dtype=np.float32)


def face_bbox_from_landmarks(px, img_w, img_h, pad=0.15):
    xs = [p[0] for p in px.values()]
    ys = [p[1] for p in px.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    x0 -= pad * w
    x1 += pad * w
    y0 -= pad * h
    y1 += pad * h
    return (
        max(0, int(x0)), max(0, int(y0)),
        min(img_w, int(x1)), min(img_h, int(y1)),
    )
