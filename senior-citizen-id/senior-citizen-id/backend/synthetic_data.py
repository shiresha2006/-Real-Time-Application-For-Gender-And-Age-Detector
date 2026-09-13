"""
Synthetic training data, generated directly at the feature level (not as
raw images/landmarks — see face_features.py for what these numbers mean).
Distributions are set from well-known coarse anthropometric and skin-
texture trends, then jittered. This lets the classifier learn sensible
decision boundaries without needing a bundled face-image dataset; real
MediaPipe landmarks + pixel crops feed the *same* feature functions at
inference time.

Replace with real, labelled feature extractions (run extract on a folder
of consented, labelled photos) for meaningfully better accuracy.
"""
import numpy as np

GENDER_FEATURE_NAMES = [
    "face_h_w_ratio", "jaw_to_face_ratio", "brow_eye_ratio",
    "lip_thickness_ratio", "eye_spacing_ratio", "mouth_width_ratio",
]
AGE_FEATURE_NAMES = [
    "edge_density_forehead", "edge_density_eyecorner",
    "grey_hair_ratio", "skin_texture_variance",
]

# (mean, std) per feature, per class
GENDER_DIST = {
    "Male": [(1.30, 0.07), (0.78, 0.05), (0.10, 0.02), (0.045, 0.010), (0.30, 0.03), (0.45, 0.04)],
    "Female": [(1.22, 0.07), (0.68, 0.05), (0.14, 0.02), (0.065, 0.012), (0.29, 0.03), (0.40, 0.04)],
}

AGE_DIST = {
    # class -> (feature dists, age range for the numeric target)
    "Adult": ([(0.05, 0.02), (0.04, 0.015), (0.05, 0.04), (0.20, 0.08)], (18, 59)),
    "Senior": ([(0.16, 0.04), (0.14, 0.035), (0.45, 0.20), (0.55, 0.15)], (60, 85)),
}


def _sample(dist, rng, n):
    return np.array([
        np.clip(rng.normal(mean, std, n), 0.0, None) for mean, std in dist
    ]).T


def build_gender_dataset(samples_per_class=400, seed=1):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, dist in GENDER_DIST.items():
        X.append(_sample(dist, rng, samples_per_class))
        y += [label] * samples_per_class
    return np.vstack(X), y


def build_age_dataset(samples_per_class=400, seed=2):
    rng = np.random.default_rng(seed)
    X, ages = [], []
    for label, (dist, (lo, hi)) in AGE_DIST.items():
        X.append(_sample(dist, rng, samples_per_class))
        ages += list(rng.uniform(lo, hi, samples_per_class))
    return np.vstack(X), np.array(ages, dtype=np.float32)
