"""
Train the sign/gesture classifier on the synthetic landmark dataset and
save it (+ label encoder) to model.pkl. Run: python train_model.py
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from gesture_data import build_dataset, LABELS
from features import landmarks_to_vector

print("Generating synthetic training data...")
X_landmarks, y_labels = build_dataset(samples_per_class=350)

print("Extracting features...")
X = np.array([landmarks_to_vector(lm) for lm in X_landmarks])

le = LabelEncoder()
le.fit(LABELS)
y = le.transform(y_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

clf = RandomForestClassifier(
    n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump({"model": clf, "label_encoder": le}, "model.pkl")
print("Saved model.pkl")
