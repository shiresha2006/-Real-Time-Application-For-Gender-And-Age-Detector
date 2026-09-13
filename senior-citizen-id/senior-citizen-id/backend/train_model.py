"""
Trains two models and saves them together in model.pkl:
  - gender_clf: RandomForestClassifier over geometric face-ratio features
  - age_reg:    RandomForestRegressor over texture/colour features -> age

Run: python train_model.py
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error

from synthetic_data import build_gender_dataset, build_age_dataset

print("Training gender classifier...")
Xg, yg = build_gender_dataset()
le = LabelEncoder().fit(yg)
yg_enc = le.transform(yg)
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg_enc, test_size=0.2, random_state=1, stratify=yg_enc)
gender_clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=1, n_jobs=-1)
gender_clf.fit(Xg_tr, yg_tr)
gender_acc = accuracy_score(yg_te, gender_clf.predict(Xg_te))
print(f"  gender test accuracy: {gender_acc:.3f}")

print("Training age regressor...")
Xa, ya = build_age_dataset()
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.2, random_state=2)
age_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=2, n_jobs=-1)
age_reg.fit(Xa_tr, ya_tr)
age_mae = mean_absolute_error(ya_te, age_reg.predict(Xa_te))
print(f"  age regressor test MAE: {age_mae:.2f} years")

# sanity check: senior/not-senior accuracy derived from the regressor
senior_true = ya_te > 60
senior_pred = age_reg.predict(Xa_te) > 60
senior_acc = accuracy_score(senior_true, senior_pred)
print(f"  derived senior-citizen (>60) accuracy: {senior_acc:.3f}")

joblib.dump({"gender_clf": gender_clf, "gender_label_encoder": le, "age_reg": age_reg}, "model.pkl")
print("Saved model.pkl")
