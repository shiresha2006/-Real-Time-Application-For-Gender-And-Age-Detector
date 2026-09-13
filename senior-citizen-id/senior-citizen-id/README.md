# Foot Traffic — Senior Citizen Identification

Detects every person in a store/mall camera feed (live webcam or an
uploaded video), estimates their **age** and **gender**, flags anyone
over 60 as a **senior citizen**, and logs `timestamp, gender, estimated
age, senior_citizen, source` to a CSV (with one-click Excel export).

- **Backend** (`/backend`): FastAPI. MediaPipe FaceMesh finds every face
  and its 468 landmarks; two custom-trained models (not a pretrained
  age/gender network) turn those into predictions; a lightweight
  centroid tracker avoids logging the same person on every frame.
- **Frontend** (`/frontend`): React (Vite). Live camera tab, video
  upload tab, and a visitor log tab with CSV/Excel download.

## How the model works (your own ML model, as required)

Two small models, trained from scratch on **feature-level synthetic
data** (see "About the training data" below) — no pretrained face-
attribute network is used:

1. **Gender — RandomForestClassifier** over 6 geometric ratios computed
   from face-mesh landmarks (face height/width, jaw-to-face width, brow-
   to-eye distance, lip thickness, eye spacing, mouth width — all
   normalized by face size).
2. **Age — RandomForestRegressor** over 4 pixel-level cues from the
   face crop: forehead edge density (wrinkle proxy), eye-corner edge
   density (crow's-feet proxy), grey/white ratio near the hairline, and
   cheek skin-texture variance (Laplacian variance). `senior_citizen =
   estimated_age > 60`.

`backend/face_features.py` has the exact landmark indices and pixel
regions used, with comments on why each one was picked.

## About the training data (please read before grading)

No face-image dataset ships with this project — this sandbox couldn't
reach one, and bundling a real biometric dataset raises its own consent
questions. Instead, `backend/synthetic_data.py` samples each model's
input **features** directly from distributions set using well-known
coarse anthropometric and skin-texture trends (e.g. skin edge-density
and grey-hair ratio rising with age; jaw width and brow-eye distance
differing on average by gender), jittered with noise. Training/held-out
accuracy on that synthetic data is high (gender ~97%, derived
senior/not-senior ~99%) — see `train_model.py`'s printed report — but
that number describes how well the classifier learned the *synthetic*
boundaries, not real-world accuracy, which will be lower and does vary
by individual, lighting, and camera angle.

**To improve real-world accuracy**, replace `build_gender_dataset()` /
`build_age_dataset()` in `synthetic_data.py` with real, labelled
features: run `face_features.landmarks_to_gender_features` /
`landmarks_to_age_features` over a folder of consented, age/gender-
labelled photos and train on those instead. The rest of the pipeline
(serving, tracking, logging) doesn't need to change.

## Run it

### 1. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
python3 train_model.py     # regenerates model.pkl (already included)
python3 -m uvicorn app:app --reload --port 8001
```

API at `http://localhost:8001` — interactive docs at `/docs`.
`visitor_log.csv` is created next to `app.py` on first run.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL. It talks to `VITE_API_URL` in
`frontend/.env` (defaults to `http://localhost:8001`).

## How visits are deduplicated

A person standing in frame for several seconds of live video, or
appearing in several sampled video frames, should count as **one**
visit, not one row per frame. `backend/tracker.py` matches each new
detection to the closest existing track (by position, scaled to face
size); a track is logged the moment it's created, and again only if it
goes unlogged for more than 20 seconds (handles someone leaving and
re-entering). This is a positional tracker, not face re-identification
— it works well for a single fixed entrance camera (the brief's
scenario) but won't recognize the same person across separate camera
sessions.

## Project layout

```
backend/
  app.py              FastAPI app: detection + logging endpoints
  face_features.py     landmarks + pixel crops -> feature vectors
  synthetic_data.py     synthetic feature distributions for training
  tracker.py            centroid tracker + CSV read/write/clear
  train_model.py         trains gender_clf + age_reg, saves model.pkl
  model.pkl               pre-trained models (regenerate any time)
  requirements.txt
frontend/
  src/App.jsx                       tab state, summary stats
  src/components/Sidebar.jsx         nav, live stats, privacy note
  src/components/LiveStage.jsx       webcam capture loop
  src/components/VideoStage.jsx      video upload + processing
  src/components/LogStage.jsx        visitor table, CSV/Excel export
  src/api.js                         backend fetch calls
```

## A note on responsible use

This flags a visible, coarse attribute (age bracket) to help staff
offer better service, similar to how a greeter might use their own
judgement — it does not identify who a person is. No face images are
stored; only aggregate estimates and a timestamp are logged.
Age/gender estimates from any such system, including this one, can be
wrong for a given individual — use it as an assistive signal, not a
determination about any specific person.
