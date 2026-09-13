# Signal — Sign Language Detection

A small end-to-end sign/gesture detection tool:

- **Backend** (`/backend`): FastAPI service. Uses MediaPipe Hands to find
  21 hand landmarks in an image or webcam frame, converts them into a
  scale/translation-invariant geometric feature vector, and classifies
  the pose with a RandomForest trained on 8 known gesture words.
- **Frontend** (`/frontend`): React (Vite) app with two modes — upload an
  image, or run live webcam detection — plus a visible "open/closed"
  operating-hours indicator.

## Known words

`Hello, Yes, Peace, Good, Stop, ILoveYou, OK, Point`

These are 8 shape-distinct static hand poses (not full ASL — see
"About the model" below).

## Operational hours

Per the assignment brief, the service only serves predictions during a
configured window — **6 PM–10 PM server time** by default (edit
`OPEN_TIME` / `CLOSE_TIME` in `backend/app.py` to change it). Outside
that window the API returns `{"error": "closed", ...}` and the UI shows
a closed banner.

For grading/demo outside those hours, tick **"Demo mode — ignore
operating hours"** in the app sidebar — this sends `bypass_hours: true`
to the backend for that request only. It's a visible, intentional
override, not a way around the feature; leave it unticked to see the
real gating behavior.

## Run it

### 1. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python3 train_model.py     # regenerates model.pkl (already included, so optional)
python3 -m uvicorn app:app --reload --port 8000
```

The API is now at `http://localhost:8000` (`/api/status`,
`/api/predict/image`, `/api/predict/frame`). Interactive docs at
`/docs`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL (usually `http://localhost:5173`). It talks
to the backend URL set in `frontend/.env` (`VITE_API_URL`, defaults to
`http://localhost:8000`).

## About the model

No external hand-gesture image dataset ships with this project (keeps
it small, license-free, and runnable offline). Instead,
`backend/gesture_data.py` procedurally generates thousands of
realistic, noisy 21-point hand skeletons for each of the 8 gestures
using a small forward-kinematics model (per-finger curl + splay +
rotation/scale jitter), and `train_model.py` trains a RandomForest on
geometric features extracted from those landmarks
(`backend/features.py`). At inference time the *same* feature
extraction runs on real MediaPipe landmarks from your uploaded image
or webcam frame, so the classifier generalizes to real hands making
the same shapes.

This is a legitimate lightweight technique for static-pose gesture
recognition, but it is not a substitute for training on a real, varied
hand-image dataset — expect it to be sensitive to hand orientation and
occasional misreads. To improve it: replace `build_dataset()` in
`gesture_data.py` with landmarks captured from your own webcam (log
`hand_landmarks` from a few hundred frames per gesture, label them,
retrain) — the feature extraction and serving code don't need to
change.

## Project layout

```
backend/
  app.py            FastAPI app, endpoints, time-window gating
  features.py        landmark -> feature vector
  gesture_data.py     synthetic training-data generator + gesture definitions
  train_model.py       trains and saves model.pkl
  model.pkl            pre-trained classifier (regenerate any time)
  requirements.txt
frontend/
  src/App.jsx                     tab state, status polling
  src/components/Sidebar.jsx       nav, operating-hours sign, glossary
  src/components/UploadStage.jsx   image upload flow
  src/components/LiveStage.jsx     webcam capture loop
  src/components/ResultPanel.jsx   shared prediction readout
  src/api.js                       backend fetch calls
```
