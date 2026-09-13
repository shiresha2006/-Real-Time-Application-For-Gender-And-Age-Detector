# Real-Time Age & Gender Detector

A real-time, multi-face age & gender detection system — built to go beyond
the standard tutorial version of this project (single-frame OpenCV + 2015
Caffe model, no tracking, no smoothing, no discussion of reliability).

**Live demo:** Streamlit web app with browser webcam access (deployable to
Streamlit Community Cloud). **Local demo:** OpenCV script for webcam or
video file, used to record `demo/demo_video.mp4`.

## What makes this different from the typical version of this project

| | Typical tutorial clone | This project |
|---|---|---|
| Face detection | Haar cascade only | OpenCV DNN (SSD) detector, robust to pose/lighting |
| Age/gender model | 2015 Levi & Hassner Caffe net | DeepFace (VGG-Face based), newer & better-calibrated |
| Per-face identity | None — every frame independent | Centroid tracker assigns a stable ID per face |
| Prediction stability | Flickers frame to frame | Confidence-weighted temporal smoothing (EMA + majority vote) |
| Performance at scale | Full inference every face, every frame | Heavy inference scheduled every N frames per track |
| Output | Bounding box + label | Live analytics dashboard, CSV session export |
| Reliability framing | Presented as fact | Uncertainty flagging + fairness/bias disclosure |
| Deployment | Local script only | Deployable browser web app (streamlit-webrtc) |

## Architecture

```
Frame -> detect_faces() -> CentroidTracker -> [per track, every N frames]
      -> infer_age_gender() (DeepFace) -> TrackSmoother -> fairness flag
      -> SessionLog -> Streamlit dashboard / annotated video frame
```

- `src/detector.py` — face detection (OpenCV DNN) + DeepFace age/gender inference
- `src/tracker.py` — lightweight centroid-distance multi-object tracker
- `src/smoothing.py` — rolling, confidence-weighted temporal smoothing per track
- `src/pipeline.py` — wires detection → tracking → scheduled inference → smoothing → logging
- `src/analytics.py` — session logging, age-bucketing, CSV export
- `src/fairness_audit.py` — per-prediction uncertainty flags + session fairness report
- `app.py` — Streamlit web app (live feed + analytics + fairness panel)
- `run_webcam.py` — local OpenCV entry point (webcam or video file), for recording the demo

## Setup

```bash
git clone <this-repo-url>
cd realtime-age-gender-detector
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads DeepFace's age/gender model weights automatically
(cached under `~/.deepface`).

### Run the web app (recommended for the demo)
```bash
streamlit run app.py
```
Click **Start** to enable your browser's webcam. The right panel shows
live gender/age-group distributions, a fairness/uncertainty note, and a
CSV download of the session log.

### Run locally with OpenCV (for recording demo video)
```bash
python run_webcam.py --save demo/demo_video.mp4
# or on a video file instead of a webcam:
python run_webcam.py --source path/to/input.mp4 --save demo/demo_video.mp4
```
Press `q` to quit, `s` to export the session log at any time.

### Run tests
```bash
python -m pytest tests/ -v
```
Covers the tracker, temporal smoothing, analytics, and fairness-flagging
logic (no webcam or model download required).

## Model limitations & fairness notes

Age and gender outputs are statistical estimates from a model trained on
public face datasets — not ground truth, and "gender" here means a binary
classification of outward appearance, not a measurement of gender
identity. Published audits of face-analysis systems (Buolamwini & Gebru,
*Gender Shades*, FAT* 2018) found materially higher error rates for
darker-skinned individuals and for women relative to lighter-skinned
individuals and men; this project has not been independently bias-audited
and inherits that risk from its underlying public model. The app surfaces
this directly: every prediction carries a live confidence/agreement score,
low-confidence predictions are flagged in the UI rather than presented as
certain, and the session summary reports what fraction of output was
low-confidence. Full disclosure text is also shown in-app and in
`src/fairness_audit.py`.

## Tech stack

Python, OpenCV (DNN face detector), DeepFace, Streamlit, streamlit-webrtc.

## License

MIT — see `LICENSE`.
