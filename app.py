"""
Real-Time Age & Gender Detector -- Streamlit web app.

Unlike a plain OpenCV script, this runs as a deployable web app: browser
webcam access via streamlit-webrtc (works locally and on Streamlit
Community Cloud), a live analytics dashboard (gender ratio, age-group
histogram, session log with CSV export), and a fairness/uncertainty
panel that reports how much of the session's output was low-confidence.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from src.pipeline import RealtimePipeline
from src.fairness_audit import DISCLOSURE_TEXT, session_fairness_report

st.set_page_config(page_title="Real-Time Age & Gender Detector", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RealtimePipeline()

pipeline = st.session_state.pipeline

st.title("Real-Time Age & Gender Detector")
st.caption(
    "Multi-face tracking · temporal smoothing · scheduled inference for real-time "
    "performance · live analytics · fairness & uncertainty reporting."
)

left, right = st.columns([2, 1])

with left:
    st.subheader("Live feed")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated, _ = pipeline.process_frame(img)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="age-gender-detector",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.info(
        "If the camera doesn't start: click **Start** above, and allow camera "
        "permission in the browser. On Streamlit Cloud this uses WebRTC so it "
        "works without installing anything locally."
    )

with right:
    st.subheader("Live analytics")
    summary = pipeline.session_log.summary()

    c1, c2 = st.columns(2)
    c1.metric("Faces seen (unique)", summary.get("unique_faces", 0))
    c2.metric("Predictions logged", summary.get("total_events", 0))

    gender_counts = summary.get("gender_counts", {})
    if gender_counts:
        st.write("**Gender distribution**")
        st.bar_chart(pd.Series(gender_counts, name="count"))

    age_groups = summary.get("age_group_counts", {})
    if age_groups:
        st.write("**Age-group distribution**")
        st.bar_chart(pd.Series(age_groups, name="count"))

    st.write("**Fairness & uncertainty**")
    st.caption(session_fairness_report(summary))

    if pipeline.session_log.entries:
        csv_bytes = pipeline.session_log.to_csv_bytes()
        st.download_button(
            "Download session log (CSV)",
            data=csv_bytes,
            file_name="session_log.csv",
            mime="text/csv",
        )

    if st.button("Reset session"):
        st.session_state.pipeline = RealtimePipeline()
        st.rerun()

with st.expander("Model limitations & fairness notes (read before demoing this)"):
    st.markdown(DISCLOSURE_TEXT)

with st.expander("How this differs from a standard tutorial project"):
    st.markdown(
        """
- **Multi-face temporal tracking** (`src/tracker.py`) assigns a stable ID
  to each face so predictions don't flicker frame to frame.
- **Scheduled inference** (`src/pipeline.py`) runs the (comparatively
  expensive) age/gender model every N frames per tracked face instead of
  every frame for every face -- keeps the app real-time with multiple
  faces on screen.
- **Confidence-weighted temporal smoothing** (`src/smoothing.py`)
  averages age and majority-votes gender over a rolling window, weighted
  by per-prediction confidence.
- **Live analytics dashboard** with CSV export, not just a bounding box.
- **Fairness & uncertainty reporting** (`src/fairness_audit.py`) --
  predictions are shown as estimates with a visible confidence signal,
  and the session reports what fraction of output was low-confidence,
  with an explicit note on documented bias risk in face-analysis models.
        """
    )
