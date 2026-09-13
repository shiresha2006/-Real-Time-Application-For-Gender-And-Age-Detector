"""
Senior Citizen Identification API.

Endpoints
---------
GET  /api/status                    -> service info
POST /api/detect/frame              -> base64 JPEG frame (live webcam), detects
                                        every face, logs new/re-entered visits
POST /api/detect/video              -> uploaded video file, samples frames,
                                        detects + logs
GET  /api/log                       -> current visitor log as JSON
GET  /api/log/csv                   -> download the CSV
GET  /api/log/excel                 -> download an .xlsx export
POST /api/log/clear                 -> wipe the log (demo reset)
"""
import base64
import tempfile
import os

import cv2
import joblib
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from face_features import (
    landmarks_to_gender_features, landmarks_to_age_features, face_bbox_from_landmarks,
)
from tracker import VisitorTracker, read_log_rows, clear_log, LOG_PATH, ensure_log_file

MAX_FACES = 8
SENIOR_AGE = 60

app = FastAPI(title="Senior Citizen Identification API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ensure_log_file()

bundle = joblib.load("model.pkl")
gender_clf = bundle["gender_clf"]
gender_le = bundle["gender_label_encoder"]
age_reg = bundle["age_reg"]

mp_face_mesh = mp.solutions.face_mesh
face_mesh_static = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=MAX_FACES, min_detection_confidence=0.5)
face_mesh_stream = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=MAX_FACES,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)

live_tracker = VisitorTracker()  # persists across live-feed frames for this server process


def _predict_face(bgr_image, face_landmarks, img_w, img_h):
    px = {i: (lm.x * img_w, lm.y * img_h) for i, lm in enumerate(face_landmarks.landmark)}

    gender_vec = landmarks_to_gender_features(px).reshape(1, -1)
    gender_probs = gender_clf.predict_proba(gender_vec)[0]
    gi = int(np.argmax(gender_probs))
    gender = gender_le.inverse_transform([gi])[0]
    gender_conf = float(gender_probs[gi])

    age_vec = landmarks_to_age_features(bgr_image, px).reshape(1, -1)
    age = float(age_reg.predict(age_vec)[0])

    bbox = face_bbox_from_landmarks(px, img_w, img_h)
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    face_w = bbox[2] - bbox[0]

    return {
        "bbox": bbox, "cx": cx, "cy": cy, "face_w": face_w,
        "gender": gender, "gender_confidence": gender_conf,
        "estimated_age": age, "senior_citizen": age > SENIOR_AGE,
    }


def _draw_annotations(bgr_image, detections):
    annotated = bgr_image.copy()
    for d in detections:
        x0, y0, x1, y1 = d["bbox"]
        color = (8, 160, 227) if d["senior_citizen"] else (140, 200, 79)  # BGR: amber-ish / green-ish
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        label = f"{d['gender']} - {int(round(d['estimated_age']))}y"
        if d["senior_citizen"]:
            label += " - SENIOR"
        cv2.putText(annotated, label, (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated


def _detect_all_faces(bgr_image, face_mesh_model):
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    result = face_mesh_model.process(rgb)
    detections = []
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            detections.append(_predict_face(bgr_image, face_landmarks, w, h))
    return detections


@app.get("/api/status")
def status():
    return {"known_words": None, "senior_age_threshold": SENIOR_AGE, "max_faces": MAX_FACES}


class FramePayload(BaseModel):
    image: str
    source: str = "live"


@app.post("/api/detect/frame")
def detect_frame(payload: FramePayload):
    header, _, b64data = payload.image.partition(",")
    img_bytes = base64.b64decode(b64data if b64data else payload.image)
    npimg = np.frombuffer(img_bytes, np.uint8)
    bgr_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if bgr_image is None:
        return {"error": "invalid_image"}

    detections = _detect_all_faces(bgr_image, face_mesh_stream)

    new_visits = 0
    for d in detections:
        _, logged = live_tracker.update(
            d["cx"], d["cy"], d["face_w"], d["gender"], d["gender_confidence"],
            d["estimated_age"], payload.source,
        )
        d["new_visit_logged"] = logged
        new_visits += int(logged)
    live_tracker.gc()

    annotated = _draw_annotations(bgr_image, detections)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    annotated_b64 = base64.b64encode(buf).decode("utf-8") if ok else None

    for d in detections:
        d.pop("bbox", None)

    return {
        "face_count": len(detections),
        "detections": detections,
        "new_visits_logged": new_visits,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
    }


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    sample_every_n = max(1, int(round(fps)))  # ~1 sample per second of video
    max_samples = 90  # cap processing for demo-sized videos

    tracker = VisitorTracker()
    frame_idx = 0
    samples_taken = 0
    total_face_hits = 0

    while samples_taken < max_samples:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % sample_every_n == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                detections = _detect_all_faces(frame, face_mesh_static)
                total_face_hits += len(detections)
                for d in detections:
                    tracker.update(
                        d["cx"], d["cy"], d["face_w"], d["gender"], d["gender_confidence"],
                        d["estimated_age"], f"video:{file.filename}",
                    )
                tracker.gc()
            samples_taken += 1
        frame_idx += 1

    cap.release()
    os.unlink(tmp_path)

    return {
        "frames_sampled": samples_taken,
        "face_detections_total": total_face_hits,
        "unique_visitors_logged": tracker._next_id - 1,
        "source": file.filename,
    }


@app.get("/api/log")
def get_log():
    return {"rows": read_log_rows()}


@app.get("/api/log/csv")
def download_csv():
    ensure_log_file()
    return FileResponse(LOG_PATH, media_type="text/csv", filename="visitor_log.csv")


@app.get("/api/log/excel")
def download_excel():
    import pandas as pd
    rows = read_log_rows()
    df = pd.DataFrame(rows)
    xlsx_path = os.path.join(tempfile.gettempdir(), "visitor_log.xlsx")
    df.to_excel(xlsx_path, index=False)
    return FileResponse(
        xlsx_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="visitor_log.xlsx",
    )


@app.post("/api/log/clear")
def clear_log_endpoint():
    clear_log()
    live_tracker.tracks = []
    live_tracker._next_id = 1
    return {"cleared": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
