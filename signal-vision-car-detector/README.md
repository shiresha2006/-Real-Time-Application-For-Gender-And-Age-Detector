# Signal Vision — Car Colour & People Detection at Traffic Signals

## What it does
- Detects cars in an uploaded traffic image (Haar cascade).
- Classifies each car's dominant colour (HSV-based).
- Draws a **red** box around **blue** cars, a **blue** box around all other cars.
- Detects people (pedestrians) and draws a **green** box around each.
- Shows live counts of cars and people, with an image upload/preview web UI.

## Run
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 in a browser, upload/drop a traffic photo, click "Run Detection".

## Structure
- `app.py` — Flask server + `/api/detect` endpoint
- `detector.py` — Haar cascade car/person detection + HSV colour classification
- `models/` — pretrained Haar cascades (cars.xml, haarcascade_fullbody.xml)
- `templates/`, `static/` — web UI (HTML/CSS/JS)

## Notes
- Detection accuracy depends on image quality/angle — Haar cascades work best on clear, front/side-on daylight shots.
- Colour classifier samples the central region of each car box to avoid background bleed, and separately handles black/white/gray/silver via low-saturation brightness thresholds.
- To swap in a deep-learning detector (e.g. YOLO) later, replace the `detect()` function in `detector.py` and keep the same return shape.
