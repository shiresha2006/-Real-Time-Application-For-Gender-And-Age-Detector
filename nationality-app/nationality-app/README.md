# Face Analysis Studio

A React app that analyzes a portrait photo for **facial emotion**, **estimated age**,
and **dominant clothing colour** — with which fields are shown controlled by a
user-selected origin category (India / United States / African nations / Other),
matching the original assignment's branching spec.

## Why origin is selected, not predicted

The original brief asked for nationality to be *predicted from the photo*. That's not
a valid computer-vision task: nationality isn't a visual property of a face. A model
trained to guess it would really be learning a mapping from skin tone and facial
features to ethnicity, then treating people differently downstream — which is
biometric ethnic profiling, not a legitimate classifier. This build keeps every other
requirement (differentiated outputs per category, GUI with preview, results section)
and replaces only that one step with an explicit user selection, so the branching
logic is visible and honest rather than laundered through a fake prediction.

## What's real ML here

- **Face detection** — TinyFaceDetector (face-api.js / TensorFlow.js)
- **Emotion recognition** — 7-class expression model (happy, sad, angry, fearful,
  disgusted, surprised, neutral), run on the detected face
- **Age estimation** — regression model bundled with face-api.js
- **Dress colour** — a lightweight heuristic (not a deep model): it samples the
  average pixel colour of the region just below the detected face box and matches
  it to the nearest named colour. This is called out honestly in the UI/code rather
  than dressed up as more sophisticated than it is.

All inference runs client-side in the browser via TensorFlow.js. No image ever
leaves the user's device.

## Running it

```bash
npm install
npm run dev      # local dev server
npm run build    # production build → dist/
```

The face-api.js model weights are fetched at runtime from a public CDN mirror the
first time the app loads (see `MODEL_URL` in `src/utils/analysis.js`), so an internet
connection is required on first use; after that the browser caches them.

## Project structure

```
src/
  App.jsx                     top-level state + analysis flow
  components/
    UploadPane.jsx             drag/drop image upload + live preview + scan animation
    NationalitySelect.jsx      origin selector (drives which fields appear)
    ReadoutPanel.jsx           results display
  utils/
    analysis.js                model loading, detection, emotion/age, dress-colour sampling
  index.css                   styling
```

## Verified

`npm install && npm run build` was run during development and completes cleanly
(211 modules, no errors) — see build output in project history.
