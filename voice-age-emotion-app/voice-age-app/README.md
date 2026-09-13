# Voice Age & Emotion Studio

A React app that analyzes an uploaded voice note: rejects female voices with the
required message, estimates age for male voices, and — for males over 60 — adds a
senior-citizen flag plus an emotion reading.

## How the "model" works

No pretrained network or labelled voice-age dataset is bundled here (there isn't a
usable one readily available at this scope). Instead this ships a real, from-scratch
signal-processing pipeline that runs entirely in the browser via the Web Audio API:

1. **Pitch tracking** — time-domain autocorrelation over 1024-sample frames extracts
   fundamental frequency (F0) across the clip, with a rank-based energy floor to skip
   silence/noise.
2. **Gender classification** — median F0 compared against the standard ~165 Hz
   male/female boundary used in speech science.
3. **Age estimation (male only)** — a hand-set linear combination of median F0 (lower
   F0 within the male range skews older) and jitter, the frame-to-frame pitch
   perturbation that's well documented to increase with vocal aging.
4. **Emotion (senior branch only)** — pitch-variability and energy-variability ratios
   mapped to a small rule set (Calm, Agitated/Stressed, Sad/Subdued, Neutral).

This is presented honestly in the UI as an approximation grounded in real acoustic
features, not a clinical-grade or deep-learning estimator — see the in-app "How does
this actually work?" note.

## Running it

```bash
npm install
npm run dev      # local dev server
npm run build    # production build → dist/
```

No external services or model downloads are required — everything runs client-side.

## Project structure

```
src/
  App.jsx                     upload → analysis → branching flow
  components/
    UploadPane.jsx             drag/drop audio upload, playback, waveform
    Waveform.jsx                canvas waveform renderer
    ReadoutPanel.jsx            results / rejection display
  utils/
    voiceAnalysis.js            pitch/jitter/energy extraction + gender/age/emotion rules
  index.css                    styling
```

## Verified

`npm install && npm run build` completes cleanly (35 modules, no errors), and the
core gender/age/emotion functions were sanity-checked against synthetic feature
values (young low-jitter voice → ~29, higher-jitter lower-F0 voice → ~58,
high-F0 voice → classified female) before packaging.
