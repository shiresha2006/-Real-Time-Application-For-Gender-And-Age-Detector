import { useRef, useState } from "react";
import { predictImageFile } from "../api.js";
import ResultPanel from "./ResultPanel.jsx";

export default function UploadStage({ bypassHours, onMatched }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef(null);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
  }

  async function runPrediction() {
    if (!file) return;
    setLoading(true);
    try {
      const res = await predictImageFile(file, bypassHours);
      setResult(res);
      if (res.detected && res.label && res.label !== "Uncertain") {
        onMatched(res.label);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="stage">
      <div className="stage-header">
        <h1>Upload image</h1>
        <p>Upload a photo of a single hand making one of the known signs below.</p>
      </div>

      <div className="preview-panel">
        {result?.annotated_image ? (
          <img src={result.annotated_image} alt="Detected hand landmarks" />
        ) : previewUrl ? (
          <img src={previewUrl} alt="Selected upload" />
        ) : (
          <button className="drop-zone" onClick={() => inputRef.current?.click()}>
            <span>Click to choose an image</span>
            <span style={{ fontSize: 12 }}>JPG or PNG, one hand, plain background</span>
          </button>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      <div className="controls-row">
        <button className="btn secondary" onClick={() => inputRef.current?.click()}>
          Choose image
        </button>
        <button className="btn" disabled={!file || loading} onClick={runPrediction}>
          {loading ? "Reading hand…" : "Detect sign"}
        </button>
      </div>

      <ResultPanel result={result} />
    </div>
  );
}
