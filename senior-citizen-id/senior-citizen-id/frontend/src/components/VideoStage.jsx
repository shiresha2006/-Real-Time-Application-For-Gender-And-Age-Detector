import { useRef, useState } from "react";
import { detectVideo } from "../api.js";

export default function VideoStage({ onLogged }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  async function run() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await detectVideo(file);
      setResult(res);
      if (res.unique_visitors_logged) onLogged(res.unique_visitors_logged);
    } catch {
      setError("Couldn't process that video — try a shorter clip or a different format.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="stage">
      <div className="stage-header">
        <h1>Upload video</h1>
        <p>Processes a recorded clip, sampling roughly one frame per second (up to 90 samples) and logging each new visitor once.</p>
      </div>

      <div className="preview-panel">
        {previewUrl ? (
          <video src={previewUrl} controls />
        ) : (
          <button className="drop-zone" onClick={() => inputRef.current?.click()}>
            <span>Click to choose a video</span>
            <span style={{ fontSize: 12 }}>MP4 or MOV, entrance-facing footage</span>
          </button>
        )}
        <input ref={inputRef} type="file" accept="video/*" hidden onChange={(e) => handleFile(e.target.files?.[0])} />
      </div>

      <div className="controls-row">
        <button className="btn secondary" onClick={() => inputRef.current?.click()}>Choose video</button>
        <button className="btn" disabled={!file || loading} onClick={run}>
          {loading ? "Processing…" : "Process video"}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && !error && (
        <div className="result-card">
          <div className="row"><span className="label">Frames sampled</span><span className="value">{result.frames_sampled}</span></div>
          <div className="row"><span className="label">Face detections (total)</span><span className="value">{result.face_detections_total}</span></div>
          <div className="row"><span className="label">Unique visitors logged</span><span className="value">{result.unique_visitors_logged}</span></div>
          <div className="row"><span className="label">Source</span><span className="value">{result.source}</span></div>
        </div>
      )}
    </div>
  );
}
