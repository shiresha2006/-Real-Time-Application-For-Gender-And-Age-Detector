import { useEffect, useRef, useState } from "react";
import { detectFrame } from "../api.js";

const CAPTURE_INTERVAL_MS = 900;

export default function LiveStage({ onLogged }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setRunning(true);
    } catch {
      setError("Camera access was denied or is unavailable.");
    }
  }

  function stop() {
    setRunning(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    clearTimeout(timerRef.current);
  }

  useEffect(() => () => stop(), []);

  useEffect(() => {
    if (!running) return;
    let cancelled = false;

    async function tick() {
      if (cancelled || !videoRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);

      setBusy(true);
      try {
        const res = await detectFrame(dataUrl, "live");
        if (cancelled) return;
        setResult(res);
        const newlyLogged = (res.detections || []).filter((d) => d.new_visit_logged);
        if (newlyLogged.length) onLogged(newlyLogged.length);
      } catch {
        if (!cancelled) setError("Lost connection to the detection server.");
      } finally {
        if (!cancelled) setBusy(false);
      }

      if (!cancelled) timerRef.current = setTimeout(tick, CAPTURE_INTERVAL_MS);
    }

    tick();
    return () => {
      cancelled = true;
      clearTimeout(timerRef.current);
    };
  }, [running]);

  const showAnnotated = running && result?.annotated_image;
  const detections = result?.detections || [];

  return (
    <div className="stage">
      <div className="stage-header">
        <h1>Live camera</h1>
        <p>Point the camera at the entrance. Every face in frame is estimated roughly once a second.</p>
      </div>

      <div className="preview-panel">
        <video ref={videoRef} muted playsInline style={{ display: showAnnotated ? "none" : running ? "block" : "none" }} />
        {showAnnotated && <img src={result.annotated_image} alt="Live detections" />}
        {!running && (
          <div className="preview-empty">
            Start the camera to begin. Frames are processed live and are not saved anywhere.
          </div>
        )}
      </div>

      <div className="controls-row">
        {!running ? (
          <button className="btn" onClick={start}>Start camera</button>
        ) : (
          <button className="btn secondary" onClick={stop}>Stop camera</button>
        )}
        <span className={`live-indicator ${running ? "on" : ""}`}>
          <span className="rec-dot" />
          {running ? (busy ? "Reading frame…" : "Live") : "Idle"}
        </span>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {detections.length > 0 && (
        <div className="face-list">
          {detections.map((d, i) => (
            <div className="face-row" key={i}>
              <span>{d.gender} · {Math.round(d.estimated_age)}y</span>
              <span className={`badge ${d.senior_citizen ? "senior" : ""}`}>
                {d.senior_citizen ? "Senior citizen" : "Under 60"}
              </span>
              <span className={`badge ${d.new_visit_logged ? "logged" : ""}`}>
                {d.new_visit_logged ? "Logged" : "Tracking"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
