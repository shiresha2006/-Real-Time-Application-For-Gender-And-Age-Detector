import { useEffect, useRef, useState } from "react";
import { predictFrame } from "../api.js";
import ResultPanel from "./ResultPanel.jsx";

const CAPTURE_INTERVAL_MS = 800;

export default function LiveStage({ bypassHours, onMatched }) {
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
    } catch (e) {
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
        const res = await predictFrame(dataUrl, bypassHours);
        if (cancelled) return;
        setResult(res);
        if (res.detected && res.label && res.label !== "Uncertain") {
          onMatched(res.label);
        }
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, bypassHours]);

  const showAnnotated = running && result?.annotated_image;

  return (
    <div className="stage">
      <div className="stage-header">
        <h1>Live camera</h1>
        <p>Hold one hand steady in frame. A prediction refreshes roughly once a second.</p>
      </div>

      <div className="preview-panel">
        <video
          ref={videoRef}
          muted
          playsInline
          style={{ display: showAnnotated ? "none" : running ? "block" : "none" }}
        />
        {showAnnotated && <img src={result.annotated_image} alt="Live hand landmarks" />}
        {!running && (
          <div className="preview-empty">
            Start the camera to begin real-time detection. Video is processed
            frame by frame and never stored.
          </div>
        )}
      </div>

      <div className="controls-row">
        {!running ? (
          <button className="btn" onClick={start}>
            Start camera
          </button>
        ) : (
          <button className="btn secondary" onClick={stop}>
            Stop camera
          </button>
        )}
        <span className={`live-indicator ${running ? "on" : ""}`}>
          <span className="rec-dot" />
          {running ? (busy ? "Reading frame…" : "Live") : "Idle"}
        </span>
      </div>

      {error && <div className="error-banner">{error}</div>}
      <ResultPanel result={result} />
    </div>
  );
}
