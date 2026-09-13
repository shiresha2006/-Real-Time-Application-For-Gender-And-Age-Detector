export default function ResultPanel({ result }) {
  if (!result) return null;

  if (result.error === "closed") {
    return (
      <div className="closed-banner">
        <strong>Terminal closed.</strong> Detections run{" "}
        {result.open_time}–{result.close_time} server time. Enable demo mode
        in the sidebar to try it outside those hours.
      </div>
    );
  }

  if (result.error) {
    return <div className="error-banner">Couldn't read that input — try another file or angle.</div>;
  }

  if (!result.detected) {
    return (
      <div className="result-panel">
        <div className="result-headline">
          <span className="result-word none">No hand detected</span>
        </div>
        <p style={{ color: "var(--text-dim)", fontSize: 13, margin: 0 }}>
          Frame the hand clearly against a plain background, fingers visible.
        </p>
      </div>
    );
  }

  const wordClass = result.label === "Uncertain" ? "uncertain" : "";

  return (
    <div className="result-panel">
      <div className="result-headline">
        <span className={`result-word ${wordClass}`}>{result.label}</span>
        <span className="result-confidence">{Math.round(result.confidence * 100)}% confidence</span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${result.confidence * 100}%` }} />
      </div>

      {result.top3 && (
        <div className="top3-list">
          {result.top3.map((t) => (
            <div className="top3-row" key={t.label}>
              <span>{t.label}</span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${t.confidence * 100}%` }} />
              </span>
              <span>{Math.round(t.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
