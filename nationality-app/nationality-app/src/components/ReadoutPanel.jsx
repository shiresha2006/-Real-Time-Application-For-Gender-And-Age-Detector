const EMOTION_GLYPH = {
  Happy: '◠',
  Sad: '◡',
  Angry: '▲',
  Fearful: '◇',
  Disgusted: '✕',
  Surprised: '◎',
  Neutral: '—',
}

function Field({ label, value, sub }) {
  return (
    <div className="field">
      <div className="field-label">{label}</div>
      <div className="field-value">{value}</div>
      {sub && <div className="field-sub">{sub}</div>}
    </div>
  )
}

export default function ReadoutPanel({ status, result, error }) {
  return (
    <div className="readout-pane">
      <div className="pane-label">
        <span>03</span> Reading
      </div>

      {status === 'idle' && (
        <div className="readout-empty">
          <p>Upload a portrait and pick an origin to run a reading.</p>
        </div>
      )}

      {status === 'loading-models' && (
        <div className="readout-empty">
          <div className="pulse-dot" />
          <p>Calibrating detection models…</p>
        </div>
      )}

      {status === 'analyzing' && (
        <div className="readout-empty">
          <div className="pulse-dot" />
          <p>Scanning specimen…</p>
        </div>
      )}

      {status === 'error' && (
        <div className="readout-empty readout-error">
          <p>{error || 'No face could be detected in this image.'}</p>
        </div>
      )}

      {status === 'done' && result && result.faceFound && (
        <div className="fields">
          <Field
            label="Emotion"
            value={
              <>
                <span className="emotion-glyph">{EMOTION_GLYPH[result.emotion.label] || '—'}</span>
                {result.emotion.label}
              </>
            }
            sub={`${Math.round(result.emotion.confidence * 100)}% confidence`}
          />

          {result.age !== null && (
            <Field label="Estimated age" value={`${result.age} years`} />
          )}

          {result.dressColor && (
            <Field
              label="Dress colour"
              value={
                <span className="color-value">
                  <span
                    className="color-swatch"
                    style={{ background: result.dressColor.hex }}
                  />
                  {result.dressColor.name}
                </span>
              }
              sub={result.dressColor.hex.toUpperCase()}
            />
          )}

          {result.nationality === 'other' && (
            <Field label="Declared origin" value="Other (unspecified)" />
          )}
        </div>
      )}

      {status === 'done' && result && !result.faceFound && (
        <div className="readout-empty readout-error">
          <p>No face detected. Try a clearer, front-facing photo.</p>
        </div>
      )}
    </div>
  )
}
