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
      <div className="pane-label"><span>02</span> Reading</div>

      {status === 'idle' && (
        <div className="readout-empty">
          <p>Upload a voice note and run a reading.</p>
        </div>
      )}

      {status === 'analyzing' && (
        <div className="readout-empty">
          <div className="pulse-dot" />
          <p>Extracting pitch, jitter &amp; energy…</p>
        </div>
      )}

      {status === 'error' && (
        <div className="readout-empty readout-error">
          <p>{error || 'Could not analyze this clip.'}</p>
        </div>
      )}

      {status === 'rejected' && (
        <div className="readout-empty readout-reject">
          <div className="reject-icon">✕</div>
          <p className="reject-message">Upload male voice.</p>
          <p className="reject-sub">
            Detected a female voice profile (median pitch {result?.medianF0}&nbsp;Hz).
            This model is scoped to male voices only, per spec.
          </p>
        </div>
      )}

      {status === 'done' && result && (
        <div className="fields">
          <Field label="Voice profile" value="Male" sub={`Median pitch ${result.medianF0} Hz`} />
          <Field
            label="Estimated age"
            value={`${result.age} years`}
            sub={`Jitter index ${result.jitter}`}
          />
          {result.isSenior && (
            <>
              <Field label="Category" value="Senior citizen" sub="Age over 60" />
              <Field
                label="Emotion"
                value={result.emotion.label}
                sub={result.emotion.detail}
              />
            </>
          )}
        </div>
      )}
    </div>
  )
}
