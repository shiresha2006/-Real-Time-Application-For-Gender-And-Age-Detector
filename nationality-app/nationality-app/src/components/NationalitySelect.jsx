const OPTIONS = [
  { id: 'indian', label: 'India', reads: 'Emotion · Age · Dress colour' },
  { id: 'us', label: 'United States', reads: 'Emotion · Age' },
  { id: 'african', label: 'African nations', reads: 'Emotion · Dress colour' },
  { id: 'other', label: 'Other', reads: 'Emotion only' },
]

export default function NationalitySelect({ value, onChange }) {
  return (
    <div className="nationality-select">
      <div className="pane-label">
        <span>02</span> Declared origin
      </div>
      <p className="field-hint">
        The reading fields below change with this selection — see why on the right.
      </p>
      <div className="chip-row">
        {OPTIONS.map((opt) => (
          <button
            key={opt.id}
            className={`chip ${value === opt.id ? 'chip-active' : ''}`}
            onClick={() => onChange(opt.id)}
          >
            <span className="chip-label">{opt.label}</span>
            <span className="chip-reads">{opt.reads}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
