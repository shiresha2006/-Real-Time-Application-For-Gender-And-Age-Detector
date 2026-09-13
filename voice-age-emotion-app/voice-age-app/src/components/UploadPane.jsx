import { useRef, useState } from 'react'
import Waveform from './Waveform.jsx'

export default function UploadPane({ audioSrc, samples, fileName, scanning, onFileSelected }) {
  const inputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFiles = (files) => {
    const file = files?.[0]
    if (!file || !file.type.startsWith('audio/')) return
    onFileSelected(file)
  }

  return (
    <div
      className={`upload-pane ${dragActive ? 'is-dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files) }}
    >
      <div className="pane-label"><span>01</span> Voice note</div>

      {!audioSrc && (
        <button className="dropzone" onClick={() => inputRef.current?.click()}>
          <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3">
            <path d="M12 15a3 3 0 003-3V6a3 3 0 10-6 0v6a3 3 0 003 3z" strokeLinecap="round" />
            <path d="M19 11a7 7 0 01-14 0M12 18v3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <p className="dropzone-title">Drop a voice note here</p>
          <p className="dropzone-sub">or click to browse — MP3, WAV, M4A</p>
        </button>
      )}

      {audioSrc && (
        <div className="audio-frame">
          <Waveform samples={samples} scanning={scanning} />
          <div className="audio-meta">
            <span className="file-name">{fileName}</span>
            <button className="swap-btn" onClick={() => inputRef.current?.click()}>Replace</button>
          </div>
          <audio src={audioSrc} controls className="audio-player" />
        </div>
      )}

      <input ref={inputRef} type="file" accept="audio/*" hidden onChange={(e) => handleFiles(e.target.files)} />
    </div>
  )
}
