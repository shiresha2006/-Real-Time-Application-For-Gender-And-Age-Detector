import { useCallback, useRef, useState } from 'react'

export default function UploadPane({ imageSrc, onImageSelected, scanning, imgRef }) {
  const inputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0]
      if (!file || !file.type.startsWith('image/')) return
      const reader = new FileReader()
      reader.onload = (e) => onImageSelected(e.target.result)
      reader.readAsDataURL(file)
    },
    [onImageSelected]
  )

  return (
    <div
      className={`upload-pane ${dragActive ? 'is-dragging' : ''} ${imageSrc ? 'has-image' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragActive(true)
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragActive(false)
        handleFiles(e.dataTransfer.files)
      }}
    >
      <div className="pane-label">
        <span>01</span> Specimen
      </div>

      {!imageSrc && (
        <button className="dropzone" onClick={() => inputRef.current?.click()}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3">
            <path d="M12 16V4M12 4l-4 4M12 4l4 4" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <p className="dropzone-title">Drop a portrait here</p>
          <p className="dropzone-sub">or click to browse — JPG, PNG</p>
        </button>
      )}

      {imageSrc && (
        <div className="preview-frame">
          <img ref={imgRef} src={imageSrc} alt="Uploaded specimen" crossOrigin="anonymous" />
          {scanning && <div className="scan-line" />}
          <button className="swap-btn" onClick={() => inputRef.current?.click()}>
            Replace image
          </button>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  )
}
