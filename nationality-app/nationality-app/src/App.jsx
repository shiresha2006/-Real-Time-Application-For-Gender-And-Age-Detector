import { useEffect, useRef, useState } from 'react'
import UploadPane from './components/UploadPane.jsx'
import NationalitySelect from './components/NationalitySelect.jsx'
import ReadoutPanel from './components/ReadoutPanel.jsx'
import { analyzeImage, loadModels } from './utils/analysis.js'

export default function App() {
  const [imageSrc, setImageSrc] = useState(null)
  const [nationality, setNationality] = useState('indian')
  const [status, setStatus] = useState('idle') // idle | loading-models | analyzing | done | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const imgRef = useRef(null)

  useEffect(() => {
    // Warm the models in the background as soon as the app opens.
    loadModels().catch(() => {})
  }, [])

  const runAnalysis = async () => {
    if (!imageSrc || !imgRef.current) return
    setStatus('analyzing')
    setError(null)
    try {
      const img = imgRef.current
      if (!img.complete) {
        await new Promise((res) => (img.onload = res))
      }
      const res = await analyzeImage(img, nationality)
      setResult(res)
      setStatus('done')
    } catch (err) {
      console.error(err)
      setError('Something went wrong reading the model. Try a different image.')
      setStatus('error')
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">Face Analysis Studio</span>
        </div>
        <p className="brand-tagline">A vision instrument for emotion, age &amp; dress-colour readings</p>
      </header>

      <main className="studio-grid">
        <UploadPane
          imageSrc={imageSrc}
          onImageSelected={(src) => {
            setImageSrc(src)
            setStatus('idle')
            setResult(null)
          }}
          scanning={status === 'analyzing'}
          imgRef={imgRef}
        />

        <div className="middle-column">
          <NationalitySelect value={nationality} onChange={setNationality} />

          <button
            className="run-btn"
            disabled={!imageSrc || status === 'analyzing'}
            onClick={runAnalysis}
          >
            {status === 'analyzing' ? 'Reading…' : 'Run reading'}
          </button>

          <details className="note">
            <summary>Why does origin come from a dropdown, not the photo?</summary>
            <p>
              Nationality isn't a visual property of a face, so this tool doesn't try to infer
              it from pixels. You tell it the category; the model only measures what a
              vision model can actually measure — expression, approximate age, and the
              dominant colour of clothing in frame.
            </p>
          </details>
        </div>

        <ReadoutPanel status={status} result={result} error={error} />
      </main>

      <footer className="app-footer">
        Detection runs entirely in your browser via face-api.js. No images leave your device.
      </footer>
    </div>
  )
}
