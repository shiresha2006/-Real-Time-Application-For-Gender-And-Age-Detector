import { useState } from 'react'
import UploadPane from './components/UploadPane.jsx'
import ReadoutPanel from './components/ReadoutPanel.jsx'
import {
  decodeAudioFile,
  extractAcousticFeatures,
  classifyGender,
  estimateMaleAge,
  estimateEmotion,
  SENIOR_AGE_THRESHOLD,
} from './utils/voiceAnalysis.js'

export default function App() {
  const [audioSrc, setAudioSrc] = useState(null)
  const [fileName, setFileName] = useState('')
  const [samples, setSamples] = useState(null)
  const [decoded, setDecoded] = useState(null)
  const [status, setStatus] = useState('idle') // idle | analyzing | done | rejected | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileSelected = async (file) => {
    setStatus('idle')
    setResult(null)
    setError(null)
    setFileName(file.name)
    setAudioSrc(URL.createObjectURL(file))
    try {
      const audio = await decodeAudioFile(file)
      setDecoded(audio)
      setSamples(audio.samples)
    } catch (err) {
      console.error(err)
      setError('Could not decode this audio file.')
      setStatus('error')
    }
  }

  const runAnalysis = async () => {
    if (!decoded) return
    setStatus('analyzing')
    setError(null)
    // brief delay so the scan animation is perceptible on short clips
    await new Promise((r) => setTimeout(r, 450))
    try {
      const features = extractAcousticFeatures(decoded)
      if (!features.meanF0 || features.voicedFrameCount < 5) {
        setError('Not enough voiced speech detected in this clip. Try a clearer recording.')
        setStatus('error')
        return
      }

      const gender = classifyGender(features)
      if (gender === 'female') {
        setResult({ medianF0: Math.round(features.medianF0) })
        setStatus('rejected')
        return
      }

      const age = estimateMaleAge(features)
      const isSenior = age > SENIOR_AGE_THRESHOLD
      const emotion = isSenior ? estimateEmotion(features) : null

      setResult({
        medianF0: Math.round(features.medianF0),
        jitter: features.jitter.toFixed(3),
        age,
        isSenior,
        emotion,
      })
      setStatus('done')
    } catch (err) {
      console.error(err)
      setError('Something went wrong analyzing the audio.')
      setStatus('error')
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">Voice Age &amp; Emotion Studio</span>
        </div>
        <p className="brand-tagline">Pitch, jitter &amp; energy analysis for male voice notes</p>
      </header>

      <main className="studio-grid">
        <UploadPane
          audioSrc={audioSrc}
          samples={samples}
          fileName={fileName}
          scanning={status === 'analyzing'}
          onFileSelected={handleFileSelected}
        />

        <div className="middle-column">
          <div className="pane-label"><span>—</span> Run</div>
          <p className="field-hint">
            Male voices under 60 get an age reading. Over 60, the model also flags a
            senior category and reads emotion. Female voices are rejected per spec.
          </p>
          <button className="run-btn" disabled={!decoded || status === 'analyzing'} onClick={runAnalysis}>
            {status === 'analyzing' ? 'Reading…' : 'Run reading'}
          </button>

          <details className="note">
            <summary>How does this actually work?</summary>
            <p>
              No pretrained network or labelled age dataset ships with this project.
              Instead it runs a real signal-processing pipeline in your browser:
              autocorrelation pitch tracking, jitter (pitch-instability) and energy
              features, then rule-based classifiers grounded in known speech-science
              relationships between these features and age, sex and affect. Treat
              the numbers as a working demo, not clinical-grade estimation.
            </p>
          </details>
        </div>

        <ReadoutPanel status={status} result={result} error={error} />
      </main>

      <footer className="app-footer">
        All analysis runs locally in your browser via the Web Audio API. No audio is uploaded anywhere.
      </footer>
    </div>
  )
}
