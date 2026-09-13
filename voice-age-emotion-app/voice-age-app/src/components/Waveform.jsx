import { useEffect, useRef } from 'react'

export default function Waveform({ samples, scanning }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !samples) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)

    const step = Math.ceil(samples.length / w)
    const mid = h / 2

    ctx.strokeStyle = '#5cc2b0'
    ctx.lineWidth = 1.4
    ctx.beginPath()
    for (let x = 0; x < w; x++) {
      let min = 1, max = -1
      for (let i = 0; i < step; i++) {
        const idx = x * step + i
        if (idx >= samples.length) break
        const v = samples[idx]
        if (v < min) min = v
        if (v > max) max = v
      }
      ctx.moveTo(x, mid + min * mid * 0.9)
      ctx.lineTo(x, mid + max * mid * 0.9)
    }
    ctx.stroke()
  }, [samples])

  return (
    <div className="waveform-wrap">
      <canvas ref={canvasRef} className="waveform-canvas" />
      {scanning && <div className="waveform-scan" />}
    </div>
  )
}
