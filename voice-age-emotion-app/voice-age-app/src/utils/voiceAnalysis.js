// A from-scratch acoustic analysis pipeline. There is no pretrained
// network here and no labelled voice-age dataset bundled with this
// project — instead this implements the classic signal-processing
// features speech science actually uses for these tasks (fundamental
// frequency, jitter, energy) and combines them with simple, documented
// rules/regressions. This is an honest "own model" for a task where a
// deep model would need a large labelled corpus this project doesn't
// ship with.

const FRAME_SIZE = 1024
const HOP_SIZE = 512
const MIN_F0 = 70 // Hz
const MAX_F0 = 400 // Hz
const VOICED_ENERGY_PERCENTILE = 0.35 // frames below this energy rank are treated as silence

export async function decodeAudioFile(file) {
  const arrayBuffer = await file.arrayBuffer()
  const AudioCtx = window.AudioContext || window.webkitAudioContext
  const ctx = new AudioCtx()
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer)
  const channel = audioBuffer.getChannelData(0)
  const sampleRate = audioBuffer.sampleRate
  ctx.close()
  return { samples: channel, sampleRate, duration: audioBuffer.duration }
}

// Time-domain autocorrelation pitch detector for a single frame.
function autocorrelationF0(frame, sampleRate) {
  const n = frame.length
  const minLag = Math.floor(sampleRate / MAX_F0)
  const maxLag = Math.floor(sampleRate / MIN_F0)

  let bestLag = -1
  let bestCorr = 0

  for (let lag = minLag; lag <= maxLag && lag < n; lag++) {
    let sum = 0
    for (let i = 0; i < n - lag; i++) {
      sum += frame[i] * frame[i + lag]
    }
    if (sum > bestCorr) {
      bestCorr = sum
      bestLag = lag
    }
  }

  if (bestLag <= 0) return null
  return sampleRate / bestLag
}

function frameEnergy(frame) {
  let sum = 0
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i]
  return Math.sqrt(sum / frame.length)
}

function median(arr) {
  if (!arr.length) return 0
  const s = [...arr].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0
}

function stdDev(arr) {
  if (arr.length < 2) return 0
  const m = mean(arr)
  return Math.sqrt(mean(arr.map((v) => (v - m) ** 2)))
}

// Extracts per-frame F0 + energy across the whole clip, keeping only
// "voiced" frames (energy above a rank-based noise floor and a valid
// pitch estimate in the human voice range).
export function extractAcousticFeatures({ samples, sampleRate }) {
  const f0Track = []
  const energyTrack = []

  for (let start = 0; start + FRAME_SIZE < samples.length; start += HOP_SIZE) {
    const frame = samples.subarray(start, start + FRAME_SIZE)
    energyTrack.push(frameEnergy(frame))
  }
  const energyFloor = [...energyTrack].sort((a, b) => a - b)[
    Math.floor(energyTrack.length * VOICED_ENERGY_PERCENTILE)
  ] || 0

  let idx = 0
  const voicedF0 = []
  const voicedEnergy = []
  for (let start = 0; start + FRAME_SIZE < samples.length; start += HOP_SIZE) {
    const frame = samples.subarray(start, start + FRAME_SIZE)
    const energy = energyTrack[idx]
    idx++
    if (energy < energyFloor * 1.4) continue // likely silence/noise
    const f0 = autocorrelationF0(frame, sampleRate)
    if (f0 && f0 >= MIN_F0 && f0 <= MAX_F0) {
      voicedF0.push(f0)
      voicedEnergy.push(energy)
    }
  }

  if (voicedF0.length < 5) {
    return { voicedFrameCount: voicedF0.length }
  }

  // Jitter: average absolute frame-to-frame F0 perturbation, normalised
  // by mean F0 — a standard voice-quality metric.
  let jitterSum = 0
  for (let i = 1; i < voicedF0.length; i++) {
    jitterSum += Math.abs(voicedF0[i] - voicedF0[i - 1])
  }
  const jitter = (jitterSum / (voicedF0.length - 1)) / mean(voicedF0)

  return {
    voicedFrameCount: voicedF0.length,
    meanF0: mean(voicedF0),
    medianF0: median(voicedF0),
    f0Std: stdDev(voicedF0),
    jitter,
    meanEnergy: mean(voicedEnergy),
    energyStd: stdDev(voicedEnergy),
  }
}

// Gender classification from fundamental frequency. Typical adult
// ranges: male ~85-165 Hz, female ~165-255 Hz. The midpoint threshold
// is the standard rule-of-thumb boundary used in speech science.
const GENDER_F0_THRESHOLD = 165

export function classifyGender(features) {
  return features.medianF0 < GENDER_F0_THRESHOLD ? 'male' : 'female'
}

// Age estimate for male voices from F0 and jitter. Grounded in two
// well-documented aging effects in the male voice: F0 tends to creep
// back up in older age after the post-puberty drop, and jitter
// (pitch instability) increases with age due to reduced vocal fold
// control. This is a hand-set linear approximation, not a model
// fitted to a labelled corpus, and is presented as such in the UI.
export function estimateMaleAge(features) {
  const { medianF0, jitter } = features
  const f0Component = Math.max(0, (140 - medianF0) * 0.3) // lower F0 within male range → skews older
  const jitterComponent = Math.min(25, Math.max(0, (jitter - 0.006) * 900)) // instability above a normal baseline → skews older
  let age = 24 + f0Component + jitterComponent
  age = Math.max(18, Math.min(85, age))
  return Math.round(age)
}

// Coarse emotion heuristic from pitch variability and energy
// variability, used only for the senior-citizen branch per the spec.
export function estimateEmotion(features) {
  const { f0Std, meanF0, energyStd, meanEnergy } = features
  const pitchVarRatio = f0Std / meanF0
  const energyVarRatio = energyStd / (meanEnergy || 1)

  if (pitchVarRatio > 0.16 && energyVarRatio > 0.5) {
    return { label: 'Agitated / Stressed', detail: 'High pitch and energy variability' }
  }
  if (pitchVarRatio < 0.06 && energyVarRatio < 0.3) {
    return { label: 'Calm / Content', detail: 'Steady pitch and even energy' }
  }
  if (pitchVarRatio < 0.08 && energyVarRatio > 0.45) {
    return { label: 'Sad / Subdued', detail: 'Flat pitch with uneven, low energy' }
  }
  return { label: 'Neutral', detail: 'No strong variability signal either way' }
}

export const SENIOR_AGE_THRESHOLD = 60
