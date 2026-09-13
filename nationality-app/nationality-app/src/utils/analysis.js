import * as faceapi from 'face-api.js'

// Models are fetched at runtime from a public CDN mirror of the
// face-api.js weight files, so no binary weight files need to ship
// inside this project.
const MODEL_URL =
  'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights'

let modelsReady = false
let loadingPromise = null

export function loadModels(onProgress) {
  if (modelsReady) return Promise.resolve()
  if (loadingPromise) return loadingPromise

  loadingPromise = (async () => {
    const steps = [
      ['Tiny Face Detector', faceapi.nets.tinyFaceDetector],
      ['Expression Model', faceapi.nets.faceExpressionNet],
      ['Age & Gender Model', faceapi.nets.ageGenderNet],
    ]
    for (let i = 0; i < steps.length; i++) {
      const [label, net] = steps[i]
      onProgress?.({ label, index: i, total: steps.length })
      await net.loadFromUri(MODEL_URL)
    }
    modelsReady = true
  })()

  return loadingPromise
}

// Samples the average colour of the torso region just below the
// detected face box, as a lightweight stand-in for a dedicated
// clothing-segmentation model.
function sampleDressColor(image, box) {
  const canvas = document.createElement('canvas')
  const w = image.naturalWidth || image.width
  const h = image.naturalHeight || image.height
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  ctx.drawImage(image, 0, 0, w, h)

  const sampleX = Math.max(0, box.x - box.width * 0.25)
  const sampleY = Math.min(h - 1, box.y + box.height * 1.15)
  const sampleW = Math.min(w - sampleX, box.width * 1.5)
  const sampleH = Math.min(h - sampleY, box.height * 0.9)

  if (sampleW <= 0 || sampleH <= 0) return null

  const data = ctx.getImageData(
    Math.round(sampleX),
    Math.round(sampleY),
    Math.max(1, Math.round(sampleW)),
    Math.max(1, Math.round(sampleH))
  ).data

  let r = 0, g = 0, b = 0, n = 0
  for (let i = 0; i < data.length; i += 4) {
    r += data[i]
    g += data[i + 1]
    b += data[i + 2]
    n++
  }
  if (!n) return null
  r = Math.round(r / n)
  g = Math.round(g / n)
  b = Math.round(b / n)

  return { r, g, b, hex: rgbToHex(r, g, b), name: nearestColorName(r, g, b) }
}

function rgbToHex(r, g, b) {
  return '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')
}

const NAMED_COLORS = [
  ['Black', 20, 20, 20],
  ['White', 245, 245, 245],
  ['Grey', 130, 130, 130],
  ['Red', 200, 40, 40],
  ['Maroon', 120, 30, 40],
  ['Orange', 220, 120, 40],
  ['Yellow', 220, 200, 60],
  ['Beige', 220, 200, 170],
  ['Green', 60, 140, 80],
  ['Teal', 40, 140, 140],
  ['Blue', 50, 90, 180],
  ['Navy', 30, 45, 90],
  ['Purple', 120, 60, 150],
  ['Pink', 220, 140, 170],
  ['Brown', 110, 75, 50],
]

function nearestColorName(r, g, b) {
  let best = NAMED_COLORS[0]
  let bestDist = Infinity
  for (const c of NAMED_COLORS) {
    const d = (r - c[1]) ** 2 + (g - c[2]) ** 2 + (b - c[3]) ** 2
    if (d < bestDist) {
      bestDist = d
      best = c
    }
  }
  return best[0]
}

const EMOTION_LABELS = {
  neutral: 'Neutral',
  happy: 'Happy',
  sad: 'Sad',
  angry: 'Angry',
  fearful: 'Fearful',
  disgusted: 'Disgusted',
  surprised: 'Surprised',
}

// Runs face detection + expression + age estimation, and derives a
// dress-colour sample. `nationality` is supplied by the user (see
// README for why this is not predicted from the image).
export async function analyzeImage(imageEl, nationality) {
  await loadModels()

  const detection = await faceapi
    .detectSingleFace(imageEl, new faceapi.TinyFaceDetectorOptions())
    .withFaceExpressions()
    .withAgeAndGender()

  if (!detection) {
    return { faceFound: false }
  }

  const expressions = detection.expressions
  const topEmotion = Object.entries(expressions).sort((a, b) => b[1] - a[1])[0]
  const emotion = {
    label: EMOTION_LABELS[topEmotion[0]] || topEmotion[0],
    confidence: topEmotion[1],
    all: expressions,
  }

  const age = Math.round(detection.age)
  const dressColor = sampleDressColor(imageEl, detection.detection.box)

  const needsAge = nationality === 'indian' || nationality === 'us'
  const needsDressColor = nationality === 'indian' || nationality === 'african'

  return {
    faceFound: true,
    nationality,
    emotion,
    age: needsAge ? age : null,
    dressColor: needsDressColor ? dressColor : null,
    box: detection.detection.box,
  }
}
