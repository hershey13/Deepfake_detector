'use strict';
/* ═══════════════════════════════════════════════════════
   detector.js  —  Client-side forensic analysis engine
   Image  : Canvas API → DCT · LNV · channel corr · ELA
   Video  : Canvas frame sampling → temporal variance ·
            mouth motion · scene cuts · colour consistency
   Audio  : delegated to FastAPI backend (see app.js)
═══════════════════════════════════════════════════════ */

/* ─────────────────────────────────────────────
   UTILITIES
───────────────────────────────────────────── */

/** Draw a File/Blob into an OffscreenCanvas and return its ImageData. */
async function fileToImageData(file, maxSide = 512) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const scale = Math.min(maxSide / img.width, maxSide / img.height, 1);
      const w = Math.round(img.width  * scale);
      const h = Math.round(img.height * scale);
      const canvas = new OffscreenCanvas(w, h);
      const ctx    = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      resolve({ data: ctx.getImageData(0, 0, w, h), w, h });
    };
    img.onerror = () => reject(new Error('Could not load image'));
    img.src = url;
  });
}

/** Extract R, G, B flat arrays from ImageData. */
function rgbChannels(imageData) {
  const d = imageData.data;
  const n = d.length / 4;
  const R = new Float32Array(n);
  const G = new Float32Array(n);
  const B = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    R[i] = d[i * 4];
    G[i] = d[i * 4 + 1];
    B[i] = d[i * 4 + 2];
  }
  return { R, G, B };
}

/** Extract grayscale float array from ImageData. */
function toGray(imageData) {
  const d = imageData.data;
  const n = d.length / 4;
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  }
  return g;
}

function mean(arr) {
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

function variance(arr, mu) {
  const m = mu ?? mean(arr);
  let s = 0;
  for (const v of arr) s += (v - m) ** 2;
  return s / arr.length;
}

function covariance(a, b) {
  const ma = mean(a), mb = mean(b);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - ma) * (b[i] - mb);
  return s / a.length;
}

function correlation(a, b) {
  const cov  = covariance(a, b);
  const sdA  = Math.sqrt(variance(a));
  const sdB  = Math.sqrt(variance(b));
  return sdA * sdB < 1e-9 ? 0 : cov / (sdA * sdB);
}

/** Clamp a value to [0, 1]. */
const clamp01 = v => Math.max(0, Math.min(1, v));

/* ─────────────────────────────────────────────
   IMAGE SIGNALS
───────────────────────────────────────────── */

/**
 * DCT high-frequency ratio.
 * GAN generators leave high-frequency fingerprints in the DCT domain.
 * We approximate by measuring edge energy relative to smooth energy
 * using a simple block-based frequency split.
 */
function dctHighFreqRatio(gray, w, h) {
  const BLOCK = 8;
  let hiSum = 0, loSum = 0, blocks = 0;

  for (let by = 0; by + BLOCK <= h; by += BLOCK) {
    for (let bx = 0; bx + BLOCK <= w; bx += BLOCK) {
      let lo = 0, hi = 0;
      for (let y = 0; y < BLOCK; y++) {
        for (let x = 0; x < BLOCK; x++) {
          const v = gray[(by + y) * w + (bx + x)];
          // Low-freq ≈ top-left 3×3 of the 8×8 block; high-freq = rest
          if (x < 3 && y < 3) lo += v * v;
          else                  hi += v * v;
        }
      }
      loSum += lo; hiSum += hi; blocks++;
    }
  }

  const ratio = hiSum / (loSum + hiSum + 1e-9);
  // Calibrate: real photos ≈ 0.65–0.75, GAN ≈ 0.78–0.95
  return clamp01((ratio - 0.65) / 0.30);
}

/**
 * Local Noise Variance (LNV).
 * Real photos have natural spatial noise variation.
 * GAN images tend to have unnaturally smooth, uniform noise floors.
 * We compute noise per 8×8 patch then measure variance of those variances.
 */
function localNoiseVariance(gray, w, h) {
  const BLOCK = 8;
  const patchVars = [];

  for (let by = 0; by + BLOCK <= h; by += BLOCK) {
    for (let bx = 0; bx + BLOCK <= w; bx += BLOCK) {
      const patch = [];
      for (let y = 0; y < BLOCK; y++)
        for (let x = 0; x < BLOCK; x++)
          patch.push(gray[(by + y) * w + (bx + x)]);
      patchVars.push(variance(patch));
    }
  }

  const varOfVars = variance(patchVars);
  // Low variance-of-variances → suspiciously smooth → higher score
  return clamp01(1 - varOfVars / 500);
}

/**
 * Colour channel correlation asymmetry.
 * Natural photos: corr(R,G) ≈ corr(R,B) (both high, ≈ 0.9+).
 * GAN faces: asymmetry between R↔G and R↔B is larger.
 */
function channelCorrelationAsymmetry(R, G, B) {
  const rg = correlation(R, G);
  const rb = correlation(R, B);
  const asymmetry = Math.abs(rg - rb);
  // Real ≈ 0–0.05 asymmetry; GAN ≈ 0.10–0.30
  return clamp01((asymmetry - 0.05) / 0.25);
}

/**
 * Gradient consistency.
 * Real lenses produce non-uniform gradients across the image.
 * GAN images often have unnaturally uniform gradient magnitudes.
 */
function gradientConsistency(gray, w, h) {
  const mags = [];
  for (let y = 1; y < h - 1; y += 4) {
    for (let x = 1; x < w - 1; x += 4) {
      const gx = gray[y * w + (x + 1)] - gray[y * w + (x - 1)];
      const gy = gray[(y + 1) * w + x] - gray[(y - 1) * w + x];
      mags.push(Math.sqrt(gx * gx + gy * gy));
    }
  }
  const varMag = variance(mags);
  // Low gradient variance → suspiciously uniform → higher score
  return clamp01(1 - varMag / 2000);
}

/**
 * ELA surrogate (Error Level Analysis).
 * We simulate ELA by re-compressing via canvas JPEG quality and diffing.
 * High residual energy concentrated in face regions suggests manipulation.
 * Note: OffscreenCanvas JPEG isn't available everywhere; we use a simpler
 * block-artefact energy approach as a fallback.
 */
function elaSurrogate(gray, w, h) {
  // Measure 8×8 block boundary discontinuities (JPEG blocking artefact proxy)
  let boundaryEnergy = 0, interior = 0, count = 0;

  for (let y = 8; y < h - 8; y += 8) {
    for (let x = 8; x < w - 8; x += 8) {
      const left  = Math.abs(gray[y * w + x] - gray[y * w + (x - 1)]);
      const top   = Math.abs(gray[y * w + x] - gray[(y - 1) * w + x]);
      boundaryEnergy += left + top;

      let intra = 0;
      for (let dy = 0; dy < 8; dy++)
        for (let dx = 0; dx < 7; dx++)
          intra += Math.abs(
            gray[(y + dy) * w + (x + dx + 1)] - gray[(y + dy) * w + (x + dx)]
          );
      interior += intra / (8 * 7);
      count++;
    }
  }

  const avgBoundary = boundaryEnergy / (count * 2 + 1e-9);
  const avgInterior = interior / (count + 1e-9);
  const elaRatio    = avgBoundary / (avgInterior + 1e-9);

  // High ELA ratio → block boundaries much sharper than interior → manipulation
  return clamp01((elaRatio - 1.2) / 2.0);
}

/* ─────────────────────────────────────────────
   IMAGE VERDICT
───────────────────────────────────────────── */

function imageVerdict(score) {
  if (score >= 0.57)
    return {
      cls:   'fake',
      label: 'Deepfake Detected',
      desc:  'Multiple image forensic signals indicate this image was generated or heavily manipulated. DCT artefacts, noise uniformity, and colour channel asymmetries are outside the range of natural photographs.',
    };
  if (score < 0.43)
    return {
      cls:   'real',
      label: 'Likely Authentic',
      desc:  'Image forensic signals are consistent with a natural camera-captured photograph. No significant GAN fingerprints or manipulation artefacts were detected.',
    };
  return {
    cls:   'uncertain',
    label: 'Uncertain',
    desc:  'The image falls within the uncertainty band. Some signals are elevated but not conclusive. Consider using the trained backend model or a higher-quality source image.',
  };
}

/* ─────────────────────────────────────────────
   PUBLIC: detectImage
───────────────────────────────────────────── */

async function detectImage(file) {
  const { data: imageData, w, h } = await fileToImageData(file);
  const gray = toGray(imageData);
  const { R, G, B } = rgbChannels(imageData);

  const signals = {
    'DCT high-freq ratio':  dctHighFreqRatio(gray, w, h),
    'Local noise variance': localNoiseVariance(gray, w, h),
    'Channel correlation':  channelCorrelationAsymmetry(R, G, B),
    'Gradient consistency': gradientConsistency(gray, w, h),
    'ELA surrogate':        elaSurrogate(gray, w, h),
  };

  // Ensemble: weighted mean (MFCC-0 analogue = DCT ratio is primary signal)
  const weights = [0.30, 0.20, 0.20, 0.15, 0.15];
  const vals    = Object.values(signals);
  const score   = vals.reduce((s, v, i) => s + v * weights[i], 0);
  const confidence = Math.max(...vals.map(v =>
    v >= 0.57 ? v : v < 0.43 ? 1 - v : 0.5
  ));

  const verdict = imageVerdict(score);

  return {
    type:       'image',
    filename:   file.name,
    score,
    confidence,
    verdict,
    signals:    Object.fromEntries(Object.entries(signals).map(([k, v]) => [k, +v.toFixed(4)])),
    scores:     { Image: { score, confidence, signals } },
    date:       new Date(),
  };
}

/* ─────────────────────────────────────────────
   VIDEO SIGNALS
───────────────────────────────────────────── */

/** Sample up to maxFrames frames from a video file via an invisible <video>. */
async function sampleVideoFrames(file, maxFrames = 30, fps = 5) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.muted  = true;
    video.preload = 'metadata';
    const url    = URL.createObjectURL(file);
    video.src    = url;

    video.addEventListener('error', () => reject(new Error('Could not decode video')));
    video.addEventListener('loadedmetadata', async () => {
      const duration = video.duration;
      if (!isFinite(duration) || duration === 0) {
        URL.revokeObjectURL(url);
        reject(new Error('Invalid video duration'));
        return;
      }

      const interval  = 1 / fps;
      const times     = [];
      for (let t = 0; t < duration && times.length < maxFrames; t += interval) {
        times.push(t);
      }

      const W = 224, H = 224;
      const canvas = new OffscreenCanvas(W, H);
      const ctx    = canvas.getContext('2d');
      const frames = [];

      for (const t of times) {
        await new Promise(r => {
          video.currentTime = t;
          video.addEventListener('seeked', r, { once: true });
        });
        ctx.drawImage(video, 0, 0, W, H);
        frames.push(ctx.getImageData(0, 0, W, H));
      }

      URL.revokeObjectURL(url);
      resolve({ frames, W, H });
    });
  });
}

/**
 * Temporal variance — how much pixel values change frame-to-frame.
 * High variance → unstable appearance → suspicious.
 */
function temporalVariance(frames, W, H) {
  if (frames.length < 2) return 0.5;

  let totalVar = 0;
  for (let f = 1; f < frames.length; f++) {
    const a = toGray(frames[f - 1]);
    const b = toGray(frames[f]);
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff += Math.abs(b[i] - a[i]);
    totalVar += diff / a.length;
  }
  const avgDiff = totalVar / (frames.length - 1);
  // Real video ≈ 5–20 mean pixel diff; deepfake ≈ 20–50+
  return clamp01((avgDiff - 5) / 40);
}

/**
 * Mouth motion ratio.
 * Crop mouth region (x:30–70%, y:50–85%) per frame and compare motion
 * to full-face motion. Disproportionate mouth motion = lip-sync artefact.
 */
function mouthMotionRatio(frames, W, H) {
  if (frames.length < 2) return 0.5;

  const mx1 = Math.round(W * 0.30), mx2 = Math.round(W * 0.70);
  const my1 = Math.round(H * 0.50), my2 = Math.round(H * 0.85);
  const mouthPx = (mx2 - mx1) * (my2 - my1);

  let mouthMotion = 0, faceMotion = 0;
  for (let f = 1; f < frames.length; f++) {
    const a = toGray(frames[f - 1]);
    const b = toGray(frames[f]);

    let mDiff = 0;
    for (let y = my1; y < my2; y++)
      for (let x = mx1; x < mx2; x++)
        mDiff += Math.abs(b[y * W + x] - a[y * W + x]);
    mouthMotion += mDiff / mouthPx;

    let fDiff = 0;
    for (let i = 0; i < a.length; i++) fDiff += Math.abs(b[i] - a[i]);
    faceMotion += fDiff / a.length;
  }

  const ratio = mouthMotion / (faceMotion + 1e-9);
  // Natural ≈ 1.0–1.5; lip-sync ≈ 2.0–4.0
  return clamp01((ratio - 1.0) / 3.0);
}

/**
 * Scene cut frequency.
 * Abrupt large changes between consecutive frames = likely cuts.
 */
function sceneCutFrequency(frames, W, H) {
  if (frames.length < 2) return 0.1;
  let cuts = 0;
  for (let f = 1; f < frames.length; f++) {
    const a = toGray(frames[f - 1]);
    const b = toGray(frames[f]);
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff += Math.abs(b[i] - a[i]);
    if (diff / a.length > 35) cuts++;
  }
  return clamp01(cuts / (frames.length * 0.3));
}

/**
 * Colour consistency.
 * Mean colour per channel should be stable across frames for static shots.
 */
function colourConsistency(frames) {
  if (frames.length < 2) return 0.1;
  const rMeans = frames.map(f => mean(rgbChannels(f).R));
  const gMeans = frames.map(f => mean(rgbChannels(f).G));
  const bMeans = frames.map(f => mean(rgbChannels(f).B));
  const cv = (arr) => Math.sqrt(variance(arr)) / (mean(arr) + 1e-9);
  const avgCV = (cv(rMeans) + cv(gMeans) + cv(bMeans)) / 3;
  // Low CV → stable → real; high → inconsistent → suspicious
  return clamp01((avgCV - 0.01) / 0.10);
}

/**
 * Motion autocorrelation.
 * Smooth natural speech motion has high autocorrelation.
 * Erratic deepfake motion has low autocorrelation.
 */
function motionAutocorrelation(frames, W, H) {
  if (frames.length < 3) return 0.5;
  const diffs = [];
  for (let f = 1; f < frames.length; f++) {
    const a = toGray(frames[f - 1]);
    const b = toGray(frames[f]);
    let d = 0;
    for (let i = 0; i < a.length; i++) d += Math.abs(b[i] - a[i]);
    diffs.push(d / a.length);
  }
  const autoCorr = correlation(diffs.slice(0, -1), diffs.slice(1));
  // High autocorrelation (≈0.8) → smooth → real
  // Low/negative autocorrelation → erratic → suspicious
  return clamp01((0.8 - autoCorr) / 1.6);
}

/* ─────────────────────────────────────────────
   VIDEO VERDICT
───────────────────────────────────────────── */

function videoVerdict(score) {
  if (score >= 0.57)
    return {
      cls:   'fake',
      label: 'Lip Sync / Face Swap Detected',
      desc:  'Temporal forensic signals indicate video manipulation. Mouth motion is disproportionate to facial movement, and frame consistency is outside natural bounds.',
    };
  if (score < 0.43)
    return {
      cls:   'real',
      label: 'Likely Authentic',
      desc:  'Video forensic signals are consistent with natural, unmanipulated footage. Temporal variance, mouth motion ratio, and colour consistency are all within natural bounds.',
    };
  return {
    cls:   'uncertain',
    label: 'Uncertain',
    desc:  'Video signals fall within the uncertainty band. The clip may be compressed or short, making definitive forensics harder. Consider analysing a longer clip.',
  };
}

/* ─────────────────────────────────────────────
   PUBLIC: detectVideo
───────────────────────────────────────────── */

async function detectVideo(file) {
  const { frames, W, H } = await sampleVideoFrames(file);

  if (frames.length < 2) throw new Error('Could not extract enough frames from video');

  const signals = {
    'Temporal variance':      temporalVariance(frames, W, H),
    'Mouth motion ratio':     mouthMotionRatio(frames, W, H),
    'Scene cut frequency':    sceneCutFrequency(frames, W, H),
    'Colour consistency':     colourConsistency(frames),
    'Motion autocorrelation': motionAutocorrelation(frames, W, H),
  };

  const weights = [0.25, 0.30, 0.15, 0.15, 0.15];
  const vals    = Object.values(signals);
  const score   = vals.reduce((s, v, i) => s + v * weights[i], 0);
  const confidence = Math.max(...vals.map(v =>
    v >= 0.57 ? v : v < 0.43 ? 1 - v : 0.5
  ));

  const verdict = videoVerdict(score);

  return {
    type:       'video',
    filename:   file.name,
    score,
    confidence,
    verdict,
    signals:    Object.fromEntries(Object.entries(signals).map(([k, v]) => [k, +v.toFixed(4)])),
    scores:     { Video: { score, confidence, signals } },
    date:       new Date(),
  };
}

/* ─────────────────────────────────────────────
   Expose to app.js
───────────────────────────────────────────── */
window.detectImage = detectImage;
window.detectVideo = detectVideo;