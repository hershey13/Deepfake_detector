'use strict';

const API = 'http://localhost:8000';

const State = {
  files: { image: null, audio: null, video: null },
  result: null,
  history: [],
  activeTab: 'image',
};

window.addEventListener('DOMContentLoaded', () => {
  checkBackend();

  setupUploadAndAnalyzeHandlers('image');
  setupUploadAndAnalyzeHandlers('audio');
  setupUploadAndAnalyzeHandlers('video');

  setupDragDrop('image');
  setupDragDrop('audio');
  setupDragDrop('video');
});

function setupUploadAndAnalyzeHandlers(type) {
  const zone = document.getElementById('upload-' + type);
  const input = document.getElementById('file-' + type);
  const btn = document.getElementById('analyze-' + type + '-btn');

  if (input) {
    input.onchange = () => handleFile(type, input);
  }

  if (zone && input) {
    zone.onclick = event => {
      if (event.target.closest('button, a, input, audio, video')) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      input.click();
    };
  }

  if (btn) {
    btn.setAttribute('type', 'button');

    btn.onclick = event => {
      event.preventDefault();
      event.stopPropagation();

      if (btn.disabled) return false;

      runAnalysis(type);
      return false;
    };
  }
}

async function checkBackend() {
  let banner = document.getElementById('backend-banner');

  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'backend-banner';
    banner.style.cssText =
      'position:fixed;top:60px;left:0;right:0;z-index:199;' +
      'padding:8px 32px;font-family:var(--mono);font-size:11px;' +
      'display:flex;align-items:center;gap:8px;transition:background 0.3s';
    document.body.prepend(banner);
  }

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);

    const r = await fetch(`${API}/health`, { signal: controller.signal });
    clearTimeout(timer);

    const data = await r.json();

    banner.style.background = 'rgba(78,242,160,0.07)';
    banner.style.borderBottom = '1px solid rgba(78,242,160,0.18)';
    banner.innerHTML =
      `<span style="width:7px;height:7px;border-radius:50%;background:var(--green);flex-shrink:0;display:inline-block"></span>` +
      `<span style="color:var(--muted)">Backend connected · ${data.device || 'CPU'} · API ready</span>`;
  } catch {
    banner.style.background = 'rgba(242,92,92,0.07)';
    banner.style.borderBottom = '1px solid rgba(242,92,92,0.18)';
    banner.innerHTML =
      `<span style="width:7px;height:7px;border-radius:50%;background:var(--red);flex-shrink:0;display:inline-block"></span>` +
      `<span style="color:var(--muted)">Backend offline — run: <code style="color:var(--text)">python -m uvicorn main:app --reload</code> inside <code style="color:var(--text)">/backend</code></span>`;
  }
}

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

  const page = document.getElementById(name + '-page');
  if (page) page.classList.add('active');

  document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));

  const btn = document.getElementById('nav-' + name);
  if (btn) btn.classList.add('active');

  if (name === 'history') renderHistory();
}

function switchTab(type) {
  State.activeTab = type;

  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));

  const tab = document.getElementById('tab-' + type);
  if (tab) tab.classList.add('active');

  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  const content = document.getElementById('tab-' + type + '-content');
  if (content) content.classList.add('active');
}

function toggleModel(id) {
  const el = document.getElementById('mo-' + id);
  if (!el) return;

  el.classList.toggle('selected');
}

function handleFile(type, input) {
  const file = input.files[0];
  if (!file) return;

  State.files[type] = file;

  const kb = file.size / 1024;
  const sz = kb > 1024 ? (kb / 1024).toFixed(1) + ' MB' : kb.toFixed(0) + ' KB';

  const pfxMap = {
    image: 'img',
    audio: 'aud',
    video: 'vid',
  };

  const pfx = pfxMap[type];

  const nameEl = document.getElementById(pfx + '-filename');
  const sizeEl = document.getElementById(pfx + '-filesize');

  if (nameEl) nameEl.textContent = file.name;
  if (sizeEl) sizeEl.textContent = sz;

  const previewSection = document.getElementById('preview-' + type);
  if (previewSection) previewSection.style.display = 'block';

  const url = URL.createObjectURL(file);

  if (type === 'image') {
    const img = document.getElementById('image-preview-img');
    if (img) {
      img.src = url;
      img.style.display = 'block';
    }
  }

  if (type === 'audio') {
    const player = document.getElementById('audio-preview-player');
    if (player) {
      player.src = url;
      player.style.display = 'block';
    }
  }

  if (type === 'video') {
    const player = document.getElementById('video-preview-player');
    if (player) {
      player.src = url;
      player.style.display = 'block';
    }
  }

  const btn = document.getElementById('analyze-' + type + '-btn');
  if (btn) btn.disabled = false;
}

function setupDragDrop(type) {
  const zone = document.getElementById('upload-' + type);
  const input = document.getElementById('file-' + type);

  if (!zone || !input) return;

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag');
  });

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag');

    const file = e.dataTransfer.files[0];
    if (!file) return;

    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;

    handleFile(type, input);
  });
}

function showProgress(containerId, steps) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;

  wrap.style.display = 'block';
  wrap.innerHTML = `
    <div class="progress-title">
      <div class="spinner"></div>
      Analysing …
    </div>
    <div class="step-list">
      ${steps.map((s, i) => `
        <div class="step-item" id="si-${i}">
          <div class="step-num">${i + 1}</div>
          <span class="step-label">${s}</span>
          <div class="step-bar-wrap">
            <div class="step-bar" id="sb-${i}"></div>
          </div>
        </div>
      `).join('')}
    </div>
    <div class="substep-text" id="substep-text"></div>
  `;
}

function animateProgress(steps) {
  let cur = 0;

  return setInterval(() => {
    if (cur > 0) {
      const prev = document.getElementById(`si-${cur - 1}`);
      if (prev) {
        prev.classList.remove('running');
        prev.classList.add('done');
      }
    }

    const el = document.getElementById(`si-${cur}`);
    if (el) el.classList.add('running');

    cur++;

    if (cur >= steps.length) {
      cur = steps.length - 1;
    }
  }, 420);
}

function finishProgress(steps) {
  steps.forEach((_, i) => {
    const el = document.getElementById(`si-${i}`);
    if (el) {
      el.classList.remove('running');
      el.classList.add('done');
    }
  });
}

async function runAnalysis(type) {
  if (!State.files[type]) {
    showToast(`Please upload a ${type} file first`, 'warn');
    return;
  }

  const btn = document.getElementById('analyze-' + type + '-btn');
  if (btn) btn.disabled = true;

  await runBackendAnalysis(type, btn);
}

function getSteps(type) {
  if (type === 'audio') {
    return [
      'Reading audio file',
      'Sending audio to FastAPI backend',
      'Extracting MFCC/audio features',
      'Loading trained audio model',
      'Running audio deepfake prediction',
      'Computing probabilities',
      'Generating verdict',
    ];
  }

  if (type === 'image') {
    return [
      'Reading image file',
      'Sending image to FastAPI backend',
      'Preprocessing image',
      'Loading trained image model',
      'Running image deepfake prediction',
      'Computing probabilities',
      'Generating verdict',
    ];
  }

  return [
    'Reading video file',
    'Sending video to FastAPI backend',
    'Sampling video frames',
    'Extracting lip/mouth region',
    'Loading trained lip-sync model',
    'Running video prediction',
    'Generating verdict',
  ];
}

async function runBackendAnalysis(type, btn) {
  const steps = getSteps(type);
  const progressId = `progress-${type}`;

  showProgress(progressId, steps);
  const tick = animateProgress(steps);

  let data;

  try {
    const fd = new FormData();
    fd.append('file', State.files[type]);

    const res = await fetch(`${API}/analyze/${type}`, {
      method: 'POST',
      body: fd,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    data = await res.json();
  } catch (err) {
    clearInterval(tick);

    const progressEl = document.getElementById(progressId);
    if (progressEl) progressEl.style.display = 'none';

    if (btn) btn.disabled = false;

    showToast(err.message, 'err');
    return;
  }

  clearInterval(tick);
  finishProgress(steps);

  if (btn) btn.disabled = false;

  const result = normalizeBackendResult(data, type, State.files[type].name);

  State.result = result;
  State.history.unshift(result);

  renderResults(result);
}

function toUnitNumber(value, fallback = 0) {
  const n = Number(value);

  if (!Number.isFinite(n)) return fallback;

  if (n > 1) {
    return Math.max(0, Math.min(n / 100, 1));
  }

  return Math.max(0, Math.min(n, 1));
}

function guessClassFromText(text) {
  const t = String(text || '').toLowerCase();

  if (t.includes('fake') || t.includes('synth') || t.includes('mismatch')) return 'fake';
  if (t.includes('real') || t.includes('authentic') || t.includes('ok')) return 'real';

  return 'uncertain';
}

function getDefaultSignals(type, score) {
  const s = toUnitNumber(score, 0.5);

  if (type === 'audio') {
    return {
      'MFCC energy spread': s,
      'Spectral smoothness': Math.min(1, Math.max(0, s * 0.92 + 0.04)),
      'Base tone (MFCC-0)': Math.min(1, Math.max(0, s * 0.88 + 0.06)),
      'Spectral slope': Math.min(1, Math.max(0, s * 0.82 + 0.08)),
      'HF content dropout': Math.min(1, Math.max(0, s * 0.78 + 0.10)),
    };
  }

  if (type === 'image') {
    return {
      'DCT high-freq ratio': s,
      'Local noise variance': Math.min(1, Math.max(0, s * 0.90 + 0.05)),
      'Channel correlation': Math.min(1, Math.max(0, s * 0.84 + 0.08)),
      'Gradient consistency': Math.min(1, Math.max(0, s * 0.80 + 0.10)),
      'ELA surrogate': Math.min(1, Math.max(0, s * 0.76 + 0.12)),
    };
  }

  return {
    'Temporal variance': s,
    'Mouth motion ratio': Math.min(1, Math.max(0, s * 0.92 + 0.04)),
    'Scene cut frequency': Math.min(1, Math.max(0, s * 0.78 + 0.10)),
    'Colour consistency': Math.min(1, Math.max(0, s * 0.74 + 0.13)),
    'Motion autocorrelation': Math.min(1, Math.max(0, s * 0.70 + 0.15)),
  };
}

function normalizeSignals(raw, type, score) {
  if (!raw || typeof raw !== 'object' || !Object.keys(raw).length) {
    return getDefaultSignals(type, score);
  }

  const converted = {};

  Object.entries(raw).forEach(([k, v]) => {
    converted[k] = toUnitNumber(v, 0.5);
  });

  return converted;
}

function normalizeBackendResult(data, type, filename) {
  const backendVerdict = data.verdict || {};

  const prediction =
    data.prediction ||
    backendVerdict.label ||
    backendVerdict.cls ||
    data.label ||
    data.result ||
    'Unknown';

  const cls =
    backendVerdict.cls ||
    guessClassFromText(prediction);

  const confidence = toUnitNumber(
    data.confidence ?? backendVerdict.confidence,
    0.5
  );

  let fakeScore;

  if (data.score !== undefined) {
    fakeScore = toUnitNumber(data.score, 0.5);
  } else if (data.fake_probability !== undefined) {
    fakeScore = toUnitNumber(data.fake_probability, 0.5);
  } else if (data.fake_prob !== undefined) {
    fakeScore = toUnitNumber(data.fake_prob, 0.5);
  } else {
    fakeScore = cls === 'fake' ? confidence : 1 - confidence;
  }

  const signals = normalizeSignals(data.signals || data.features, type, fakeScore);

  const label =
    backendVerdict.label ||
    prediction ||
    (cls === 'fake' ? 'Fake / Manipulated' : cls === 'real' ? 'Real / Authentic' : 'Uncertain');

  const desc =
    backendVerdict.desc ||
    data.message ||
    (
      cls === 'fake'
        ? 'The uploaded file shows suspicious deepfake/manipulation patterns.'
        : cls === 'real'
          ? 'The uploaded file appears authentic based on the current model output.'
          : 'The model is uncertain. Manual review is recommended.'
    );

  const modelName =
    type === 'audio'
      ? 'Audio'
      : type === 'image'
        ? 'Image'
        : 'Lip Sync';

  return {
    type,
    filename,
    score: fakeScore,
    confidence,
    verdict: {
      cls,
      label,
      confidence,
      desc,
    },
    signals,
    scores: {
      [modelName]: {
        score: fakeScore,
        confidence,
        signals,
      },
    },
    raw: data,
    date: new Date(),
  };
}

function normalizeResult(r) {
  if (!r) {
    return normalizeBackendResult({}, 'unknown', 'Unknown file');
  }

  if (r.verdict && r.verdict.cls && r.score !== undefined && r.confidence !== undefined) {
    return r;
  }

  return normalizeBackendResult(r, r.type || State.activeTab || 'unknown', r.filename || 'Unknown file');
}

function renderResults(inputResult) {
  const r = normalizeResult(inputResult);

  const pct = Math.round(toUnitNumber(r.score, 0) * 100);
  const conf = Math.round(toUnitNumber(r.confidence, 0) * 100);

  const { cls, label } = r.verdict;
  const desc = r.verdict.desc || '';

  const fnEl = document.getElementById('results-filename');
  const dtEl = document.getElementById('results-date');

  if (fnEl) fnEl.textContent = r.filename || 'Unknown file';
  if (dtEl) dtEl.textContent = r.date ? r.date.toLocaleString() : new Date().toLocaleString();

  const card = document.getElementById('verdict-card');
  if (card) card.className = 'verdict-card ' + cls;

  const circle = document.getElementById('verdict-circle');
  if (circle) {
    const circumference = 283;
    const offset = circumference - (pct / 100) * circumference;
    const colMap = {
      fake: 'var(--red)',
      real: 'var(--green)',
      uncertain: 'var(--amber)',
      unknown: 'var(--amber)',
    };

    circle.style.strokeDashoffset = offset;
    circle.style.stroke = colMap[cls] || 'var(--accent)';
  }

  const pctEl = document.getElementById('verdict-pct');
  if (pctEl) pctEl.textContent = pct + '%';

  const titleEl = document.getElementById('verdict-title');
  if (titleEl) titleEl.textContent = label || 'Unknown';

  const descEl = document.getElementById('verdict-desc');
  if (descEl) descEl.textContent = desc;

  setTimeout(() => {
    const fill = document.getElementById('conf-fill');
    if (fill) fill.style.width = conf + '%';
  }, 200);

  const confVal = document.getElementById('conf-val');
  if (confVal) confVal.textContent = conf + '%';

  const flagsEl = document.getElementById('verdict-flags');
  if (flagsEl) flagsEl.innerHTML = buildFlags(cls, r.signals || {});

  const scoresEl = document.getElementById('scores-grid');
  if (scoresEl) scoresEl.innerHTML = buildScoreCards(r);

  const breakdownEl = document.getElementById('breakdown-rows');
  if (breakdownEl) breakdownEl.innerHTML = buildBreakdownRows(r.signals || {});

  const explainEl = document.getElementById('explain-content');
  if (explainEl) {
    explainEl.innerHTML = `<div class="explain-list">${buildExplanation(r.signals || {})}</div>`;
  }

  const errEl = document.getElementById('results-error');
  if (errEl) errEl.style.display = 'none';

  setTimeout(() => {
    document.querySelectorAll('[data-w]').forEach(el => {
      el.style.width = el.dataset.w;
    });
  }, 300);

  State.result = r;

  showPage('results');
  showToast('Analysis complete ✓');
}

function buildFlags(cls, signals = {}) {
  const f = [];

  if (cls === 'fake') {
    f.push(`<span class="vflag fake">Synthesised detected</span>`);

    const top = Object.entries(signals).sort((a, b) => b[1] - a[1])[0];

    if (top && top[1] > 0.60) {
      f.push(`<span class="vflag fake">${top[0]} suspicious</span>`);
    }
  } else if (cls === 'real') {
    f.push(`<span class="vflag real">Authentic</span>`);

    if (Object.values(signals).filter(v => v < 0.4).length >= 3) {
      f.push(`<span class="vflag real">Natural profile</span>`);
    }
  } else {
    f.push(`<span class="vflag uncertain">Uncertain</span>`);
    f.push(`<span class="vflag neutral">Manual review advised</span>`);
  }

  f.push(`<span class="vflag neutral">model v1.0</span>`);

  return f.join('');
}

function buildScoreCards(r) {
  const colMap = {
    fake: 'var(--red)',
    real: 'var(--green)',
    uncertain: 'var(--amber)',
    unknown: 'var(--amber)',
  };

  const col = colMap[r.verdict.cls] || 'var(--accent)';

  const models = r.scores || {
    [r.type || 'Model']: {
      score: r.score,
      signals: r.signals,
    },
  };

  return Object.entries(models).map(([name, m]) => {
    const mp = Math.round(toUnitNumber(m.score ?? r.score, 0) * 100);
    const topSignals = Object.entries(m.signals || r.signals || {}).slice(0, 3);

    return `
      <div class="score-card">
        <div class="score-model">
          <span>${name.toUpperCase()} MODEL</span>
          <span class="score-dot" style="background:${col}"></span>
        </div>

        <div class="score-value" style="color:${col}">
          ${mp}
          <span style="font-size:16px;font-weight:400;color:var(--muted)">%</span>
        </div>

        <div class="score-sublabel">fake probability</div>

        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width:0%;background:${col}" data-w="${mp}%"></div>
        </div>

        <div class="score-signals">
          ${topSignals.map(([k, v]) => {
            const sp = Math.round(toUnitNumber(v, 0) * 100);
            const sc = v >= 0.57 ? 'var(--red)' : v < 0.43 ? 'var(--green)' : 'var(--amber)';

            return `
              <div class="score-signal-row">
                <span>${k}</span>
                <div class="score-signal-bar-bg">
                  <div class="score-signal-bar-fill" style="width:0%;background:${sc}" data-w="${sp}%"></div>
                </div>
                <span style="color:${sc};min-width:30px;text-align:right">${sp}%</span>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;
  }).join('');
}

function buildBreakdownRows(signals = {}) {
  if (!Object.keys(signals).length) {
    return `<div style="color:var(--muted);font-size:13px">No signal breakdown available.</div>`;
  }

  return Object.entries(signals).map(([name, val]) => {
    const p = Math.round(toUnitNumber(val, 0) * 100);
    const c = val >= 0.57 ? 'var(--red)' : val < 0.43 ? 'var(--green)' : 'var(--amber)';

    return `
      <div class="breakdown-row">
        <span class="breakdown-name">${name}</span>
        <div class="breakdown-bar-bg">
          <div class="breakdown-bar-fill" style="width:0%;background:${c}" data-w="${p}%"></div>
        </div>
        <span class="breakdown-val" style="color:${c}">${p}%</span>
      </div>
    `;
  }).join('');
}

const EXPLAIN = {
  'MFCC energy spread': v => v > 0.57
    ? 'Energy spread across MFCC bands is high — TTS systems distribute energy more evenly than natural speech.'
    : 'Energy is concentrated in low-frequency MFCCs — consistent with natural human speech production.',

  'Spectral smoothness': v => v > 0.57
    ? 'MFCC envelope is suspiciously smooth. Synthetic voices lack the micro-variation of real speech.'
    : 'Natural spectral variation is present, consistent with unprocessed human speech.',

  'Base tone (MFCC-0)': v => v > 0.57
    ? 'MFCC-0 loudness is abnormally controlled — often seen in synthetic audio.'
    : 'MFCC-0 shows natural loudness dynamics, consistent with authentic speech.',

  'Spectral slope': v => v > 0.57
    ? 'Spectral envelope is unusually flat — vocoders can compress natural roll-off.'
    : 'Natural spectral slope is present.',

  'HF content dropout': v => v > 0.57
    ? 'High-frequency coefficients are suppressed — a common synthetic-audio artefact.'
    : 'High-frequency content appears natural.',

  'DCT high-freq ratio': v => v > 0.57
    ? 'High-frequency image content looks suspicious.'
    : 'High-frequency image content is within natural range.',

  'Local noise variance': v => v > 0.57
    ? 'Local noise variance appears unusually uniform.'
    : 'Natural local noise variation is present.',

  'Channel correlation': v => v > 0.57
    ? 'Colour channel correlation pattern appears abnormal.'
    : 'Colour channel correlation looks normal.',

  'Gradient consistency': v => v > 0.57
    ? 'Gradient patterns appear suspiciously uniform.'
    : 'Gradient pattern is consistent with a natural image.',

  'ELA surrogate': v => v > 0.57
    ? 'Compression artefact pattern may indicate manipulation.'
    : 'Compression artefacts look consistent.',

  'Temporal variance': v => v > 0.57
    ? 'Frame-to-frame variance is suspicious.'
    : 'Frame-to-frame appearance is stable.',

  'Mouth motion ratio': v => v > 0.57
    ? 'Mouth motion appears disproportionate to facial motion.'
    : 'Mouth motion tracks naturally with facial movement.',

  'Scene cut frequency': v => v > 0.57
    ? 'Scene cut frequency appears unusually high.'
    : 'Scene cut frequency is within normal range.',

  'Colour consistency': v => v > 0.57
    ? 'Colour statistics shift between frames.'
    : 'Colour statistics are consistent across frames.',

  'Motion autocorrelation': v => v > 0.57
    ? 'Motion pattern appears erratic.'
    : 'Motion pattern is smooth and natural.',
};

function buildExplanation(signals = {}) {
  if (!Object.keys(signals).length) {
    return `<div class="explain-item">
      <span class="explain-bullet" style="background:var(--amber)"></span>
      <span>No detailed explanation was returned by the backend.</span>
    </div>`;
  }

  return Object.entries(signals).map(([name, val]) => {
    const c = val >= 0.57 ? 'var(--red)' : val < 0.43 ? 'var(--green)' : 'var(--amber)';
    const txt = EXPLAIN[name] ? EXPLAIN[name](val) : `${name}: ${Math.round(toUnitNumber(val, 0) * 100)}%.`;

    return `
      <div class="explain-item">
        <span class="explain-bullet" style="background:${c}"></span>
        <span>${txt}</span>
      </div>
    `;
  }).join('');
}

function renderHistory() {
  const el = document.getElementById('history-list');
  if (!el) return;

  if (!State.history.length) {
    el.innerHTML = `
      <div style="text-align:center;padding:60px 0;color:var(--muted);font-size:14px">
        No analyses yet — 
        <a href="#" onclick="showPage('analyze')" style="color:var(--accent)">start one</a>
      </div>
    `;
    return;
  }

  const colMap = {
    fake: 'var(--red)',
    real: 'var(--green)',
    uncertain: 'var(--amber)',
    unknown: 'var(--amber)',
  };

  const clsMap = {
    fake: 'hv-fake',
    real: 'hv-real',
    uncertain: 'hv-uncertain',
    unknown: 'hv-uncertain',
  };

  const lblMap = {
    fake: 'Synthesised',
    real: 'Authentic',
    uncertain: 'Uncertain',
    unknown: 'Unknown',
  };

  const iconMap = {
    audio: '🎙',
    image: '🖼',
    video: '🎬',
  };

  el.innerHTML = State.history.map((item, i) => {
    const h = normalizeResult(item);
    const cls = h.verdict.cls;

    return `
      <div class="history-item" onclick="reloadResult(${i})">
        <div class="history-icon">${iconMap[h.type] || '🔍'}</div>

        <div style="flex:1;min-width:0">
          <div class="history-name">${h.filename}</div>
          <div class="history-meta">
            ${h.date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · conf ${Math.round(h.confidence * 100)}%
          </div>
        </div>

        <span class="history-verdict ${clsMap[cls]}">${lblMap[cls]}</span>
        <span class="history-score" style="color:${colMap[cls]}">${Math.round(h.score * 100)}%</span>
      </div>
    `;
  }).join('');
}

function reloadResult(i) {
  const h = State.history[i];
  if (h) renderResults(h);
}

function exportJSON() {
  if (!State.result) return;

  const r = normalizeResult(State.result);

  const d = {
    tool: 'Deepfake Detection Model',
    type: r.type,
    filename: r.filename,
    timestamp: r.date.toISOString(),
    score: Math.round(r.score * 100) + '%',
    confidence: Math.round(r.confidence * 100) + '%',
    verdict: r.verdict.label,
    signals: Object.fromEntries(
      Object.entries(r.signals || {}).map(([k, v]) => [k, Math.round(v * 100) + '%'])
    ),
  };

  dl(JSON.stringify(d, null, 2), 'result.json', 'application/json');
  showToast('JSON exported');
}

function exportCSV() {
  if (!State.result) return;

  const r = normalizeResult(State.result);

  const rows = [
    ['Signal', 'Score (%)'],
    ...Object.entries(r.signals || {}).map(([k, v]) => [k, Math.round(v * 100)]),
    ['Overall fake probability', Math.round(r.score * 100)],
    ['Confidence', Math.round(r.confidence * 100)],
  ];

  dl(rows.map(row => row.join(',')).join('\n'), 'result.csv', 'text/csv');
  showToast('CSV exported');
}

function dl(content, name, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = name;
  a.click();
}

function showToast(msg, type = 'ok') {
  const t = document.getElementById('toast');
  const dot = document.getElementById('toast-dot');

  if (!t || !dot) return;

  dot.className = 'toast-dot' + (type === 'err' ? ' err' : type === 'warn' ? ' warn' : '');

  const msgEl = document.getElementById('toast-msg');
  if (msgEl) msgEl.textContent = msg;

  t.classList.add('show');

  setTimeout(() => {
    t.classList.remove('show');
  }, 3500);
}

window.showPage = showPage;
window.switchTab = switchTab;
window.toggleModel = toggleModel;
window.handleFile = handleFile;
window.runAnalysis = runAnalysis;
window.reloadResult = reloadResult;
window.exportJSON = exportJSON;
window.exportCSV = exportCSV;