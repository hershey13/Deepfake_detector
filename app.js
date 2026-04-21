/**
 * Deepfake Detection Model — app.js
 * UI controller: routing, file handling, progress, results rendering.
 * Calls window.DeepfakeDetector (detector.js) for actual signal extraction.
 */

'use strict';

/* ─── State ─── */
const State = {
  history: [],
  currentResult: null,
  activeTab: 'image',
  files: { image: null, audio: null, video: null },
  running: false,
};

/* ─── Page routing ─── */
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const page = document.getElementById(name + '-page');
  if (page) page.classList.add('active');
  document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('nav-' + name);
  if (btn) btn.classList.add('active');
  if (name === 'history') renderHistory();
}

/* ─── Tab switching ─── */
function switchTab(name) {
  State.activeTab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name)?.classList.add('active');
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name + '-content')?.classList.add('active');
}

/* ─── File handling ─── */
function handleFile(type, input) {
  const file = input.files[0];
  if (!file) return;
  State.files[type] = file;

  const sizeTxt = file.size > 1048576
    ? (file.size / 1048576).toFixed(1) + ' MB'
    : (file.size / 1024).toFixed(0) + ' KB';

  if (type === 'image') {
    document.getElementById('img-filename').textContent = file.name;
    document.getElementById('img-filesize').textContent = sizeTxt;
    const img = document.getElementById('image-preview-img');
    img.src = URL.createObjectURL(file);
    img.style.display = 'block';
    document.getElementById('preview-image').style.display = 'block';
    document.getElementById('analyze-image-btn').disabled = false;
  } else if (type === 'audio') {
    document.getElementById('aud-filename').textContent = file.name;
    document.getElementById('aud-filesize').textContent = sizeTxt;
    const aud = document.getElementById('audio-preview-player');
    aud.src = URL.createObjectURL(file);
    aud.style.display = 'block';
    document.getElementById('preview-audio').style.display = 'block';
    document.getElementById('analyze-audio-btn').disabled = false;
  } else if (type === 'video') {
    document.getElementById('vid-filename').textContent = file.name;
    document.getElementById('vid-filesize').textContent = sizeTxt;
    const vid = document.getElementById('video-preview-player');
    vid.src = URL.createObjectURL(file);
    vid.style.display = 'block';
    document.getElementById('preview-video').style.display = 'block';
    document.getElementById('analyze-video-btn').disabled = false;
  }
}

/* ─── Model toggles ─── */
function toggleModel(id) {
  const el = document.getElementById('mo-' + id);
  el?.classList.toggle('selected');
}

/* ─── Run analysis ─── */
async function runAnalysis(type) {
  if (State.running) return;
  const file = State.files[type];
  if (!file) return;

  State.running = true;
  const btn = document.getElementById('analyze-' + type + '-btn');
  btn.disabled = true;

  const stepsMap = {
    image: [
      'Loading image into Canvas',
      'Computing DCT frequency artefacts',
      'Measuring noise variance (LNV)',
      'Analysing colour channel correlation',
      'Running gradient consistency check',
      'ELA surrogate analysis',
      'Computing ensemble verdict',
    ],
    audio: [
      'Decoding audio buffer',
      'Running FFT spectral analysis',
      'Measuring spectral flatness',
      'Checking silence purity',
      'Analysing sub-band energy',
      'Pitch periodicity check',
      'Computing ensemble verdict',
    ],
    video: [
      'Sampling video frames',
      'Extracting facial region luma',
      'Analysing mouth-region motion',
      'Detecting scene cuts',
      'Checking colour consistency',
      'Frame-diff autocorrelation',
      'Computing ensemble verdict',
    ],
  };

  const steps = stepsMap[type];
  const progressEl = document.getElementById('progress-' + type);
  progressEl.innerHTML = `
    <div class="progress-title"><div class="spinner"></div> Analysing — this may take a few seconds</div>
    <div class="step-list">
      ${steps.map((s, i) => `
        <div class="step-item" id="step-${type}-${i}">
          <div class="step-num">${i + 1}</div>
          <span class="step-label">${s}</span>
          <div class="step-bar-wrap"><div class="step-bar" id="bar-${type}-${i}"></div></div>
        </div>
      `).join('')}
    </div>
    <div class="substep-text" id="substep-${type}"></div>
  `;
  progressEl.style.display = 'block';

  // Animate UI steps in parallel with actual analysis
  let uiStep = 0;
  const totalSteps = steps.length;
  const uiInterval = setInterval(() => {
    if (uiStep > 0) {
      const prev = document.getElementById(`step-${type}-${uiStep - 1}`);
      prev?.classList.remove('running'); prev?.classList.add('done');
      document.getElementById(`bar-${type}-${uiStep - 1}`)?.style && (document.getElementById(`bar-${type}-${uiStep - 1}`).style.width = '100%');
    }
    if (uiStep < totalSteps) {
      const el = document.getElementById(`step-${type}-${uiStep}`);
      el?.classList.add('running');
      uiStep++;
    }
  }, 700);

  // Run actual detection
  let analysisResult;
  const substep = document.getElementById(`substep-${type}`);
  try {
    if (substep) substep.textContent = '→ Extracting signals from file…';
    if (type === 'image') {
      analysisResult = await window.DeepfakeDetector.analyseImage(file);
    } else if (type === 'audio') {
      analysisResult = await window.DeepfakeDetector.analyseAudio(file);
    } else if (type === 'video') {
      analysisResult = await window.DeepfakeDetector.analyseVideo(file);
    }
    if (substep) substep.textContent = '→ Running ensemble fusion…';
  } catch(e) {
    analysisResult = { score: 0.5, signals: {}, error: e.message };
    if (substep) substep.textContent = '→ Analysis error — using fallback';
  }

  // Finish UI steps
  clearInterval(uiInterval);
  for (let i = 0; i < totalSteps; i++) {
    const el = document.getElementById(`step-${type}-${i}`);
    el?.classList.remove('running'); el?.classList.add('done');
    if (document.getElementById(`bar-${type}-${i}`))
      document.getElementById(`bar-${type}-${i}`).style.width = '100%';
  }

  // Ensemble (single model here — multi-model when both run)
  const fusion = window.DeepfakeDetector.ensembleFusion({ [type]: analysisResult });
  const verdict = window.DeepfakeDetector.getVerdict(fusion.score, fusion.confidence);

  const result = {
    filename: file.name,
    type,
    score: fusion.score,
    confidence: fusion.confidence,
    verdict,
    signals: analysisResult.signals || {},
    raw: { [type]: analysisResult },
    date: new Date(),
    error: analysisResult.error,
  };

  State.running = false;
  btn.disabled = false;
  showResults(result);
}

/* ─── Render results ─── */
function showResults(result) {
  State.currentResult = result;
  State.history.unshift(result);

  const scorePct = Math.round(result.score * 100);
  const confPct  = Math.round(result.confidence * 100);
  const { cls, title, desc } = result.verdict;

  document.getElementById('results-filename').textContent = result.filename;
  document.getElementById('results-date').textContent =
    result.date.toLocaleDateString() + ' · ' + result.date.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});

  /* Verdict card */
  const card = document.getElementById('verdict-card');
  card.className = 'verdict-card ' + cls;

  const pctEl = document.getElementById('verdict-pct');
  pctEl.textContent = scorePct + '%';
  pctEl.style.color = cls === 'fake' ? 'var(--red)' : cls === 'real' ? 'var(--green)' : 'var(--amber)';

  document.getElementById('verdict-title').textContent = title;
  document.getElementById('verdict-desc').textContent  = desc;

  /* Ring animation */
  const circle = document.getElementById('verdict-circle');
  const circ = 289;
  const offset = circ - (result.score * circ);
  circle.style.stroke = cls === 'fake' ? 'var(--red)' : cls === 'real' ? 'var(--green)' : 'var(--amber)';
  setTimeout(() => { circle.style.strokeDashoffset = offset; }, 120);

  /* Confidence band */
  document.getElementById('conf-fill').style.width = confPct + '%';
  document.getElementById('conf-val').textContent = confPct + '%';

  /* Flags */
  const flagsEl = document.getElementById('verdict-flags');
  const flags = buildFlags(cls, result.signals, result.type);
  flagsEl.innerHTML = flags.map(f =>
    `<span class="vflag ${f.cls}">${f.icon} ${f.text}</span>`
  ).join('');

  /* Model score cards */
  const typeLabel = { image: 'Image model', audio: 'Audio model', video: 'Lip sync / video' };
  const typeColor = { image: 'var(--accent)', audio: 'var(--amber)', video: 'var(--blue)' };
  const scoresGrid = document.getElementById('scores-grid');
  scoresGrid.innerHTML = `
    <div class="score-card">
      <div class="score-model">${typeLabel[result.type] || result.type} <span class="score-dot" style="background:${typeColor[result.type]}"></span></div>
      <div class="score-value" style="color:${typeColor[result.type]}">${scorePct}<span style="font-size:15px;color:var(--muted)">%</span></div>
      <div class="score-sublabel">fake probability</div>
      <div class="score-bar-bg"><div class="score-bar-fill" style="width:0%;background:${typeColor[result.type]}" data-w="${scorePct}%"></div></div>
      <div class="score-signals">
        ${Object.entries(result.signals).slice(0,3).map(([name, s]) => `
          <div class="score-signal-row">
            <span>${name}</span>
            <div class="score-signal-bar-bg"><div class="score-signal-bar-fill" style="width:0%;background:${typeColor[result.type]}" data-w="${Math.round(s.val*100)}%"></div></div>
            <span>${Math.round(s.val*100)}%</span>
          </div>`).join('')}
      </div>
    </div>
    <div class="score-card">
      <div class="score-model">Confidence <span class="score-dot" style="background:var(--green)"></span></div>
      <div class="score-value" style="color:var(--green)">${confPct}<span style="font-size:15px;color:var(--muted)">%</span></div>
      <div class="score-sublabel">model certainty</div>
      <div class="score-bar-bg"><div class="score-bar-fill" style="width:0%;background:var(--green)" data-w="${confPct}%"></div></div>
      <div style="margin-top:12px;font-size:11px;color:var(--muted);line-height:1.6">
        ${confPct >= 80 ? 'High confidence. Signals are strongly coherent.' :
          confPct >= 60 ? 'Medium confidence. Result is likely correct but verify.' :
                         'Low confidence. Treat verdict with caution.'}
      </div>
    </div>
    <div class="score-card">
      <div class="score-model">Signals detected <span class="score-dot" style="background:var(--muted)"></span></div>
      <div class="score-value" style="color:var(--text)">${Object.keys(result.signals).length}<span style="font-size:15px;color:var(--muted)">/${Object.keys(result.signals).length}</span></div>
      <div class="score-sublabel">forensic signals run</div>
      <div style="margin-top:14px;display:flex;flex-direction:column;gap:5px;">
        ${Object.entries(result.signals).map(([k,s]) => `
          <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--muted)">
            <span>${k}</span>
            <span style="color:${s.val>0.58?'var(--red)':s.val<0.42?'var(--green)':'var(--amber)'}">
              ${s.val > 0.58 ? 'high' : s.val < 0.42 ? 'low' : 'med'}
            </span>
          </div>`).join('')}
      </div>
    </div>
  `;

  /* Breakdown */
  const breakdownEl = document.getElementById('breakdown-rows');
  breakdownEl.innerHTML = Object.entries(result.signals).map(([name, s]) => {
    const pct = Math.round(s.val * 100);
    const c = s.val >= 0.58 ? 'var(--red)' : s.val < 0.42 ? 'var(--green)' : 'var(--amber)';
    return `
      <div class="breakdown-row">
        <span class="breakdown-name">${name}</span>
        <div class="breakdown-bar-bg">
          <div class="breakdown-bar-fill" style="width:0%;background:${c}" data-w="${pct}%"></div>
        </div>
        <span class="breakdown-val" style="color:${c}">${pct}%</span>
      </div>`;
  }).join('');

  /* Explain box */
  document.getElementById('explain-content').innerHTML = buildExplanation(result);

  /* Error notice if analysis had issues */
  const errEl = document.getElementById('results-error');
  if (result.error) {
    errEl.style.display = 'block';
    errEl.textContent = '⚠ Analysis note: ' + result.error + '. Result is approximate.';
  } else {
    errEl.style.display = 'none';
  }

  /* Animate all bars */
  setTimeout(() => {
    document.querySelectorAll('[data-w]').forEach(el => {
      el.style.width = el.dataset.w;
    });
  }, 200);

  showPage('results');
  showToast('Analysis complete');
}

/* ─── Build flags from signals ─── */
function buildFlags(cls, signals, type) {
  const flags = [];
  if (cls === 'fake') {
    flags.push({ cls: 'fake', icon: '⬤', text: 'DEEPFAKE DETECTED' });
    const topSignal = Object.entries(signals).sort((a,b) => b[1].val - a[1].val)[0];
    if (topSignal && topSignal[1].val > 0.6)
      flags.push({ cls: 'fake', icon: '▲', text: topSignal[0].toUpperCase() });
  } else if (cls === 'real') {
    flags.push({ cls: 'real', icon: '⬤', text: 'AUTHENTIC' });
    const lowSignals = Object.entries(signals).filter(([,s]) => s.val < 0.35);
    if (lowSignals.length > 2) flags.push({ cls: 'real', icon: '✓', text: 'NO ARTEFACTS' });
  } else {
    flags.push({ cls: 'uncertain', icon: '⬤', text: 'UNCERTAIN' });
    flags.push({ cls: 'neutral', icon: '→', text: 'MANUAL REVIEW' });
  }
  flags.push({ cls: 'neutral', icon: '#', text: type.toUpperCase() + ' ANALYSIS' });
  return flags;
}

/* ─── Build human-readable explanation ─── */
function buildExplanation(result) {
  const entries = Object.entries(result.signals);
  if (!entries.length) return '<p style="font-size:12px;color:var(--muted)">No signal detail available.</p>';

  const items = entries.map(([name, s]) => {
    const level = s.val >= 0.58 ? 'elevated' : s.val < 0.42 ? 'low' : 'moderate';
    const dot = s.val >= 0.58 ? 'var(--red)' : s.val < 0.42 ? 'var(--green)' : 'var(--amber)';
    const explanation = {
      'DCT high-freq':       `High-frequency DCT coefficients are ${level} — ${s.val>0.58?'consistent with GAN fingerprints':'consistent with natural image compression'}.`,
      'Noise uniformity':    `Pixel noise distribution is ${level} — ${s.val>0.58?'unnaturally uniform, a common synthetic image trait':'heterogeneous as expected in real photos'}.`,
      'Channel corr.':       `R/G/B channel correlation is ${level} — ${s.val>0.58?'asymmetric, possibly from blending':'within normal range for authentic media'}.`,
      'Gradient consist':    `Edge gradient consistency is ${level} — ${s.val>0.58?'too smooth, suggesting face-blend boundaries':'natural variation expected in real photos'}.`,
      'ELA surrogate':       `Compression sensitivity (ELA) is ${level} — ${s.val>0.58?'elevated, indicating post-processing':'normal for single-capture media'}.`,
      'Spectral flatness':   `Audio spectral flatness is ${level} — ${s.val>0.58?'higher than natural speech, typical of TTS':'consistent with genuine voice recording'}.`,
      'Silence purity':      `Silence segments are ${level} — ${s.val>0.58?'unnaturally clean, common in synthesised audio':'naturally noisy as expected'}.`,
      'Sub-band distribution':`Frequency sub-band energy is ${level} — ${s.val>0.58?'abnormally distributed vs natural voice':'within natural speech range'}.`,
      'Pitch periodicity':   `Pitch periodicity is ${level} — ${s.val>0.58?'overly regular, a hallmark of neural TTS':'naturally variable as in authentic speech'}.`,
      'HF dropout':          `High-frequency content dropout is ${level} — ${s.val>0.58?'suggests vocoder processing':'full spectrum intact, consistent with real audio'}.`,
      'Facial temporal var': `Facial brightness variation over time is ${level} — ${s.val>0.58?'inconsistent with natural video':'steady as expected in authentic footage'}.`,
      'Mouth motion ratio':  `Mouth-to-face motion ratio is ${level} — ${s.val>0.58?'mouth moves independently of face, suggesting lip-sync manipulation':'correlated movement consistent with authentic video'}.`,
      'Scene cut anomaly':   `Scene-cut sharpness is ${level} — ${s.val>0.58?'abrupt transitions may indicate face splicing':'natural transitions'}.`,
      'Colour consistency':  `Frame-to-frame colour consistency is ${level} — ${s.val>0.58?'inconsistencies suggest blending or compositing':'stable colour as in authentic video'}.`,
      'Motion smoothness':   `Motion autocorrelation is ${level} — ${s.val>0.58?'unnaturally smooth motion patterns':'natural motion dynamics'}.`,
    }[name] || `${name}: ${level} signal (${Math.round(s.val*100)}%).`;
    return `<div class="explain-item"><span class="explain-bullet" style="background:${dot}"></span><span>${explanation}</span></div>`;
  }).join('');

  return `<div class="explain-list">${items}</div>`;
}

/* ─── History ─── */
function renderHistory() {
  const el = document.getElementById('history-list');
  if (!State.history.length) {
    el.innerHTML = `<div style="text-align:center;padding:60px 0;color:var(--muted);font-size:14px">
      No analyses yet — <a href="#" onclick="showPage('analyze')" style="color:var(--accent)">start one</a>
    </div>`;
    return;
  }
  const icons = { image: '🖼', audio: '🎙', video: '🎬' };
  el.innerHTML = State.history.map((h, i) => {
    const pct = Math.round(h.score * 100);
    const cls = h.verdict.cls;
    const clsMap = { fake: 'hv-fake', real: 'hv-real', uncertain: 'hv-uncertain' };
    const labelMap = { fake: 'DEEPFAKE', real: 'AUTHENTIC', uncertain: 'UNCERTAIN' };
    const colourMap = { fake: 'var(--red)', real: 'var(--green)', uncertain: 'var(--amber)' };
    const dateStr = h.date.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
    return `
      <div class="history-item" onclick="reloadResult(${i})">
        <div class="history-icon">${icons[h.type] || '📄'}</div>
        <div>
          <div class="history-name">${h.filename}</div>
          <div class="history-meta">${dateStr} · ${h.type} · conf ${Math.round(h.confidence*100)}%</div>
        </div>
        <span class="history-verdict ${clsMap[cls]}">${labelMap[cls]}</span>
        <span class="history-score" style="color:${colourMap[cls]}">${pct}%</span>
      </div>`;
  }).join('');
}

function reloadResult(i) {
  const h = State.history[i];
  if (h) showResults(h);
}

/* ─── Export ─── */
function exportJSON() {
  if (!State.currentResult) return;
  const exportData = {
    tool: 'Deepfake Detection Model',
    filename: State.currentResult.filename,
    type: State.currentResult.type,
    score: Math.round(State.currentResult.score * 100) + '%',
    confidence: Math.round(State.currentResult.confidence * 100) + '%',
    verdict: State.currentResult.verdict.title,
    signals: Object.fromEntries(
      Object.entries(State.currentResult.signals).map(([k,v]) => [k, Math.round(v.val*100)+'%'])
    ),
    timestamp: State.currentResult.date.toISOString(),
  };
  const blob = new Blob([JSON.stringify(exportData, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'deepfake_detection_result.json';
  a.click();
  showToast('JSON exported');
}

function exportCSV() {
  if (!State.currentResult) return;
  const r = State.currentResult;
  const rows = [
    ['Signal', 'Score (%)'],
    ...Object.entries(r.signals).map(([k,v]) => [k, Math.round(v.val*100)]),
    ['Overall', Math.round(r.score*100)],
    ['Confidence', Math.round(r.confidence*100)],
  ];
  const csv = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'deepfake_detection_result.csv';
  a.click();
  showToast('CSV exported');
}

/* ─── Toast ─── */
function showToast(msg, type = 'ok') {
  const t   = document.getElementById('toast');
  const dot = t.querySelector('.toast-dot');
  const txt = document.getElementById('toast-msg');
  dot.className = 'toast-dot' + (type === 'warn' ? ' warn' : type === 'err' ? ' err' : '');
  txt.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

/* ─── Drag-and-drop ─── */
['image', 'audio', 'video'].forEach(type => {
  const area = document.getElementById('upload-' + type);
  if (!area) return;
  area.addEventListener('dragover', e => { e.preventDefault(); area.classList.add('drag'); });
  area.addEventListener('dragleave', () => area.classList.remove('drag'));
  area.addEventListener('drop', e => {
    e.preventDefault(); area.classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const input = document.getElementById('file-' + type);
    const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
    handleFile(type, input);
  });
});

// Expose globals needed by inline HTML onclick handlers
window.showPage    = showPage;
window.switchTab   = switchTab;
window.handleFile  = handleFile;
window.toggleModel = toggleModel;
window.runAnalysis = runAnalysis;
window.reloadResult= reloadResult;
window.exportJSON  = exportJSON;
window.exportCSV   = exportCSV;
window.showToast   = showToast;
