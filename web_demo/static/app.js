/* ═══════════════ SONA web demo — frontend ═══════════════ */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const EQ_FREQS = ["31", "63", "125", "250", "500", "1k", "2k", "4k"];
const SCENES = ["speech", "music", "traffic", "fan", "crowd", "mixed", "unknown"];
const MODE_LABELS = { aggressive_speech: "Aggressive" };
const MODE_ICONS = {
  speech:  '<svg class="ic" viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>',
  aggressive_speech: '<svg class="ic" viewBox="0 0 24 24"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/></svg>',
  music:   '<svg class="ic" viewBox="0 0 24 24"><path d="M9 18V6l10-2v12M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0zm10-2a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"/></svg>',
  podcast: '<svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="11" r="3"/><path d="M7.5 15.5a6 6 0 1 1 9 0M4.6 18.4a10 10 0 1 1 14.8 0M12 14v7"/></svg>',
  gaming:  '<svg class="ic" viewBox="0 0 24 24"><path d="M6 9h12a4 4 0 0 1 4 4l-1 5a2 2 0 0 1-3.4 1L15 16H9l-2.6 3A2 2 0 0 1 3 18l-1-5a4 4 0 0 1 4-4z"/><path d="M8 12v2m-1-1h2m6-1h.01M18 13h.01"/></svg>',
  hearing: '<svg class="ic" viewBox="0 0 24 24"><path d="M6 9a6 6 0 1 1 11 3c-1 1.7-2.5 2.3-3 4-.4 1.4-1 3-3 3"/><path d="M9.5 9a2.5 2.5 0 0 1 4.5 1.3"/></svg>',
  studio:  '<svg class="ic" viewBox="0 0 24 24"><path d="M5 4v7m0 4v5M12 4v3m0 4v9M19 4v11m0 4v1"/><circle cx="5" cy="13" r="2"/><circle cx="12" cy="9" r="2"/><circle cx="19" cy="17" r="2"/></svg>',
};

const S = {
  presets: null,
  fileId: null, fileInfo: null,
  outputId: null, outputInfo: null, analytics: null,
  resInfo: null,
  scene: null, sceneConf: null, sceneManual: false,
  mode: null,               // null = auto (scene-routed)
  dirty: false,             // control values touched → send explicit overrides
  ab: "a",
  busy: false,
  playhead: 0,              // 0..1 for waveform overlay
};

/* ── tiny utils ─────────────────────────────────────────── */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `request failed (${res.status})`);
  return data;
}

function toast(msg, type = "err", ms = 4200) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  $("#toast-holder").appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 350); }, ms);
}

const fmtTime = (s) => {
  if (!isFinite(s)) return "0:00";
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
};

function setRangeFill(input) {
  const min = +input.min, max = +input.max, v = +input.value;
  input.style.setProperty("--fill", `${((v - min) / (max - min)) * 100}%`);
}

/* ── loading screen ─────────────────────────────────────── */

const LOAD_STEPS = [
  ["Initializing runtime…", 8],
  ["Loading GTCRN engine…", 34],
  ["Restoring Layer A checkpoint…", 58],
  ["Preparing controller…", 76],
  ["Calibrating scene memory…", 90],
  ["Ready", 100],
];

async function runLoader() {
  const fill = $("#loader-fill"), status = $("#loader-status");
  let engineReady = false;
  const poll = setInterval(async () => {
    try {
      const h = await api("/api/health");
      updateEnginePill(h);
      if (h.engine === "ready" || h.engine === "error" || h.engine === "cold") engineReady = true;
    } catch { /* server starting */ }
  }, 700);

  for (const [msg, pct] of LOAD_STEPS) {
    status.textContent = msg;
    fill.style.width = `${pct}%`;
    if (pct === 100) break;
    await new Promise((r) => setTimeout(r, 520));
  }
  // hold "Ready" until the engine reports in (max ~20 s)
  for (let i = 0; i < 28 && !engineReady; i++) await new Promise((r) => setTimeout(r, 700));
  clearInterval(poll);
  $("#loader").classList.add("done");
  $("#app").hidden = false;
  requestAnimationFrame(() => { drawAll(); });
}

function updateEnginePill(h) {
  const pill = $("#engine-pill"), text = $("#engine-pill-text");
  pill.className = "pill";
  if (h.engine === "ready") {
    pill.classList.add("pill-ready");
    text.textContent = `Engine ready · ${(h.device || "cpu").toUpperCase()}`;
  } else if (h.engine === "error") {
    pill.classList.add("pill-error");
    text.textContent = "Engine error — see server log";
  } else if (h.engine === "loading") {
    pill.classList.add("pill-loading");
    text.textContent = "Loading GTCRN…";
  } else {
    pill.classList.add("pill-loading");
    text.textContent = "Engine idle · loads on first run";
  }
}

/* ── canvas helpers ─────────────────────────────────────── */

function ctx2d(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
  // cache the design height: canvas.height assignment below clobbers the attribute
  if (!canvas.dataset.h) canvas.dataset.h = canvas.getAttribute("height") || 120;
  const h = +canvas.dataset.h;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function drawWave(canvas, wf, color, playhead = null, scalePeak = 1.0) {
  const { ctx, w, h } = ctx2d(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!wf || !wf.bins) return;
  const mid = h / 2, n = wf.bins, bw = w / n;
  const amp = (mid - 6) / Math.max(scalePeak, 0.01);
  // center line
  ctx.strokeStyle = "rgba(139,148,167,.15)";
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color[0]); grad.addColorStop(1, color[1]);
  ctx.fillStyle = grad;
  for (let i = 0; i < n; i++) {
    const y1 = mid - Math.min(Math.max(wf.maxs[i] * amp, -mid + 4), mid - 4);
    const y2 = mid - Math.min(Math.max(wf.mins[i] * amp, -mid + 4), mid - 4);
    ctx.fillRect(i * bw, Math.min(y1, y2), Math.max(bw - 0.6, 0.6), Math.max(Math.abs(y2 - y1), 1));
  }
  if (playhead !== null && playhead >= 0) {
    const x = playhead * w;
    ctx.fillStyle = "rgba(232,237,244,.9)";
    ctx.fillRect(x - 0.75, 0, 1.5, h);
    ctx.fillStyle = "rgba(232,237,244,.12)";
    ctx.fillRect(0, 0, x, h);
  }
}

function drawSpectrum(canvas) {
  const { ctx, w, h } = ctx2d(canvas);
  ctx.clearRect(0, 0, w, h);
  const specIn = S.fileInfo?.spectrum, specOut = S.outputInfo?.spectrum;
  if (!specIn) {
    ctx.fillStyle = "rgba(90,98,116,.8)";
    ctx.font = "12px Consolas, monospace"; ctx.textAlign = "center";
    ctx.fillText("select input audio to view spectrum", w / 2, h / 2);
    return;
  }
  const specRes = S.resInfo?.spectrum;
  const all = [
    ...specIn.levels_db, ...(specIn.floor_db || []),
    ...(specOut ? [...specOut.levels_db, ...(specOut.floor_db || [])] : []),
    ...(specRes ? specRes.levels_db : []),
  ];
  const maxDb = Math.max(...all), minDb = Math.max(Math.min(...all) - 3, maxDb - 90);
  const range = Math.max(maxDb - minDb, 1);
  const n = specIn.levels_db.length, bw = w / n;
  const toY = (db) => h - 18 - ((db - minDb) / range) * (h - 30);

  // grid
  ctx.strokeStyle = "rgba(28,35,49,.9)";
  for (let gy = 1; gy <= 3; gy++) {
    const y = (h - 18) * gy / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  // input bars
  ctx.fillStyle = "rgba(255,180,84,.32)";
  for (let i = 0; i < n; i++) {
    const y = toY(specIn.levels_db[i]);
    ctx.fillRect(i * bw + 0.5, y, bw - 1.5, h - 18 - y);
  }
  // residual line (what was removed)
  if (specRes) {
    ctx.beginPath();
    ctx.moveTo(0, toY(specRes.levels_db[0]));
    for (let i = 0; i < n; i++) ctx.lineTo(i * bw + bw / 2, toY(specRes.levels_db[i]));
    ctx.lineTo(w, toY(specRes.levels_db[n - 1]));
    ctx.strokeStyle = "rgba(255,107,122,.85)"; ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);
  }
  // output line + fill
  if (specOut) {
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(56,225,198,.4)"); grad.addColorStop(1, "rgba(56,225,198,.03)");
    ctx.beginPath();
    ctx.moveTo(0, toY(specOut.levels_db[0]));
    for (let i = 0; i < n; i++) ctx.lineTo(i * bw + bw / 2, toY(specOut.levels_db[i]));
    ctx.lineTo(w, toY(specOut.levels_db[n - 1]));
    ctx.strokeStyle = "#38e1c6"; ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.lineTo(w, h - 18); ctx.lineTo(0, h - 18); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
  }
  // noise-floor curves (10th percentile per band = steady background noise)
  if (specIn.floor_db && specOut?.floor_db) {
    const floorLine = (fdb, color) => {
      ctx.beginPath();
      ctx.moveTo(0, toY(fdb[0]));
      for (let i = 0; i < n; i++) ctx.lineTo(i * bw + bw / 2, toY(fdb[i]));
      ctx.strokeStyle = color; ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]); ctx.stroke(); ctx.setLineDash([]);
    };
    floorLine(specIn.floor_db, "rgba(255,180,84,.55)");
    floorLine(specOut.floor_db, "rgba(56,225,198,.55)");
  }
  // background-noise reduction annotation, measured on the noise floor >= 2 kHz
  const note = $("#hf-note");
  if (note) {
    if (specOut?.floor_db && specIn.floor_db) {
      let sum = 0, count = 0;
      for (let i = 0; i < n; i++) {
        if (specIn.freqs_hz[i] >= 2000) { sum += specIn.floor_db[i] - specOut.floor_db[i]; count++; }
      }
      const hf = count ? sum / count : 0;
      note.textContent = `background noise floor above 2 kHz: ${hf >= 0 ? "−" : "+"}${Math.abs(hf).toFixed(1)} dB (dotted = noise floor)`;
    } else {
      note.textContent = "avg magnitude · 50 Hz – 8 kHz · log";
    }
  }
  // freq labels
  ctx.fillStyle = "rgba(90,98,116,.9)"; ctx.font = "9.5px Consolas, monospace"; ctx.textAlign = "center";
  [100, 500, 1000, 2000, 4000, 8000].forEach((f) => {
    const idx = specIn.freqs_hz.findIndex((v) => v >= f);
    if (idx < 0) return;
    ctx.fillText(f >= 1000 ? `${f / 1000}k` : f, idx * bw, h - 5);
  });
}

let abSplit = 0.5;
function drawAB(canvas) {
  const { ctx, w, h } = ctx2d(canvas);
  ctx.clearRect(0, 0, w, h);
  const wfIn = S.fileInfo?.waveform, wfOut = S.outputInfo?.waveform;
  if (!wfIn || !wfOut) {
    ctx.fillStyle = "rgba(90,98,116,.8)";
    ctx.font = "12px Consolas, monospace"; ctx.textAlign = "center";
    ctx.fillText(wfIn ? "enhance audio to compare" : "select input audio first", w / 2, h / 2);
    return;
  }
  const mid = h / 2, n = wfIn.bins, bw = w / n, split = abSplit * w;
  for (let i = 0; i < n; i++) {
    const x = i * bw, left = x < split;
    const wf = left ? wfIn : wfOut;
    const y1 = mid - wf.maxs[i] * (mid - 8);
    const y2 = mid - wf.mins[i] * (mid - 8);
    ctx.fillStyle = left ? "rgba(255,180,84,.75)" : "rgba(56,225,198,.85)";
    ctx.fillRect(x, Math.min(y1, y2), Math.max(bw - 0.6, 0.6), Math.max(Math.abs(y2 - y1), 1));
  }
  // divider
  ctx.fillStyle = "rgba(232,237,244,.85)";
  ctx.fillRect(split - 1, 0, 2, h);
  ctx.font = "9.5px Consolas, monospace";
  ctx.fillStyle = "rgba(255,180,84,.9)"; ctx.textAlign = "right";
  ctx.fillText("NOISY", split - 8, 14);
  ctx.fillStyle = "rgba(56,225,198,.9)"; ctx.textAlign = "left";
  ctx.fillText("ENHANCED", split + 8, 14);
}

function waveScales() {
  // shared absolute scale by default; per-signal peaks only in "normalized view"
  const inP = S.fileInfo?.peak || 1, outP = S.outputInfo?.peak || 0, resP = S.resInfo?.peak || 0;
  if ($("#norm-view")?.checked) {
    return { in: Math.max(inP, 0.01), out: Math.max(outP, 0.01), res: Math.max(resP, 0.01) };
  }
  const shared = Math.max(inP, outP, resP, 0.01);
  return { in: shared, out: shared, res: shared };
}

function drawAll() {
  const sc = waveScales();
  drawWave($("#wave-in"), S.fileInfo?.waveform, ["#ffb454", "#c97a2b"], S.fileInfo ? S.playhead : null, sc.in);
  drawWave($("#wave-out"), S.outputInfo?.waveform, ["#38e1c6", "#2b8fc9"], S.outputInfo ? S.playhead : null, sc.out);
  drawWave($("#wave-res"), S.resInfo?.waveform, ["#ff6b7a", "#7c5cff"], S.resInfo ? S.playhead : null, sc.res);
  $("#wave-out-empty").style.display = S.outputInfo ? "none" : "grid";
  $("#wave-res-empty").style.display = S.resInfo ? "none" : "grid";
  const a = S.analytics;
  $("#res-summary").textContent = a
    ? `residual RMS ${a.residual_rms_db} dBFS · ≈${a.noise_removed_pct}% of input energy removed`
    : "";
  drawSpectrum($("#spec-canvas"));
  drawAB($("#ab-canvas"));
}
window.addEventListener("resize", () => drawAll());

/* ── input handling ─────────────────────────────────────── */

function setFile(info) {
  S.fileId = info.file_id;
  S.fileInfo = info;
  S.outputId = null; S.outputInfo = null; S.analytics = null; S.resInfo = null;
  // low-noise heuristic: high activity-to-floor ratio means little steady noise to remove
  $("#low-noise-note").hidden = !(info.est_snr_db != null && info.est_snr_db > 20);
  S.scene = null; S.sceneConf = null; S.sceneManual = false;
  stopPlayback();
  $("#players").hidden = true;
  $("#file-card").hidden = false;
  $("#file-name").textContent = info.name;
  $("#file-detail").textContent =
    `${info.duration_s.toFixed(2)} s · ${(info.sample_rate / 1000).toFixed(0)} kHz mono · peak ${info.peak_db} dBFS`;
  $("#btn-enhance").disabled = false;
  $("#btn-classify").disabled = false;
  $("#scene-note").textContent = "or enable auto-detect";
  setScene(null, null, false);
  resetPipeline();
  renderAnalytics();
  drawAll();
  const audioA = $("#audio-a");
  audioA.src = `/api/audio/input/${S.fileId}`;
  $("#audio-b").removeAttribute("src");
  $("#audio-n").removeAttribute("src");
}

async function useSample() {
  try {
    const info = await api("/api/sample", { method: "POST" });
    setFile(info);
    toast("Loaded bundled sample — real speech + babble noise @ 5 dB SNR", "ok", 3200);
  } catch (e) { toast(e.message); }
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  try {
    $("#dropzone").classList.add("drag");
    const info = await api("/api/upload", { method: "POST", body: fd });
    setFile(info);
    toast(`Loaded ${info.name}`, "ok", 2600);
  } catch (e) { toast(e.message); }
  finally { $("#dropzone").classList.remove("drag"); }
}

function wireInput() {
  const dz = $("#dropzone"), fi = $("#file-input");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => { if (fi.files[0]) uploadFile(fi.files[0]); fi.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f); });
  $("#btn-sample").addEventListener("click", useSample);
  $("#btn-clear-file").addEventListener("click", () => {
    S.fileId = null; S.fileInfo = null; S.outputId = null; S.outputInfo = null; S.resInfo = null; S.analytics = null;
    stopPlayback();
    $("#file-card").hidden = true;
    $("#low-noise-note").hidden = true;
    $("#players").hidden = true;
    $("#btn-enhance").disabled = true;
    $("#btn-classify").disabled = true;
    setScene(null, null, false);
    drawAll();
  });
}

/* ── scene ──────────────────────────────────────────────── */

function setScene(label, conf, manual) {
  S.scene = label; S.sceneConf = conf; S.sceneManual = manual;
  $("#scene-label").textContent = label || "—";
  $("#scene-conf").textContent =
    label ? (manual ? "manual" : `${Math.round(conf * 100)}% conf`) : "no scan";
  const circ = 2 * Math.PI * 34;
  $("#ring-val").style.strokeDashoffset = label && !manual ? circ * (1 - conf) : (label ? 0 : circ);
  $$(".scene-chip").forEach((c) => {
    c.classList.toggle("detected", !manual && c.dataset.scene === label);
    c.classList.toggle("manual", manual && c.dataset.scene === label);
  });
  updateMemButtons();
  updateModeSource();
}

async function classifyNow() {
  if (!S.fileId) return;
  try {
    $("#btn-classify").disabled = true;
    const r = await api("/api/classify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file_id: S.fileId }) });
    setScene(r.scene, r.confidence, false);
    toast(`Scene: ${r.scene} (${Math.round(r.confidence * 100)}%) → suggests ${r.suggested_mode} mode`, "ok", 3400);
  } catch (e) { toast(e.message); }
  finally { $("#btn-classify").disabled = !S.fileId; }
}

function wireScene() {
  const holder = $("#scene-chips");
  SCENES.forEach((sc) => {
    const b = document.createElement("button");
    b.className = "scene-chip"; b.dataset.scene = sc; b.textContent = sc;
    b.addEventListener("click", () => {
      if (S.sceneManual && S.scene === sc) setScene(null, null, false);
      else setScene(sc, null, true);
    });
    holder.appendChild(b);
  });
  $("#btn-classify").addEventListener("click", classifyNow);
}

/* ── modes ──────────────────────────────────────────────── */

function updateModeSource() {
  const el = $("#mode-source");
  if (S.mode) el.textContent = `manual · ${S.mode}`;
  else if (S.scene) el.textContent = `auto · ${S.presets?.scene_to_mode?.[S.scene] || "speech"} (from scene)`;
  else el.textContent = "auto · scene-routed";
}

function applyPresetToControls(settings) {
  $("#sl-denoise").value = settings.mask_strength;
  $("#sl-floor").value = settings.safety_floor;
  $("#sl-thresh").value = settings.compressor_threshold_db;
  $("#sl-ratio").value = settings.compressor_ratio;
  settings.eq_gains_db.forEach((g, i) => { const el = $(`#eq-${i}`); if (el) el.value = g; });
  $("#bypass-eq").checked = !!settings.bypass_eq;
  $("#bypass-comp").checked = !!settings.bypass_compressor;
  $("#limiter-ceiling").textContent = (+settings.limiter_ceiling).toFixed(2);
  refreshControlLabels();
}

function selectMode(name) {
  S.mode = S.mode === name ? null : name;
  $$(".mode-card").forEach((c) => c.classList.toggle("active", c.dataset.mode === S.mode));
  if (S.mode && S.presets) {
    applyPresetToControls(S.presets.modes[S.mode].settings);
    S.dirty = false;
  }
  updateModeSource();
}

function wireModes() {
  const grid = $("#mode-grid");
  Object.keys(MODE_ICONS).forEach((name) => {
    const b = document.createElement("button");
    b.className = "mode-card"; b.dataset.mode = name;
    const label = MODE_LABELS[name] || name[0].toUpperCase() + name.slice(1);
    b.innerHTML = `${MODE_ICONS[name]}<span>${label}</span>`;
    b.title = "";
    b.addEventListener("click", () => selectMode(name));
    grid.appendChild(b);
  });
}

/* ── controls ───────────────────────────────────────────── */

function refreshControlLabels() {
  $("#val-denoise").textContent = (+$("#sl-denoise").value).toFixed(2);
  $("#val-floor").textContent = (+$("#sl-floor").value).toFixed(2);
  $("#val-thresh").textContent = `${$("#sl-thresh").value} dB`;
  $("#val-ratio").textContent = `${(+$("#sl-ratio").value).toFixed(1)} : 1`;
  $$("input[type=range]").forEach(setRangeFill);
  EQ_FREQS.forEach((_, i) => {
    const v = +$(`#eq-${i}`).value;
    $(`#eq-val-${i}`).textContent = v === 0 ? "0" : (v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1));
  });
}

function currentControls() {
  return {
    denoise_strength: +$("#sl-denoise").value,
    safety_floor: +$("#sl-floor").value,
    eq_gains_db: EQ_FREQS.map((_, i) => +$(`#eq-${i}`).value),
    compressor_threshold_db: +$("#sl-thresh").value,
    compressor_ratio: +$("#sl-ratio").value,
    bypass_eq: $("#bypass-eq").checked,
    bypass_compressor: $("#bypass-comp").checked,
  };
}

function wireControls() {
  const eq = $("#eq");
  EQ_FREQS.forEach((f, i) => {
    const band = document.createElement("div");
    band.className = "eq-band";
    band.innerHTML = `
      <span class="eq-val" id="eq-val-${i}">0</span>
      <input type="range" id="eq-${i}" min="-12" max="12" step="0.5" value="0" orient="vertical">
      <span class="eq-freq">${f}</span>`;
    eq.appendChild(band);
  });
  $$("#panel-controls input[type=range]").forEach((sl) => {
    sl.addEventListener("input", () => { S.dirty = true; refreshControlLabels(); });
  });
  ["#bypass-eq", "#bypass-comp"].forEach((sel) => {
    $(sel).addEventListener("change", () => { S.dirty = true; });
  });
  $("#btn-reset-controls").addEventListener("click", () => {
    const preset = S.presets?.modes?.[S.mode || "speech"]?.settings;
    if (preset) applyPresetToControls(preset);
    S.dirty = false;
    toast(`Controls reset to ${S.mode || "speech"} preset`, "ok", 2200);
  });
  refreshControlLabels();
}

/* ── pipeline animation ─────────────────────────────────── */

const STAGE_ORDER = ["scene_analysis", "gtcrn_enhancement", "mode_controller", "eq_compressor", "export_wav"];
let stageTimer = null;

function resetPipeline() {
  clearTimeout(stageTimer);
  $$(".stage").forEach((s) => s.classList.remove("active", "done"));
}

function animatePipeline() {
  resetPipeline();
  const durations = [500, 2200, 450, 900, 600]; // optimistic pacing, ms
  let i = 0;
  const step = () => {
    if (i > 0) stageEl(STAGE_ORDER[i - 1])?.classList.replace("active", "done");
    if (i >= STAGE_ORDER.length || !S.busy) return;
    stageEl(STAGE_ORDER[i])?.classList.add("active");
    // hold on the last stage until the response arrives
    if (i < STAGE_ORDER.length - 1) stageTimer = setTimeout(step, durations[i]);
    i += 1;
  };
  step();
}

const stageEl = (name) => document.querySelector(`.stage[data-stage="${name}"]`);

function finishPipeline() {
  clearTimeout(stageTimer);
  $$(".stage").forEach((s) => { s.classList.remove("active"); s.classList.add("done"); });
}

/* ── enhancement ────────────────────────────────────────── */

async function enhance() {
  if (!S.fileId || S.busy) return;
  S.busy = true;
  const btn = $("#btn-enhance");
  btn.classList.add("working");
  $("#btn-enhance-text").textContent = "Processing…";
  animatePipeline();
  stopPlayback();

  const autoScene = $("#auto-scene").checked;
  const payload = {
    file_id: S.fileId,
    mode: S.mode,
    auto_scene: autoScene && !S.sceneManual,
    scene: S.sceneManual ? S.scene : (autoScene ? null : S.scene),
    use_memory: $("#use-memory").checked,
    normalize_output: $("#normalize-out").checked,
    remember: false,
  };
  if (S.dirty) Object.assign(payload, currentControls());

  try {
    const r = await api("/api/enhance", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    S.outputId = r.output_id;
    S.outputInfo = r.output;
    S.resInfo = r.residual || null;
    S.analytics = r.analytics;
    finishPipeline();
    if (r.analytics.noise_removed_pct != null && r.analytics.noise_removed_pct < 3) {
      toast("Low-noise input detected; enhancement may be subtle.", "ok", 4200);
    }

    if (r.analytics.auto_scene) setScene(r.analytics.scene, r.analytics.scene_confidence, false);
    if (r.analytics.settings) { applyPresetToControls(r.analytics.settings); S.dirty = false; }
    $("#limiter-led").classList.toggle("on", !!r.analytics.limiter_engaged);
    $("#limiter-state").textContent = r.analytics.limiter_engaged ? "engaged" : "armed";
    $("#limiter-ceiling").textContent = (+r.analytics.limiter_ceiling).toFixed(2);

    // players
    $("#players").hidden = false;
    $("#audio-b").src = `/api/audio/output/${S.outputId}`;
    $("#audio-n").src = `/api/audio/residual/${S.outputId}`;
    $("#btn-download").href = `/api/download/${S.outputId}`;
    setAB("b");

    renderAnalytics();
    drawAll();
    toast(`Enhanced in ${r.analytics.processing_time_s}s (${r.analytics.realtime_factor}× realtime)`, "ok", 3600);
  } catch (e) {
    resetPipeline();
    toast(e.message);
  } finally {
    S.busy = false;
    btn.classList.remove("working");
    $("#btn-enhance-text").textContent = "Enhance Audio";
  }
}

/* ── playback / A-B ─────────────────────────────────────── */

const audioA = () => $("#audio-a"), audioB = () => $("#audio-b"), audioN = () => $("#audio-n");
const activeAudio = () => (S.ab === "a" ? audioA() : S.ab === "n" ? audioN() : audioB());
const AB_LABELS = { a: "A · ORIGINAL", b: "B · ENHANCED", n: "N · REMOVED NOISE" };
const AB_COLORS = { a: "var(--amber)", b: "var(--teal)", n: "var(--red)" };
let rafId = null;

function setAB(which) {
  const prev = activeAudio();
  const wasPlaying = !prev.paused && prev.src;
  const t = prev.currentTime || 0;
  prev.pause();
  S.ab = which;
  $$("#ab-toggle .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.ab === which));
  const lbl = $("#ab-label");
  lbl.textContent = AB_LABELS[which];
  lbl.style.color = AB_COLORS[which];
  const cur = activeAudio();
  if (cur.src) {
    cur.currentTime = t;
    if (wasPlaying) cur.play().catch(() => {});
  }
}

function stopPlayback() {
  [audioA(), audioB(), audioN()].forEach((a) => {
    a.pause();
    if (a.src) try { a.currentTime = 0; } catch {}
  });
  S.playhead = 0;
  updateTransport();
  setPlayIcon(false);
}

function setPlayIcon(playing) {
  $("#play-icon").hidden = playing;
  $("#pause-icon").hidden = !playing;
}

function updateTransport() {
  const a = activeAudio();
  const dur = a.duration || S.fileInfo?.duration_s || 0;
  const cur = a.currentTime || 0;
  $("#transport-fill").style.width = dur ? `${(cur / dur) * 100}%` : "0%";
  $("#time-cur").textContent = fmtTime(cur);
  $("#time-total").textContent = fmtTime(dur);
  S.playhead = dur ? cur / dur : 0;
}

function tick() {
  updateTransport();
  const sc = waveScales();
  drawWave($("#wave-in"), S.fileInfo?.waveform, ["#ffb454", "#c97a2b"], S.playhead, sc.in);
  drawWave($("#wave-out"), S.outputInfo?.waveform, ["#38e1c6", "#2b8fc9"], S.outputInfo ? S.playhead : null, sc.out);
  drawWave($("#wave-res"), S.resInfo?.waveform, ["#ff6b7a", "#7c5cff"], S.resInfo ? S.playhead : null, sc.res);
  if (!activeAudio().paused) rafId = requestAnimationFrame(tick);
  else setPlayIcon(false);
}

function wirePlayback() {
  $("#btn-play").addEventListener("click", () => {
    const a = activeAudio();
    if (!a.src) return;
    if (a.paused) { a.play().catch(() => {}); setPlayIcon(true); cancelAnimationFrame(rafId); rafId = requestAnimationFrame(tick); }
    else { a.pause(); setPlayIcon(false); }
  });
  $$("#ab-toggle .seg-btn").forEach((b) => b.addEventListener("click", () => setAB(b.dataset.ab)));
  [audioA(), audioB(), audioN()].forEach((a) => a.addEventListener("ended", () => { setPlayIcon(false); updateTransport(); }));
  $("#transport-track").addEventListener("click", (e) => {
    const a = activeAudio();
    if (!a.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    a.currentTime = ((e.clientX - rect.left) / rect.width) * a.duration;
    updateTransport(); drawAll();
  });
  // click waveforms to seek
  ["#wave-in", "#wave-out"].forEach((sel) => $(sel).addEventListener("click", (e) => {
    const a = activeAudio();
    if (!a.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    a.currentTime = ((e.clientX - rect.left) / rect.width) * a.duration;
    updateTransport(); drawAll();
  }));
}

/* ── A/B slider viz ─────────────────────────────────────── */

function wireABCanvas() {
  const cv = $("#ab-canvas");
  let dragging = false;
  const move = (e) => {
    const rect = cv.getBoundingClientRect();
    abSplit = Math.min(0.98, Math.max(0.02, (e.clientX - rect.left) / rect.width));
    drawAB(cv);
  };
  cv.addEventListener("mousedown", (e) => { dragging = true; move(e); });
  window.addEventListener("mousemove", (e) => { if (dragging) move(e); });
  window.addEventListener("mouseup", () => { dragging = false; });
}

/* ── visualization tabs ─────────────────────────────────── */

function wireVizTabs() {
  $$("#viz-tabs .seg-btn").forEach((b) => b.addEventListener("click", () => {
    $$("#viz-tabs .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
    $("#viz-wave").hidden = b.dataset.viz !== "wave";
    $("#viz-res").hidden = b.dataset.viz !== "res";
    $("#viz-spec").hidden = b.dataset.viz !== "spec";
    $("#viz-ab").hidden = b.dataset.viz !== "ab";
    drawAll();
  }));
  $("#norm-view").addEventListener("change", drawAll);
}

/* ── analytics ──────────────────────────────────────────── */

function statTile(k, v, cls = "") {
  return `<div class="stat"><div class="stat-k">${k}</div><div class="stat-v ${cls}">${v}</div></div>`;
}

function renderAnalytics() {
  const g = $("#stats-grid"), st = $("#stage-times"), op = $("#output-path");
  const a = S.analytics;
  if (!a) {
    g.innerHTML = S.fileInfo
      ? statTile("Input duration", `${S.fileInfo.duration_s.toFixed(2)}<small> s</small>`)
        + statTile("Input peak", `${S.fileInfo.peak_db}<small> dBFS</small>`)
        + statTile("Input RMS", `${S.fileInfo.rms_db}<small> dBFS</small>`)
      : "";
    st.innerHTML = "";
    op.hidden = true;
    $("#analytics-empty").hidden = !!S.fileInfo;
    return;
  }
  $("#analytics-empty").hidden = true;
  const rtfCls = a.realtime_factor < 1 ? "good" : "warn";
  const conf = a.scene_confidence != null ? ` <small>${Math.round(a.scene_confidence * 100)}%</small>` : "";
  g.innerHTML =
    statTile("Input duration", `${a.input_duration_s.toFixed(2)}<small> s</small>`) +
    statTile("Output duration", `${a.output_duration_s.toFixed(2)}<small> s</small>`) +
    statTile("Scene", `${a.scene || "—"}${conf}`, "good") +
    statTile("Mode", a.mode, "good") +
    statTile("Denoise strength", a.denoise_strength.toFixed(2)) +
    statTile("Safety floor", a.safety_floor.toFixed(2)) +
    statTile("Processing time", `${a.processing_time_s}<small> s</small>`) +
    statTile("Realtime factor", `${a.realtime_factor}×`, rtfCls) +
    statTile("Input peak", `${a.input_peak.toFixed(3)}`) +
    statTile("Output peak", `${a.output_peak.toFixed(3)}`) +
    statTile("Input RMS", `${a.input_rms_db}<small> dBFS</small>`) +
    statTile("Output RMS", `${a.output_rms_db}<small> dBFS</small>`) +
    statTile("Residual RMS", `${a.residual_rms_db}<small> dBFS</small>`) +
    statTile("Noise removed", `≈${a.noise_removed_pct}<small> % energy</small>`,
             a.noise_removed_pct >= 3 ? "good" : "warn") +
    statTile("Loudness Δ", `${a.loudness_change_db > 0 ? "+" : ""}${a.loudness_change_db}<small> dB</small>`,
             Math.abs(a.loudness_change_db) > 8 ? "warn" : "") +
    statTile("Limiter", a.limiter_engaged ? "engaged" : "clean", a.limiter_engaged ? "warn" : "good") +
    statTile("Memory", a.memory_used ? "applied" : (a.remembered ? "saved" : "—")) +
    statTile("Engine", `${a.device.toUpperCase()}<small> · ${a.checkpoint.split("_")[0]}</small>`);
  const total = a.stages.reduce((s, x) => s + x.seconds, 0) || 1;
  const names = { scene_analysis: "Scene analysis", mode_controller: "Mode controller", gtcrn_enhancement: "GTCRN enhancement", eq_compressor: "EQ / Compressor", export_wav: "Export WAV" };
  st.innerHTML = a.stages.map((s) => `
    <div class="st-row">
      <span class="st-name">${names[s.stage] || s.stage}</span>
      <div class="st-bar"><div class="st-fill" data-w="${(s.seconds / total) * 100}"></div></div>
      <span class="st-val">${(s.seconds * 1000).toFixed(0)} ms</span>
    </div>`).join("");
  requestAnimationFrame(() => $$(".st-fill").forEach((f) => { f.style.width = `${f.dataset.w}%`; }));
  op.hidden = false;
  $("#output-path-text").textContent = a.output_path;
}

/* ── preference memory ──────────────────────────────────── */

function updateMemButtons() {
  const has = !!S.scene;
  $("#btn-remember").disabled = !has;
  $("#btn-restore").disabled = !has;
  $("#btn-forget").disabled = !has;
}

async function refreshMemList() {
  try {
    const r = await api("/api/preferences");
    const entries = Object.entries(r.preferences || {});
    const list = $("#mem-list");
    if (!entries.length) { list.innerHTML = '<span class="muted-note">no scenes remembered yet</span>'; return; }
    list.innerHTML = entries.map(([scene, rec]) => `
      <div class="mem-item">
        <span class="mem-scene">${scene}</span>
        <span class="mem-mode">${rec.mode} · dn ${(+rec.settings.mask_strength).toFixed(2)}</span>
        <span class="mem-date">${(rec.updated_at || "").slice(0, 10)}</span>
        <button class="mem-x" data-scene="${scene}" title="forget">✕</button>
      </div>`).join("");
    $$(".mem-x").forEach((b) => b.addEventListener("click", async () => {
      try { await api(`/api/preferences/${b.dataset.scene}`, { method: "DELETE" }); refreshMemList(); }
      catch (e) { toast(e.message); }
    }));
  } catch { /* non-fatal */ }
}

function wireMemory() {
  $("#btn-remember").addEventListener("click", async () => {
    if (!S.scene) return;
    const mode = S.mode || S.analytics?.mode || S.presets?.scene_to_mode?.[S.scene] || "speech";
    try {
      await api("/api/preferences", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene: S.scene, mode, ...currentControls() }),
      });
      toast(`Remembered ${mode} settings for "${S.scene}" scenes`, "ok", 3000);
      refreshMemList();
    } catch (e) { toast(e.message); }
  });
  $("#btn-restore").addEventListener("click", async () => {
    if (!S.scene) return;
    try {
      const r = await api("/api/preferences");
      const rec = r.preferences?.[S.scene];
      if (!rec) { toast(`No stored preference for "${S.scene}"`, "err", 2600); return; }
      applyPresetToControls(rec.settings);
      S.mode = rec.mode;
      $$(".mode-card").forEach((c) => c.classList.toggle("active", c.dataset.mode === rec.mode));
      S.dirty = true;
      updateModeSource();
      toast(`Restored "${S.scene}" preference (${rec.mode})`, "ok", 2800);
    } catch (e) { toast(e.message); }
  });
  $("#btn-forget").addEventListener("click", async () => {
    if (!S.scene) return;
    try {
      const r = await api(`/api/preferences/${S.scene}`, { method: "DELETE" });
      toast(r.deleted ? `Cleared memory for "${S.scene}"` : `Nothing stored for "${S.scene}"`, "ok", 2600);
      refreshMemList();
    } catch (e) { toast(e.message); }
  });
}

/* ── init ───────────────────────────────────────────────── */

async function init() {
  wireInput(); wireScene(); wireModes(); wireControls();
  wirePlayback(); wireVizTabs(); wireABCanvas(); wireMemory();
  $("#btn-enhance").addEventListener("click", enhance);
  runLoader();
  try {
    S.presets = await api("/api/modes");
    Object.entries(S.presets.modes).forEach(([name, m]) => {
      const card = document.querySelector(`.mode-card[data-mode="${name}"]`);
      if (card) card.title = m.description;
    });
  } catch (e) { toast("Could not load mode presets: " + e.message); }
  refreshMemList();
  setInterval(async () => {
    try { updateEnginePill(await api("/api/health")); } catch { /* offline */ }
  }, 5000);
}

document.addEventListener("DOMContentLoaded", init);
