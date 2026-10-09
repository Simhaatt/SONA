/* SONA web app — one task per page, settings in a drawer. */
"use strict";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];

const EQ_FREQS = ["31", "63", "125", "250", "500", "1k", "2k", "4k"];
const SCENES = ["speech", "music", "traffic", "fan", "crowd", "mixed", "unknown"];
const MODE_LABELS = { aggressive_speech: "Strong speech" };
const MODE_ICON = {
  auto: '<path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1M5.6 18.4l2.1-2.1m8.6-8.6 2.1-2.1"/><circle cx="12" cy="12" r="3"/>',
  speech: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
  aggressive_speech: '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/>',
  music: '<path d="M9 18V6l10-2v12"/><circle cx="6" cy="18" r="3"/><circle cx="16" cy="16" r="3"/>',
  podcast: '<circle cx="12" cy="11" r="3"/><path d="M7.5 15.5a6 6 0 1 1 9 0M12 14v7"/>',
  gaming: '<path d="M6 9h12a4 4 0 0 1 4 4l-1 5a2 2 0 0 1-3.4 1L15 16H9l-2.6 3A2 2 0 0 1 3 18l-1-5a4 4 0 0 1 4-4z"/>',
  hearing: '<path d="M6 9a6 6 0 1 1 11 3c-1 1.7-2.5 2.3-3 4-.4 1.4-1 3-3 3"/>',
  studio: '<path d="M5 4v16M12 4v16M19 4v16"/><circle cx="5" cy="13" r="2"/><circle cx="12" cy="9" r="2"/><circle cx="19" cy="16" r="2"/>',
};
const STAGES = { scene_analysis: "Detecting scene", mode_controller: "Choosing settings", hum_notch: "Removing hum",
  enhancement: "Removing noise", online_adaptation: "Adapting and removing noise", eq_compressor: "Shaping the sound", export_wav: "Saving" };
const PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15l13-7.5z"/></svg>';
const PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4h4v16H7zM13 4h4v16h-4z"/></svg>';

const S = {
  route: "home", engines: [], engineById: {}, presets: null, samples: [],
  fileId: null, fileInfo: null, refInfo: null, sampleId: null,
  outputId: null, outputInfo: null, resInfo: null, analytics: null,
  engine: "gtcrn", mode: null, hum: 0, dirty: false,
  scene: null, sceneConf: null, sceneManual: false,
  ab: "b", busy: false, playhead: 0, history: [],
  cmp: null, cmpMetric: null, cmpSort: null, cmpPlaying: null,
  rtBlock: 64, rtEngine: "gtcrn", rt: null, aboutTable: null, tables: null,
};

/* ── utilities ─────────────────────────────── */
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}
const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
function toast(msg, error = false, ms = 3800) {
  const el = document.createElement("div");
  el.className = `toast${error ? " error" : ""}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}
const tok = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmtTime = (s) => (isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "0:00");
const num = (v, d = 2) => (v == null || !isFinite(v) ? "—" : (+v).toFixed(d));
const signed = (v, d = 2) => (v == null || !isFinite(v) ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const modeName = (m) => (m ? MODE_LABELS[m] || m[0].toUpperCase() + m.slice(1) : "Automatic");
function fill(input) { input.style.setProperty("--fill", `${((+input.value - +input.min) / (+input.max - +input.min)) * 100}%`); }

const REDUCED = matchMedia("(prefers-reduced-motion: reduce)");
function animate(fn, ms = 700) {
  if (REDUCED.matches) return fn(1);
  const t0 = performance.now();
  const step = (now) => { const t = Math.min(1, (now - t0) / ms); fn(1 - Math.pow(1 - t, 3)); if (t < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
function canvasFit(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 600, h = +(cv.dataset.h || (cv.dataset.h = cv.getAttribute("height") || 120));
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = `${h}px`; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
function canvas2d(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.parentElement.clientWidth || 600;
  if (!cv.dataset.h) cv.dataset.h = cv.getAttribute("height") || 120;
  const h = +cv.dataset.h;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = `${h}px`;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/* ── router ────────────────────────────────── */
const ROUTES = ["home", "enhance", "results", "compare", "live", "about"];
function route() {
  const r = (location.hash.replace(/^#\/?/, "") || "home").split("/")[0];
  const name = ROUTES.includes(r) ? r : "home";
  S.route = name;
  $$(".page").forEach((p) => { p.hidden = p.dataset.page !== name; });
  $$(".nav-link").forEach((a) => (a.dataset.route === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  if (name === "results") $("#results-dot").hidden = true;
  const page = $(`.page[data-page="${name}"]`);
  page.classList.remove("enter"); void page.offsetWidth;
  [...page.children].forEach((c, i) => c.style.setProperty("--i", i));
  page.classList.add("enter");
  moveInk();
  if (name === "home") startHome(); else stopHome();
  if (name === "about") ensureAbout();
  window.scrollTo({ top: 0 });
  const TITLES = { enhance: "Enhance", results: "Results", compare: "Compare methods", live: "Real-time", about: "About" };
  document.title = name === "home" ? "SONA — Speech Enhancement" : `${TITLES[name]} · SONA`;
  requestAnimationFrame(redraw);
}
function moveInk() {
  const ink = $(".nav-ink"), a = $('.nav-link[aria-current="page"]');
  if (!ink) return;
  if (!a) { ink.style.width = "0px"; return; }
  ink.style.left = `${a.offsetLeft + 10}px`;
  ink.style.width = `${a.offsetWidth - 20}px`;
}
function redraw() {
  moveInk();
  if (S.route === "home" && !homeRAF) drawHome(0);
  if (S.route === "enhance" && S.fileInfo) drawMiniWave();
  if (S.route === "results" && S.analytics) { drawResultsViz(); drawLive(); }
  if (S.route === "compare" && S.cmp) drawCompareChart();
  if (S.route === "live" && S.rt) drawRtChart();
}
window.addEventListener("hashchange", route);
window.addEventListener("resize", redraw);

/* ── theme + status ────────────────────────── */
function wireTheme() {
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("sona-theme", next); } catch { /* storage blocked */ }
    redraw();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
}
async function pollStatus() {
  try {
    const h = await api("/api/health");
    const pill = $("#engine-pill"), t = $("#engine-pill-text");
    pill.className = "status";
    if (h.engine === "ready") { pill.classList.add("ready"); t.textContent = `Ready · ${(h.device || "cpu").toUpperCase()}`; }
    else if (h.engine === "error") { pill.classList.add("error"); t.textContent = "Model failed to load"; }
    else t.textContent = "Loading model…";
  } catch { $("#engine-pill-text").textContent = "Server offline"; }
}

/* ── home: animated scanner ────────────────── */
let homeRAF = null;
const hash = (k) => { const x = Math.sin(k * 12.9898) * 43758.5453; return x - Math.floor(x); };
const speechEnv = (k) => Math.max(0, Math.sin(k * 0.21) * 0.55 + Math.sin(k * 0.077 + 1) * 0.45) * (0.55 + 0.45 * Math.sin(k * 0.031));
function startHome() {
  if (homeRAF) return;
  const loop = (ts) => { drawHome(ts / 1000); homeRAF = REDUCED.matches || S.route !== "home" ? null : requestAnimationFrame(loop); };
  homeRAF = requestAnimationFrame(loop);
}
function stopHome() { cancelAnimationFrame(homeRAF); homeRAF = null; }
function drawHome(t) {
  const cv = $("#home-wave");
  if (!cv || cv.offsetParent === null) return;
  const { ctx, w, h } = canvasFit(cv);
  const bw = 6, n = Math.ceil(w / bw) + 1, mid = h / 2 + 8, A = h / 2 - 34, scroll = t * 9, beam = w * 0.5;
  const plum = tok("--plum"), accent = tok("--accent"), cyan = tok("--cyan");
  const frac = scroll - Math.floor(scroll);
  for (let i = 0; i < n; i++) {
    const k = Math.floor(scroll) + i, x = (i - frac) * bw;
    const speech = speechEnv(k) * A;
    const left = x < beam;
    const noise = (0.2 + 0.22 * hash(k)) * A * (0.85 + 0.15 * Math.sin(t * 6 + k));
    const amp = left ? speech + noise : speech;
    const near = Math.max(0, 1 - Math.abs(x - beam) / 60);
    ctx.globalAlpha = left ? 0.55 + 0.45 * near : 1;
    ctx.fillStyle = near > 0.35 ? cyan : left ? plum : accent;
    ctx.fillRect(x + 1, mid - Math.max(amp, 1.5), bw - 2, Math.max(amp * 2, 3));
  }
  ctx.globalAlpha = 1;
  const g = ctx.createLinearGradient(beam - 40, 0, beam + 40, 0);
  g.addColorStop(0, "transparent"); g.addColorStop(0.5, cyan); g.addColorStop(1, "transparent");
  ctx.globalAlpha = 0.16 + 0.06 * Math.sin(t * 3);
  ctx.fillStyle = g; ctx.fillRect(beam - 40, 26, 80, h - 40);
  ctx.globalAlpha = 1;
  ctx.shadowColor = cyan; ctx.shadowBlur = 16;
  ctx.fillStyle = cyan; ctx.fillRect(beam - 1, 26, 2, h - 40);
  ctx.shadowBlur = 0;
  const sy = 26 + ((t * 70) % (h - 40));
  ctx.globalAlpha = 0.9; ctx.fillRect(beam - 7, sy, 14, 2); ctx.globalAlpha = 1;
}

/* ── engines, presets, samples ─────────────── */
async function loadEngines() {
  const r = await api("/api/engines");
  S.engines = r.engines;
  S.engineById = Object.fromEntries(r.engines.map((e) => [e.id, e]));
  S.engine = r.default;
  const list = $("#engine-list");
  list.innerHTML = r.engines.map((e) => {
    const b = e.benchmark?.si_sdr_i;
    return `<button type="button" class="engine" role="radio" aria-checked="false" data-engine="${e.id}" ${e.available ? "" : "disabled"}>
      <span class="engine-name">${esc(e.label)}${e.default ? ' <span class="tag tag-ok">recommended</span>' : ""}</span>
      <span class="engine-score">${b == null ? "" : `${signed(b, 1)} dB`}</span>
      <span class="engine-desc">${esc(e.family === "neural" ? "Neural network" : "Classical filter")} · ${esc(e.summary.split(". ")[0].replace(/\.$/, ""))}.</span></button>`;
  }).join("");
  $$(".engine").forEach((b) => b.addEventListener("click", () => selectEngine(b.dataset.engine)));
  selectEngine(S.engine);
  const neural = r.engines.filter((e) => e.family === "neural"), classical = r.engines.filter((e) => e.family === "classical");
  const check = (e) => `<label class="check"><input type="checkbox" value="${e.id}" checked ${e.available ? "" : "disabled"}>${esc(e.label)}</label>`;
  $("#cmp-neural").innerHTML = neural.map(check).join("");
  $("#cmp-classical").innerHTML = classical.map(check).join("");
  $("#rt-block").innerHTML = r.stream_blocks_ms.map((b) => `<button type="button" data-b="${b}" aria-pressed="${b === S.rtBlock}">${b} ms</button>`).join("");
  $("#rt-engine").innerHTML = neural.map((e) => `<button type="button" data-e="${e.id}" aria-pressed="${e.id === S.rtEngine}">${esc(e.label)}</button>`).join("");
  $$("#rt-block button").forEach((b) => b.addEventListener("click", () => { S.rtBlock = +b.dataset.b; pressOne("#rt-block", b); }));
  $$("#rt-engine button").forEach((b) => b.addEventListener("click", () => { S.rtEngine = b.dataset.e; pressOne("#rt-engine", b); }));
}
function pressOne(groupSel, btn) { $$(`${groupSel} button`).forEach((x) => x.setAttribute("aria-pressed", String(x === btn))); }

function selectEngine(id) {
  S.engine = id;
  $$(".engine").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.engine === id)));
  const ok = !!S.engineById[id]?.online_capable;
  const box = $("#online-adapt");
  box.disabled = !ok;
  if (!ok) box.checked = false;
  $("#online-wrap").classList.toggle("disabled", !ok);
  updateSummary();
}

async function loadPresets() {
  S.presets = await api("/api/modes");
  const ORDER = ["speech", "aggressive_speech", "podcast", "hearing", "music", "gaming", "studio"];
  const known = Object.keys(S.presets.modes);
  const modes = ["auto", ...ORDER.filter((m) => known.includes(m)), ...known.filter((m) => !ORDER.includes(m))];
  $("#mode-grid").innerHTML = modes.map((m) => `<button type="button" class="mode" role="radio" aria-checked="${m === "auto"}" data-mode="${m}">
    <svg class="ic" viewBox="0 0 24 24" aria-hidden="true">${MODE_ICON[m] || MODE_ICON.speech}</svg>${m === "auto" ? "Automatic" : modeName(m)}</button>`).join("");
  $$(".mode").forEach((b) => b.addEventListener("click", () => selectMode(b.dataset.mode === "auto" ? null : b.dataset.mode)));
  applyPreset(S.presets.modes.speech.settings);
}
function selectMode(m) {
  S.mode = m;
  $$(".mode").forEach((b) => b.setAttribute("aria-checked", String((b.dataset.mode === "auto" ? null : b.dataset.mode) === m)));
  if (m) { applyPreset(S.presets.modes[m].settings); S.dirty = false; }
  $("#mode-desc").textContent = m ? S.presets.modes[m].description : "Automatic picks a mode from the detected scene.";
  updateSummary();
}
function applyPreset(st) {
  $("#sl-denoise").value = st.mask_strength;
  $("#sl-floor").value = st.safety_floor;
  $("#sl-thresh").value = st.compressor_threshold_db;
  $("#sl-ratio").value = st.compressor_ratio;
  st.eq_gains_db.forEach((g, i) => { const el = $(`#eq-${i}`); if (el) el.value = g; });
  $("#bypass-eq").checked = !!st.bypass_eq;
  $("#bypass-comp").checked = !!st.bypass_compressor;
  refreshLabels();
}
function refreshLabels() {
  $("#val-denoise").textContent = (+$("#sl-denoise").value).toFixed(2);
  $("#val-floor").textContent = (+$("#sl-floor").value).toFixed(2);
  $("#val-thresh").textContent = `${$("#sl-thresh").value} dB`;
  $("#val-ratio").textContent = `${(+$("#sl-ratio").value).toFixed(1)} : 1`;
  $$(".drawer input[type=range]").forEach(fill);
}
function controls() {
  return {
    denoise_strength: +$("#sl-denoise").value, safety_floor: +$("#sl-floor").value,
    eq_gains_db: EQ_FREQS.map((_, i) => +$(`#eq-${i}`).value),
    compressor_threshold_db: +$("#sl-thresh").value, compressor_ratio: +$("#sl-ratio").value,
    bypass_eq: $("#bypass-eq").checked, bypass_compressor: $("#bypass-comp").checked,
  };
}
function updateSummary() {
  $("#sum-engine").textContent = S.engineById[S.engine]?.label || "GTCRN";
  $("#sum-mode").textContent = modeName(S.mode) + (S.dirty ? " (customised)" : "");
  const extras = [];
  if (S.hum) extras.push(`${S.hum} Hz hum filter`);
  if ($("#online-adapt").checked) extras.push("online adaptation");
  if (S.sceneManual && S.scene) extras.push(`scene: ${S.scene}`);
  $("#sum-extras").textContent = extras.length ? extras.join(", ") : "None";
}

async function loadSamples() {
  const r = await api("/api/samples");
  S.samples = r.samples.filter((s) => s.id !== "legacy");
  $("#sample-grid").innerHTML = S.samples.map((s) => {
    const snr = s.snr_db == null ? "" : `<span class="tag ${s.snr_db < 0 ? "tag-hard" : ""}">${s.snr_db > 0 ? "+" : ""}${s.snr_db} dB</span>`;
    return `<button type="button" class="sample" data-sample="${s.id}" aria-pressed="false">
      <span class="sample-title">${esc(s.title)}</span>
      <span class="sample-meta">${snr}${s.has_reference ? '<span class="tag tag-ok">clean ref</span>' : ""}</span>
      <span class="sample-src">${esc(s.dataset)}</span></button>`;
  }).join("");
  $$(".sample").forEach((b) => b.addEventListener("click", () => useSample(b.dataset.sample)));
}

/* ── input ─────────────────────────────────── */
function setSource(src) {
  $$("#src-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.src === src)));
  ["library", "upload"].forEach((k) => { $(`#src-${k}`).hidden = k !== src; });
}
async function useSample(id) {
  try {
    $$(".sample").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.sample === id)));
    setFile(await postJSON("/api/sample", { sample_id: id }), id);
  } catch (e) { toast(e.message, true); }
}
async function uploadFile(file, label) {
  const fd = new FormData();
  fd.append("file", file, file.name);
  try {
    const info = await api("/api/upload", { method: "POST", body: fd });
    setFile(info, null);
    toast(`Loaded ${label || info.name}`);
  } catch (e) { toast(e.message, true); }
}
async function attachReference(file) {
  const fd = new FormData();
  fd.append("file_id", S.fileId);
  fd.append("file", file, file.name);
  try {
    S.refInfo = (await api("/api/reference", { method: "POST", body: fd })).reference;
    updateClipCard();
    toast("Clean reference added. Exact quality scores are now available.");
  } catch (e) { toast(e.message, true); }
}
function setFile(info, sampleId) {
  Object.assign(S, { fileId: info.file_id, fileInfo: info, refInfo: info.reference || null, sampleId,
    outputId: null, outputInfo: null, resInfo: null, analytics: null, cmp: null, rt: null,
    scene: null, sceneConf: null, sceneManual: false });
  if (!sampleId) $$(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
  stopAudio();
  $("#cmp-results").hidden = true; $("#rt-results").hidden = true;
  setScene(null, null, false);
  updateClipCard();
  renderResults();
  $("#btn-enhance").disabled = false;
  $("#btn-classify").disabled = false;
  $("#btn-compare").disabled = false;
  $("#btn-stream").disabled = false;
  $("#enhance-hint").textContent = "Ready. The default settings work well for most clips.";
  $("#audio-a").src = `/api/audio/input/${S.fileId}`;
}
function updateClipCard() {
  const has = !!S.fileInfo;
  $("#clip-card").hidden = !has;
  $$("[data-needs-clip]").forEach((n) => { n.hidden = has; });
  if (!has) { $$("[data-clip-ref]").forEach((n) => { n.textContent = ""; }); return; }
  const i = S.fileInfo;
  $("#clip-name").textContent = i.name;
  $("#clip-detail").textContent = `${i.duration_s.toFixed(1)} s · noise level ≈ ${i.est_snr_db} dB SNR${S.refInfo ? " · clean reference" : ""}`;
  $("#btn-attach-ref").hidden = !!S.refInfo;
  $("#ab-ref-btn").hidden = !S.refInfo;
  if (S.refInfo) $("#audio-r").src = `/api/audio/reference/${S.fileId}`; else $("#audio-r").removeAttribute("src");
  const ref = S.refInfo ? "with clean reference — full scores" : "no clean reference — estimated scores only";
  $$("[data-clip-ref]").forEach((n) => { n.innerHTML = `Clip: <strong>${esc(i.name)}</strong> · ${i.duration_s.toFixed(1)} s · ${ref}`; });
  drawMiniWave();
}
function clearFile() {
  Object.assign(S, { fileId: null, fileInfo: null, refInfo: null, sampleId: null, outputId: null, outputInfo: null, resInfo: null, analytics: null, cmp: null, rt: null });
  stopAudio();
  $$(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
  ["#btn-enhance", "#btn-classify", "#btn-compare", "#btn-stream"].forEach((s) => { $(s).disabled = true; });
  $("#enhance-hint").textContent = "Choose a clip first.";
  $("#cmp-results").hidden = true; $("#rt-results").hidden = true;
  updateClipCard();
  renderResults();
}
function drawMiniWave() {
  const cv = $("#clip-wave"), wf = S.fileInfo?.waveform;
  if (!cv || !wf || cv.offsetParent === null) return;
  const { ctx, w, h } = canvas2d(cv);
  const mid = h / 2, peak = Math.max(S.fileInfo.peak, 0.01), bw = w / wf.bins;
  ctx.fillStyle = tok("--plum");
  for (let i = 0; i < wf.bins; i++) {
    const y1 = mid - (wf.maxs[i] / peak) * (mid - 2), y2 = mid - (wf.mins[i] / peak) * (mid - 2);
    ctx.fillRect(i * bw, Math.min(y1, y2), Math.max(bw, 0.6), Math.max(Math.abs(y2 - y1), 1));
  }
}
function wireInput() {
  $$("#src-tabs button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.src)));
  $$("[data-src-link]").forEach((a) => a.addEventListener("click", () => setSource(a.dataset.srcLink)));
  const dz = $("#dropzone"), fi = $("#file-input");
  fi.addEventListener("change", () => { if (fi.files[0]) uploadFile(fi.files[0]); fi.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f); });
  $("#btn-clear-file").addEventListener("click", clearFile);
  $("#btn-attach-ref").addEventListener("click", () => $("#ref-input").click());
  $("#ref-input").addEventListener("change", (e) => { if (e.target.files[0]) attachReference(e.target.files[0]); e.target.value = ""; });
}

/* ── settings drawer ───────────────────────── */
let lastFocus = null, rerunOnClose = false;
function openDrawer(tab, rerun = false) {
  lastFocus = document.activeElement;
  rerunOnClose = rerun;
  $("#drawer-done").textContent = rerun ? "Re-run with these settings" : "Done";
  $("#drawer").hidden = false; $("#drawer-backdrop").hidden = false;
  if (tab) setDrawerTab(tab);
  refreshLabels();
  setTimeout(() => $("#drawer-close").focus(), 30);
  document.addEventListener("keydown", drawerKeys);
}
function closeDrawer() {
  $("#drawer").hidden = true; $("#drawer-backdrop").hidden = true;
  document.removeEventListener("keydown", drawerKeys);
  updateSummary();
  lastFocus?.focus?.();
}
function drawerKeys(e) {
  if (e.key === "Escape") { e.preventDefault(); closeDrawer(); return; }
  if (e.key !== "Tab") return;
  const f = $$('#drawer button:not([disabled]), #drawer input:not([disabled]), #drawer summary, #drawer [tabindex="0"]').filter((el) => el.offsetParent !== null);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}
function setDrawerTab(t) {
  $$("#drawer-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.dtab === t)));
  $$("[data-dpage]").forEach((p) => { p.hidden = p.dataset.dpage !== t; });
}
function wireDrawer() {
  $$("[data-open-settings]").forEach((b) => b.addEventListener("click", () => openDrawer(null, b.hasAttribute("data-rerun"))));
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#drawer-done").addEventListener("click", () => {
    const rerun = rerunOnClose;
    closeDrawer();
    if (rerun && S.fileId) { toast("Re-running with the new settings…"); enhance(); }
  });
  $("#drawer-backdrop").addEventListener("click", closeDrawer);
  $$("#drawer-tabs button").forEach((b) => b.addEventListener("click", () => setDrawerTab(b.dataset.dtab)));
  $("#eq").innerHTML = EQ_FREQS.map((f, i) => `<div class="eq-band"><input type="range" id="eq-${i}" min="-12" max="12" step="0.5" value="0" aria-label="Equaliser ${f} Hz"><span>${f}</span></div>`).join("");
  $$(".drawer input[type=range]").forEach((r) => r.addEventListener("input", () => { S.dirty = true; refreshLabels(); updateSummary(); }));
  ["#bypass-eq", "#bypass-comp"].forEach((s) => $(s).addEventListener("change", () => { S.dirty = true; updateSummary(); }));
  $("#online-adapt").addEventListener("change", updateSummary);
  $$("#hum-seg button").forEach((b) => b.addEventListener("click", () => { S.hum = +b.dataset.hum; pressOne("#hum-seg", b); updateSummary(); }));
  $("#btn-reset-controls").addEventListener("click", () => { applyPreset(S.presets.modes[S.mode || "speech"].settings); S.dirty = false; updateSummary(); });
  $("#scene-chips").innerHTML = SCENES.map((s) => `<button type="button" data-scene="${s}" aria-pressed="false">${s}</button>`).join("");
  $$("#scene-chips button").forEach((b) => b.addEventListener("click", () => {
    if (S.sceneManual && S.scene === b.dataset.scene) setScene(null, null, false); else setScene(b.dataset.scene, null, true);
  }));
  $("#btn-classify").addEventListener("click", classify);
  $("#btn-remember").addEventListener("click", remember);
  $("#btn-forget").addEventListener("click", forget);
}
function setScene(label, conf, manual) {
  Object.assign(S, { scene: label, sceneConf: conf, sceneManual: manual });
  $("#scene-label").textContent = label ? `${label}${manual ? " (set by you)" : conf != null ? ` · ${Math.round(conf * 100)}% sure` : ""}` : "Not detected yet";
  $$("#scene-chips button").forEach((b) => b.setAttribute("aria-pressed", String(manual && b.dataset.scene === label)));
  $("#btn-remember").disabled = !label; $("#btn-forget").disabled = !label;
  updateSummary();
}
async function classify() {
  if (!S.fileId) return;
  try { const r = await postJSON("/api/classify", { file_id: S.fileId }); setScene(r.scene, r.confidence, false); }
  catch (e) { toast(e.message, true); }
}
async function refreshMemory() {
  try {
    const entries = Object.entries((await api("/api/preferences")).preferences || {});
    $("#mem-list").innerHTML = entries.length
      ? entries.map(([s, r]) => `<div class="mem-item"><span><b>${esc(s)}</b> → ${esc(modeName(r.mode))}</span><span class="hint">${esc((r.updated_at || "").slice(0, 10))}</span></div>`).join("")
      : '<p class="hint">No saved scene settings yet.</p>';
  } catch { /* not fatal */ }
}
async function remember() {
  if (!S.scene) return;
  const mode = S.mode || S.analytics?.mode || S.presets.scene_to_mode[S.scene] || "speech";
  try { await postJSON("/api/preferences", { scene: S.scene, mode, ...controls() }); toast(`Saved ${modeName(mode)} for ${S.scene} scenes`); refreshMemory(); }
  catch (e) { toast(e.message, true); }
}
async function forget() {
  if (!S.scene) return;
  try { const r = await api(`/api/preferences/${S.scene}`, { method: "DELETE" }); toast(r.deleted ? `Forgot settings for ${S.scene}` : `Nothing saved for ${S.scene}`); refreshMemory(); }
  catch (e) { toast(e.message, true); }
}

/* ── enhance ───────────────────────────────── */
function stageList() {
  const st = ["scene_analysis", "mode_controller"];
  if (S.hum) st.push("hum_notch");
  st.push($("#online-adapt").checked ? "online_adaptation" : "enhancement", "eq_compressor", "export_wav");
  return st;
}
let stageTimer = null;
function animateStages() {
  const list = stageList(), ol = $("#stages");
  ol.hidden = false;
  ol.innerHTML = list.map((s) => `<li data-s="${s}">${STAGES[s]}</li>`).join("");
  let i = 0;
  const step = () => {
    if (i > 0) $(`#stages [data-s="${list[i - 1]}"]`)?.classList.replace("active", "done");
    if (i >= list.length || !S.busy) return;
    $(`#stages [data-s="${list[i]}"]`).classList.add("active");
    if (i < list.length - 1) stageTimer = setTimeout(step, list[i].includes("enhance") || list[i].includes("adapt") ? 1400 : 350);
    i++;
  };
  step();
}
async function enhance() {
  if (!S.fileId || S.busy) return;
  S.busy = true;
  const btn = $("#btn-enhance");
  btn.classList.add("busy"); btn.setAttribute("aria-busy", "true");
  $(".run-card").classList.add("scanning");
  animateStages();
  const auto = $("#auto-scene").checked;
  const body = {
    file_id: S.fileId, engine: S.engine, online_adapt: $("#online-adapt").checked, hum_notch_hz: S.hum,
    mode: S.mode, auto_scene: auto && !S.sceneManual, scene: S.sceneManual ? S.scene : auto ? null : S.scene,
    use_memory: $("#use-memory").checked, normalize_output: $("#normalize-out").checked, remember: false,
  };
  if (S.dirty) Object.assign(body, controls());
  try {
    const r = await postJSON("/api/enhance", body);
    Object.assign(S, { outputId: r.output_id, outputInfo: r.output, resInfo: r.residual, analytics: r.analytics });
    if (r.analytics.auto_scene) setScene(r.analytics.scene, r.analytics.scene_confidence, false);
    S.history.unshift({ n: S.history.length + 1, fileId: S.fileId, name: S.fileInfo.name, outputId: S.outputId, outputInfo: S.outputInfo, resInfo: S.resInfo, analytics: r.analytics });
    clearTimeout(stageTimer);
    $$("#stages li").forEach((li) => { li.className = "done"; });
    renderResults();
    if (S.route !== "results") $("#results-dot").hidden = false;
    location.hash = "#/results";
  } catch (e) {
    toast(e.message, true);
    $("#stages").hidden = true;
  } finally {
    S.busy = false;
    $(".run-card").classList.remove("scanning");
    btn.classList.remove("busy"); btn.removeAttribute("aria-busy");
  }
}

/* ── results ───────────────────────────────── */
// Before/after tile: the original (unprocessed) value next to the cleaned value.
function kpiBA(label, orig, after, unit, d, higherBetter, lead = false, note = "") {
  const diff = after - orig, better = higherBetter ? diff >= 0 : diff <= 0;
  const fmt = (v) => (v == null || !isFinite(v) ? "—" : `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`);
  const change = `${Math.abs(diff).toFixed(d)}${unit ? " " + unit : ""}`;
  return `<div class="kpi kpi-ba${lead ? " lead" : ""}"><span class="kpi-k">${label}</span>
    <div class="ba"><span class="ba-orig"><small>Original</small><b>${fmt(orig)}</b></span><span class="ba-arrow" aria-hidden="true">→</span>
    <span class="ba-new"><small>Cleaned</small><span class="kpi-v">${fmt(after)}${unit ? `<small>${unit}</small>` : ""}</span></span></div>
    <span class="kpi-d ${better ? "up" : "down"}">${change} ${better ? "better" : "worse"}${note ? ` · ${note}` : ""}</span></div>`;
}
function noiseFloorDb(spec) {
  const v = spec.floor_db.filter((_, i) => spec.freqs_hz[i] >= 300);
  return v.reduce((s, x) => s + x, 0) / Math.max(v.length, 1);
}
function kpi(label, value, unit, delta, cls = "", lead = false) {
  return `<div class="kpi${lead ? " lead" : ""}"><span class="kpi-k">${label}</span><span class="kpi-v">${value}${unit ? `<small>${unit}</small>` : ""}</span>${delta ? `<span class="kpi-d ${cls}">${delta}</span>` : ""}</div>`;
}
function renderResults() {
  const a = S.analytics;
  $("#results-empty").hidden = !!a;
  $("#results-body").hidden = !a;
  if (!a) return;
  $("#results-sub").textContent = `${S.fileInfo.name} · ${a.engine_label} · ${modeName(a.mode)} mode${a.scene ? ` · scene: ${a.scene}` : ""}`;
  $("#audio-b").src = `/api/audio/output/${S.outputId}`;
  $("#audio-n").src = `/api/audio/residual/${S.outputId}`;
  $("#btn-download").href = `/api/download/${S.outputId}`;
  setAB("b", true);

  const m = a.metrics;
  const floorIn = noiseFloorDb(S.fileInfo.spectrum), floorOut = noiseFloorDb(S.outputInfo.spectrum);
  const noiseTile = kpiBA("Background noise level", floorIn, floorOut, "dB", 1, false, !m, "lower is quieter");
  const snrTile = kpiBA("Signal-to-noise (estimated)", a.est_snr_in_db, a.est_snr_out_db, "dB", 1, true);
  $("#kpis").innerHTML = m
    ? kpiBA("Speech quality (SI-SDR)", m.input.si_sdr, m.model.si_sdr, "dB", 1, true, true, "vs clean speech") +
      kpiBA("Intelligibility (STOI)", m.input.stoi, m.model.stoi, "", 3, true) + noiseTile + snrTile
    : noiseTile + snrTile + kpiBA("Loudness", a.input_rms_db, a.output_rms_db, "dBFS", 1, true, false, "level only") +
      kpi("Speed", (1 / Math.max(a.realtime_factor, 1e-3)).toFixed(1), "× real time", `${a.processing_time_s} s for ${a.input_duration_s.toFixed(1)} s of audio`);
  $$("#kpis .kpi").forEach((k, i) => k.style.setProperty("--i", i));
  countUp($("#kpis"));
  ["#sgram-in", "#sgram-out"].forEach((sel, i) => { const c = $(sel); c.classList.remove("reveal"); void c.offsetWidth; c.style.animationDelay = `${i * 180}ms`; c.classList.add("reveal"); });
  const ob = $("#online-block");
  ob.hidden = !a.online;
  if (a.online) {
    const o = a.online;
    ob.innerHTML = o.accepted
      ? `<strong>Online adaptation accepted.</strong> The model refined itself for this clip and passed the safety guard.`
      : `<strong>Online adaptation was rejected by the safety guard.</strong> The change it proposed was too large, so the original model's output was kept. This guard is why the feature never made results worse in testing.`;
  }
  $("#metric-table-wrap").innerHTML = m ? `<div class="table-wrap"><table class="tbl">
    <thead><tr><th>Scored against clean speech</th><th>SI-SDR</th><th>SI-SDR gain</th><th>STOI</th><th>ESTOI</th><th>LSD</th><th>PESQ</th></tr></thead>
    <tbody>${[["Original", m.input], ["Model output", m.model], ["Delivered (after sound shaping)", m.delivered]].map(([n, r]) =>
      `<tr><td>${n}</td><td>${num(r.si_sdr, 2)}</td><td>${signed(r.si_sdr_i, 2)}</td><td>${num(r.stoi, 3)}</td><td>${num(r.estoi, 3)}</td><td>${num(r.lsd, 2)}</td><td>${r.pesq == null ? "n/a" : num(r.pesq, 2)}</td></tr>`).join("")}</tbody></table></div>
    <p class="hint">SI-SDR and LSD in dB. PESQ needs an optional package and shows n/a without it.</p>` : `<p class="hint">This clip has no clean reference, so scores are estimates. Use a library clip, or add the clean version on the Enhance page, for exact scores.</p>`;
  const stat = (k, v) => `<div class="stat"><div class="stat-k">${k}</div><div class="stat-v">${v}</div></div>`;
  $("#stats-grid").innerHTML = stat("Model", esc(a.engine_label)) + stat("Sound mode", modeName(a.mode)) + stat("Scene", a.scene || "—") +
    stat("Noise removal strength", a.denoise_strength.toFixed(2)) + stat("Hum filter", a.hum_notch_hz ? `${a.hum_notch_hz} Hz` : "Off") +
    stat("Loudness change", `${signed(a.loudness_change_db, 1)} dB`) + stat("Input level", `${a.input_rms_db} dBFS`) + stat("Output level", `${a.output_rms_db} dBFS`) +
    stat("Limiter", a.limiter_engaged ? "Engaged" : "Not needed") + stat("Saved settings", a.memory_used ? "Applied" : "Not used") +
    stat("Processing speed", `${(1 / Math.max(a.realtime_factor, 1e-3)).toFixed(1)}× real time`);
  const total = a.stages.reduce((s, x) => s + x.seconds, 0) || 1;
  $("#stage-times").innerHTML = a.stages.map((s) => `<div class="timing"><span>${STAGES[s.stage] || s.stage}</span><span class="timing-bar"><i style="width:${(s.seconds / total) * 100}%"></i></span><span class="num">${(s.seconds * 1000).toFixed(0)} ms</span></div>`).join("");
  renderHistory();
  requestAnimationFrame(redraw);
}
function countUp(root) {
  $$(".kpi-v", root).forEach((el) => {
    const node = [...el.childNodes].find((c) => c.nodeType === 3);
    const m = node && node.textContent.match(/^([+−-]?)(\d+(?:\.\d+)?)(.*)$/);
    if (!m) return;
    const [, sign, numStr, rest] = m, target = parseFloat(numStr), dec = (numStr.split(".")[1] || "").length;
    animate((p) => { node.textContent = `${sign}${(target * p).toFixed(dec)}${rest}`; }, 900);
  });
}
function renderHistory() {
  const show = S.history.length > 1;
  $("#history-head").hidden = !show; $("#history-wrap").hidden = !show;
  if (!show) return;
  $("#history-table").innerHTML = `<thead><tr><th>Run</th><th>Clip</th><th>Model</th><th>Mode</th><th>Quality gain</th></tr></thead><tbody>${S.history.map((h, i) => {
    const g = h.analytics.metrics?.model?.si_sdr_i;
    return `<tr class="clickable ${h.outputId === S.outputId ? "current" : ""}" data-i="${i}" tabindex="0"><td>#${h.n}</td><td>${esc(h.name)}</td><td>${esc(h.analytics.engine_label)}</td><td>${modeName(h.analytics.mode)}</td><td>${g == null ? "—" : `${signed(g, 2)} dB`}</td></tr>`;
  }).join("")}</tbody>`;
  $$("#history-table tr.clickable").forEach((tr) => {
    const go = () => {
      const h = S.history[+tr.dataset.i];
      if (h.fileId !== S.fileId) return toast("That run used a different clip. Load that clip first.", true);
      Object.assign(S, { outputId: h.outputId, outputInfo: h.outputInfo, resInfo: h.resInfo, analytics: h.analytics });
      stopAudio(); renderResults();
    };
    tr.addEventListener("click", go);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  });
}

/* visualisations */
const LUT = (() => {
  const st = [[0, [0, 0, 4]], [0.18, [40, 11, 84]], [0.38, [140, 41, 129]], [0.58, [222, 73, 104]], [0.78, [254, 159, 109]], [1, [252, 253, 191]]];
  const lut = new Uint8ClampedArray(768);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 0;
    while (k < st.length - 2 && t > st[k + 1][0]) k++;
    const u = (t - st[k][0]) / (st[k + 1][0] - st[k][0]);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = st[k][1][c] + (st[k + 1][1][c] - st[k][1][c]) * u;
  }
  return lut;
})();
function drawSgram(cv, sg) {
  const { ctx, w, h } = canvas2d(cv);
  if (!sg) return;
  const off = document.createElement("canvas");
  off.width = sg.frames; off.height = sg.bins;
  const img = off.getContext("2d").createImageData(sg.frames, sg.bins);
  for (let f = 0; f < sg.frames; f++) for (let b = 0; b < sg.bins; b++) {
    const v = sg.data[f * sg.bins + b], p = ((sg.bins - 1 - b) * sg.frames + f) * 4;
    img.data[p] = LUT[v * 3]; img.data[p + 1] = LUT[v * 3 + 1]; img.data[p + 2] = LUT[v * 3 + 2]; img.data[p + 3] = 255;
  }
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.drawImage(off, 0, 0, w, h);
  ctx.font = `10px ${tok("--mono")}`; ctx.textAlign = "right"; ctx.fillStyle = "rgba(255,255,255,.55)";
  [4].forEach((k) => ctx.fillText(`${k} kHz`, w - 6, h - (k / 8) * h - 4));
  if (S.playhead > 0) { ctx.fillStyle = "#fff"; ctx.fillRect(S.playhead * w - 1, 0, 2, h); }
}
function drawWave(cv, wf, color, peak) {
  const { ctx, w, h } = canvas2d(cv);
  if (!wf) return;
  const mid = h / 2, bw = w / wf.bins, amp = (mid - 4) / Math.max(peak, 0.01);
  ctx.fillStyle = color;
  for (let i = 0; i < wf.bins; i++) {
    const y1 = mid - wf.maxs[i] * amp, y2 = mid - wf.mins[i] * amp;
    ctx.fillRect(i * bw, Math.min(y1, y2), Math.max(bw - 0.3, 0.6), Math.max(Math.abs(y2 - y1), 1));
  }
  if (S.playhead > 0) { ctx.fillStyle = tok("--fg"); ctx.fillRect(S.playhead * w - 0.75, 0, 1.5, h); }
}
function drawSpectrum() {
  const cv = $("#spec-canvas"), a = S.fileInfo?.spectrum, b = S.outputInfo?.spectrum;
  const { ctx, w, h } = canvas2d(cv);
  if (!a || !b) return;
  const all = [...a.levels_db, ...b.levels_db, ...a.floor_db, ...b.floor_db];
  const hi = Math.max(...all), lo = Math.max(Math.min(...all), hi - 90), n = a.levels_db.length, bw = w / n;
  const y = (d) => h - 20 - ((d - lo) / (hi - lo || 1)) * (h - 32);
  const line = (arr, col, width, dash) => { ctx.beginPath(); arr.forEach((d, i) => (i ? ctx.lineTo(i * bw + bw / 2, y(d)) : ctx.moveTo(0, y(d)))); ctx.strokeStyle = col; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]); };
  line(a.levels_db, tok("--plum"), 2); line(b.levels_db, tok("--accent"), 2);
  line(a.floor_db, tok("--plum"), 1, [3, 3]); line(b.floor_db, tok("--accent"), 1, [3, 3]);
  ctx.fillStyle = tok("--muted"); ctx.font = `10px ${tok("--mono")}`; ctx.textAlign = "center";
  [100, 500, 1000, 2000, 4000, 8000].forEach((f) => { const i = a.freqs_hz.findIndex((v) => v >= f); if (i >= 0) ctx.fillText(f >= 1000 ? `${f / 1000}k` : f, Math.max(i * bw, 12), h - 5); });
  let sum = 0, c = 0;
  a.freqs_hz.forEach((f, i) => { if (f >= 2000) { sum += a.floor_db[i] - b.floor_db[i]; c++; } });
  $("#spec-note").innerHTML = `<span style="color:var(--plum)">Purple</span> = original, <span style="color:var(--accent)">orange</span> = enhanced; dotted lines are the noise floor. Above 2 kHz the noise floor dropped by ${(c ? sum / c : 0).toFixed(1)} dB.`;
}
let abSplit = 0.5;
function drawAB() {
  const cv = $("#ab-canvas"), A = S.fileInfo?.waveform, B = S.outputInfo?.waveform;
  const { ctx, w, h } = canvas2d(cv);
  if (!A || !B) return;
  const peak = Math.max(S.fileInfo.peak, S.outputInfo.peak, 0.01), mid = h / 2, bw = w / A.bins, sx = abSplit * w;
  for (let i = 0; i < A.bins; i++) {
    const L = i * bw < sx, wf = L ? A : B;
    ctx.fillStyle = L ? tok("--plum") : tok("--accent");
    const y1 = mid - (wf.maxs[i] / peak) * (mid - 6), y2 = mid - (wf.mins[i] / peak) * (mid - 6);
    ctx.fillRect(i * bw, Math.min(y1, y2), Math.max(bw - 0.3, 0.6), Math.max(Math.abs(y2 - y1), 1));
  }
  ctx.fillStyle = tok("--fg"); ctx.fillRect(sx - 1, 0, 2, h);
  ctx.beginPath(); ctx.arc(sx, mid, 8, 0, Math.PI * 2); ctx.fill();
}
function drawResultsViz() {
  drawSgram($("#sgram-in"), S.fileInfo?.spectrogram);
  drawSgram($("#sgram-out"), S.outputInfo?.spectrogram);
  if (!$("#more").open) return;
  const viz = $("#viz-tabs [aria-selected=true]").dataset.viz;
  if (viz === "wave") {
    const peak = Math.max(S.fileInfo.peak, S.outputInfo.peak, S.resInfo.peak);
    drawWave($("#wave-in"), S.fileInfo.waveform, tok("--plum"), peak);
    drawWave($("#wave-out"), S.outputInfo.waveform, tok("--accent"), peak);
    drawWave($("#wave-res"), S.resInfo.waveform, tok("--muted"), peak);
  }
  if (viz === "spec") drawSpectrum();
  if (viz === "ab") drawAB();
}

/* player */
const AUD = { a: () => $("#audio-a"), b: () => $("#audio-b"), n: () => $("#audio-n"), r: () => $("#audio-r") };
const cur = () => AUD[S.ab]();
let raf = null;
function setAB(which, silent = false) {
  if (which === "r" && !S.refInfo) return;
  const prev = cur(), playing = !prev.paused, t = prev.currentTime || 0;
  prev.pause();
  S.ab = which;
  $$("#ab-toggle button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.ab === which)));
  const next = cur();
  if (!silent && next.src) { try { next.currentTime = t; } catch { /* not ready */ } if (playing) next.play().catch(() => {}); }
}
function stopAudio() {
  Object.values(AUD).forEach((f) => { const a = f(); a.pause(); try { a.currentTime = 0; } catch { /* empty */ } });
  S.playhead = 0; setPlayIcon(false); updateTransport();
}
function setPlayIcon(p) { $("#play-icon").hidden = p; $("#pause-icon").hidden = !p; $("#btn-play").setAttribute("aria-label", p ? "Pause" : "Play"); $("#btn-play").classList.toggle("playing", p); }
let AC = null, analyser = null, freq = null;
function ensureAnalyser() {
  if (AC) { AC.resume(); return; }
  try {
    AC = new AudioContext();
    analyser = AC.createAnalyser(); analyser.fftSize = 256; analyser.smoothingTimeConstant = 0.78;
    analyser.connect(AC.destination);
    Object.values(AUD).forEach((f) => AC.createMediaElementSource(f()).connect(analyser));
    freq = new Uint8Array(analyser.frequencyBinCount);
  } catch { AC = null; analyser = null; }
}
function drawLive() {
  const cv = $("#live-viz");
  if (!cv || cv.offsetParent === null) return;
  const { ctx, w, h } = canvasFit(cv);
  const bars = Math.max(24, Math.floor(w / 9)), bw = w / bars, playing = !cur().paused && analyser;
  if (playing) analyser.getByteFrequencyData(freq);
  const color = { a: tok("--plum"), b: tok("--accent"), n: tok("--muted"), r: tok("--cyan") }[S.ab];
  const g = ctx.createLinearGradient(0, h, 0, 0);
  g.addColorStop(0, color); g.addColorStop(1, tok("--cyan"));
  ctx.fillStyle = g;
  for (let i = 0; i < bars; i++) {
    const bin = Math.floor(Math.pow(i / bars, 1.6) * (freq ? freq.length * 0.85 : 1));
    const v = playing ? freq[bin] / 255 : 0.03;
    const bh = Math.max(2, v * (h - 6));
    ctx.globalAlpha = 0.35 + 0.65 * v;
    ctx.fillRect(i * bw + 1.5, (h - bh) / 2, bw - 3, bh);
  }
  ctx.globalAlpha = 1;
}
function updateTransport() {
  const a = cur(), d = a.duration || S.fileInfo?.duration_s || 0, t = a.currentTime || 0;
  S.playhead = d ? t / d : 0;
  $("#transport-fill").style.width = `${S.playhead * 100}%`;
  $("#transport-track").setAttribute("aria-valuenow", String(Math.round(S.playhead * 100)));
  $("#time-label").textContent = `${fmtTime(t)} / ${fmtTime(d)}`;
}
function tick() { updateTransport(); drawResultsViz(); drawLive(); if (!cur().paused) raf = requestAnimationFrame(tick); else { setPlayIcon(false); drawLive(); } }
function togglePlay() {
  const a = cur();
  if (!a.src) return;
  if (a.paused) { ensureAnalyser(); a.play().catch(() => {}); setPlayIcon(true); cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); }
  else { a.pause(); setPlayIcon(false); }
}
function seekTo(frac) { const a = cur(); if (a.duration) { a.currentTime = Math.max(0, Math.min(1, frac)) * a.duration; updateTransport(); drawResultsViz(); } }
function wirePlayer() {
  $("#btn-play").addEventListener("click", togglePlay);
  $$("#ab-toggle button").forEach((b) => b.addEventListener("click", () => setAB(b.dataset.ab)));
  Object.values(AUD).forEach((f) => f().addEventListener("ended", () => { setPlayIcon(false); updateTransport(); }));
  const seekEv = (e) => { const r = e.currentTarget.getBoundingClientRect(); seekTo((e.clientX - r.left) / r.width); };
  ["#transport-track", "#sgram-in", "#sgram-out", "#wave-in", "#wave-out", "#wave-res"].forEach((s) => $(s).addEventListener("click", seekEv));
  $("#transport-track").addEventListener("keydown", (e) => {
    const a = cur(); if (!a.duration) return;
    if (e.key === "ArrowRight") { e.preventDefault(); seekTo((a.currentTime + 2) / a.duration); }
    if (e.key === "ArrowLeft") { e.preventDefault(); seekTo((a.currentTime - 2) / a.duration); }
  });
  document.addEventListener("keydown", (e) => {
    if (S.route !== "results" || !S.analytics || !$("#drawer").hidden || e.target.closest("input, textarea, button, select, summary, [role=slider]")) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    const map = { Digit1: "a", Digit2: "b", Digit3: "n", Digit4: "r" };
    if (map[e.code]) setAB(map[e.code]);
  });
  $$("#viz-tabs button").forEach((b) => b.addEventListener("click", () => {
    $$("#viz-tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $$("[data-vizpage]").forEach((p) => { p.hidden = p.dataset.vizpage !== b.dataset.viz; });
    drawResultsViz();
  }));
  $("#more").addEventListener("toggle", drawResultsViz);
  const cv = $("#ab-canvas");
  let drag = false;
  const mv = (e) => { const r = cv.getBoundingClientRect(); abSplit = Math.min(0.98, Math.max(0.02, (e.clientX - r.left) / r.width)); drawAB(); };
  cv.addEventListener("pointerdown", (e) => { drag = true; cv.setPointerCapture(e.pointerId); mv(e); });
  cv.addEventListener("pointermove", (e) => { if (drag) mv(e); });
  cv.addEventListener("pointerup", () => { drag = false; });
}

/* ── compare ───────────────────────────────── */
const CM = {
  si_sdr_i: { label: "Quality gain", unit: "dB", hi: true, ref: true, d: 2 },
  stoi: { label: "Intelligibility", unit: "", hi: true, ref: true, d: 3 },
  lsd: { label: "Spectral distance", unit: "dB", hi: false, ref: true, d: 2 },
  floor: { label: "Noise floor drop", unit: "dB", hi: true, ref: false, d: 1 },
  rtf: { label: "Speed (RTF)", unit: "", hi: false, ref: false, d: 3 },
};
const cval = (r, k) => (r.error ? null : k === "floor" ? r.noise_floor_reduction_db : k === "rtf" ? r.rtf : r.metrics ? r.metrics[k] : null);
// the unprocessed clip as a comparison row: improvements are 0, scores are the input's own
function cmpRows() {
  const c = S.cmp;
  const orig = { label: "Original (no processing)", family: "original", original: true, metrics: c.original?.metrics || null,
    noise_floor_reduction_db: 0, rtf: null, params: 0, src: `/api/audio/input/${S.fileId}` };
  return [orig, ...c.rows.map((r) => ({ ...r, src: r.audio_id ? `/api/audio/output/${r.audio_id}` : null }))];
}
async function runCompare() {
  const engines = $$(".cmp-pick input:checked").map((i) => i.value);
  if (!engines.length) return toast("Select at least one method.", true);
  const btn = $("#btn-compare");
  btn.classList.add("busy"); btn.disabled = true;
  try {
    S.cmp = await postJSON("/api/compare", { file_id: S.fileId, engines });
    S.cmpMetric = S.cmp.has_reference ? "si_sdr_i" : "floor";
    S.cmpSort = { key: S.cmpMetric, dir: -1 };
    renderCompare();
  } catch (e) { toast(e.message, true); }
  finally { btn.classList.remove("busy"); btn.disabled = !S.fileId; }
}
function renderCompare() {
  const c = S.cmp;
  $("#cmp-results").hidden = false;
  $("#cmp-sub").textContent = c.has_reference
    ? "Scored against the clean recording. Raw model output, no sound shaping, so the comparison is fair."
    : "No clean reference for this clip, so methods are ranked by estimated noise reduction only.";
  const keys = Object.keys(CM).filter((k) => c.has_reference || !CM[k].ref);
  $("#cmp-metric").innerHTML = keys.map((k) => `<button type="button" data-k="${k}" aria-pressed="${k === S.cmpMetric}">${CM[k].label}</button>`).join("");
  $$("#cmp-metric button").forEach((b) => b.addEventListener("click", () => { S.cmpMetric = b.dataset.k; renderCompare(); }));
  animate((p) => drawCompareChart(p), 750);
  renderCompareTable(keys);
}
function drawCompareChart(p = 1) {
  const k = S.cmpMetric, sp = CM[k];
  const rows = cmpRows().filter((r) => cval(r, k) != null).sort((a, b) => (sp.hi ? cval(b, k) - cval(a, k) : cval(a, k) - cval(b, k)));
  const cv = $("#cmp-chart");
  cv.dataset.h = Math.max(120, rows.length * 34 + 10);
  const { ctx, w, h } = canvas2d(cv);
  if (!rows.length) return;
  const vals = rows.map((r) => cval(r, k)), lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const lw = Math.min(160, w * 0.34), vw = 74, pw = w - lw - vw, x = (v) => lw + ((v - lo) / (hi - lo || 1)) * pw;
  ctx.strokeStyle = tok("--line-strong"); ctx.beginPath(); ctx.moveTo(x(0), 0); ctx.lineTo(x(0), h); ctx.stroke();
  rows.forEach((r, i) => {
    const y = 6 + i * 34, v = vals[i] * p, x0 = x(Math.min(0, v)), x1 = x(Math.max(0, v));
    ctx.fillStyle = tok("--fg-2"); ctx.font = `13px ${tok("--body")}`; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(r.label, 0, y + 11);
    ctx.fillStyle = r.original ? tok("--muted") : r.family === "neural" ? tok("--neural") : tok("--classical");
    if (r.original) { ctx.fillStyle = tok("--fg"); ctx.font = `italic 13px ${tok("--body")}`; ctx.fillText(r.label, 0, y + 11); ctx.fillStyle = tok("--muted"); }
    ctx.beginPath(); (ctx.roundRect ? ctx.roundRect(x0, y, Math.max(x1 - x0, 2), 22, 4) : ctx.rect(x0, y, Math.max(x1 - x0, 2), 22)); ctx.fill();
    ctx.fillStyle = tok("--fg"); ctx.font = `500 12px ${tok("--mono")}`;
    ctx.fillText(`${k === "si_sdr_i" || k === "floor" ? signed(v, sp.d) : v.toFixed(sp.d)}${sp.unit ? " " + sp.unit : ""}`, Math.max(x1, x(0)) + 8, y + 11);
  });
  // dashed line at the original's value, so every bar reads as better or worse than doing nothing
  const o = rows.findIndex((r) => r.original);
  if (o >= 0 && vals[o] !== 0) {
    const xo = x(vals[o] * p);
    ctx.strokeStyle = tok("--muted"); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(xo, 0); ctx.lineTo(xo, h); ctx.stroke(); ctx.setLineDash([]);
  }
}
function renderCompareTable(keys) {
  const c = S.cmp, so = S.cmpSort, best = {}, all = cmpRows();
  keys.forEach((k) => { const v = c.rows.map((r) => cval(r, k)).filter((x) => x != null); if (v.length) best[k] = CM[k].hi ? Math.max(...v) : Math.min(...v); });
  const rows = [all[0], ...all.slice(1).sort((a, b) => { const va = cval(a, so.key), vb = cval(b, so.key); if (va == null) return 1; if (vb == null) return -1; return (va - vb) * so.dir; })];
  $("#cmp-table").innerHTML = `<thead><tr><th>Listen</th><th>Method</th>${keys.map((k) => `<th class="sortable ${so.key === k ? "sorted" : ""}" data-k="${k}" tabindex="0" aria-sort="${so.key === k ? (so.dir < 0 ? "descending" : "ascending") : "none"}">${CM[k].label}</th>`).join("")}<th>Size</th></tr></thead>
    <tbody>${rows.map((r) => r.error ? `<tr><td></td><td>${esc(r.label)}</td><td colspan="${keys.length + 1}" class="neg">${esc(r.error)}</td></tr>` :
      `<tr class="${r.original ? "orig-row" : ""}"><td><button type="button" class="play-mini" data-src="${r.src}" aria-pressed="false" aria-label="Play ${esc(r.label)}">${PLAY}</button></td><td>${esc(r.label)}</td>${keys.map((k) => {
        const v = cval(r, k); return `<td class="${!r.original && v === best[k] ? "best" : k === "si_sdr_i" && v < 0 ? "neg" : ""}">${v == null ? "—" : k === "si_sdr_i" || k === "floor" ? signed(v, CM[k].d) : v.toFixed(CM[k].d)}</td>`;
      }).join("")}<td>${r.params ? `${Math.round(r.params / 1000)}k` : "—"}</td></tr>`).join("")}</tbody>`;
  $$("#cmp-table th.sortable").forEach((th) => {
    const go = () => { const k = th.dataset.k; S.cmpSort = { key: k, dir: S.cmpSort.key === k ? -S.cmpSort.dir : CM[k].hi ? -1 : 1 }; renderCompareTable(keys); };
    th.addEventListener("click", go); th.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  });
  $$("#cmp-table .play-mini").forEach((b) => b.addEventListener("click", () => {
    const au = $("#cmp-audio"), reset = () => $$("#cmp-table .play-mini").forEach((x) => { x.setAttribute("aria-pressed", "false"); x.innerHTML = PLAY; });
    if (S.cmpPlaying === b.dataset.src && !au.paused) { au.pause(); reset(); S.cmpPlaying = null; return; }
    reset(); au.src = b.dataset.src; au.play().catch(() => {});
    S.cmpPlaying = b.dataset.src; b.setAttribute("aria-pressed", "true"); b.innerHTML = PAUSE;
    au.onended = () => { reset(); S.cmpPlaying = null; };
  }));
}

/* ── real-time ─────────────────────────────── */
async function runStream() {
  const btn = $("#btn-stream");
  btn.classList.add("busy"); btn.disabled = true;
  try { S.rt = await postJSON("/api/stream", { file_id: S.fileId, block_ms: S.rtBlock, engine: S.rtEngine }); renderRt(); }
  catch (e) { toast(e.message, true); }
  finally { btn.classList.remove("busy"); btn.disabled = !S.fileId; }
}
function renderRt() {
  const r = S.rt;
  $("#rt-results").hidden = false;
  $("#rt-kpis").innerHTML =
    kpi("Delay you would hear", r.algorithmic_latency_ms, "ms", `${r.block_ms} ms blocks`, "", true) +
    kpi("Average time per block", num(r.mean_block_ms, 1), "ms", `budget ${r.hop_ms} ms`) +
    kpi("Late blocks", r.deadline_misses, `of ${r.blocks}`, r.deadline_misses ? "would cause glitches on this computer" : "keeps up in real time", r.deadline_misses ? "down" : "up") +
    kpi("Speed", (1 / Math.max(r.rtf, 1e-3)).toFixed(1), "× real time", r.rtf < 1 ? "faster than real time" : "slower than real time", r.rtf < 1 ? "up" : "down");
  const ms = r.metrics_stream, mo = r.metrics_offline, mi = r.metrics_input;
  const val = (m, f) => (m ? `${signed(m.si_sdr_i, 1)}<small>dB</small>` : `−${num(f, 1)}<small>dB</small>`);
  const sub = (m) => (m ? ` · intelligibility ${num(m.stoi, 3)}` : "");
  $("#rt-compare").innerHTML =
    `<div class="kpi kpi-orig"><span class="kpi-k">Original (no processing)</span><span class="kpi-v">${mi ? "+0.0" : "−0.0"}<small>dB</small></span><span class="kpi-d">the baseline${sub(mi)}</span></div>` +
    `<div class="kpi"><span class="kpi-k">Streamed</span><span class="kpi-v">${val(ms, r.floor_reduction_stream_db)}</span><span class="kpi-d">${ms ? "quality gain" : "noise floor drop"}, block by block${sub(ms)}</span></div>` +
    `<div class="kpi"><span class="kpi-k">Whole file at once</span><span class="kpi-v">${val(mo, r.floor_reduction_offline_db)}</span><span class="kpi-d">model sees the full recording${sub(mo)}</span></div>` +
    `<div class="kpi"><span class="kpi-k">Cost of streaming</span><span class="kpi-v">${ms && mo ? signed(ms.si_sdr_i - mo.si_sdr_i, 1) : signed(r.floor_reduction_offline_db - r.floor_reduction_stream_db, 1)}<small>dB</small></span><span class="kpi-d">larger blocks reduce this but add delay</span></div>`;
  $("#rt-audio-in").src = `/api/audio/input/${S.fileId}`;
  $("#rt-audio-stream").src = `/api/audio/output/${r.output_id}`;
  animate((p) => drawRtChart(p), 900);
}
function drawRtChart(p = 1) {
  const r = S.rt, { ctx, w, h } = canvas2d($("#rt-chart")), t = r.block_times_ms, n = t.length;
  if (!n) return;
  const top = Math.max(r.hop_ms * 1.3, ...t) * 1.05, pad = 30, bw = (w - pad - 8) / n, y = (v) => h - 18 - (v / top) * (h - 30);
  ctx.font = `10px ${tok("--mono")}`; ctx.fillStyle = tok("--muted"); ctx.textAlign = "right";
  [0, Math.round(top / 2), Math.round(top)].forEach((v) => ctx.fillText(`${v}`, pad - 6, y(v) + 3));
  for (let i = 0; i < n; i++) {
    const q = Math.max(0, Math.min(1, p * 1.6 - (i / n) * 0.6)), v = t[i] * q;
    ctx.fillStyle = t[i] > r.hop_ms ? tok("--bad") : tok("--accent");
    ctx.fillRect(pad + i * bw, y(v), Math.max(bw - 1, 1), h - 18 - y(v));
  }
  const yd = y(r.hop_ms);
  ctx.strokeStyle = tok("--fg"); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(pad, yd); ctx.lineTo(w - 8, yd); ctx.stroke(); ctx.setLineDash([]);
  const label = `deadline ${r.hop_ms} ms`;
  ctx.font = `500 11px ${tok("--mono")}`; const lw = ctx.measureText(label).width + 10;
  ctx.fillStyle = tok("--surface-2"); ctx.fillRect(w - lw - 12, yd - 17, lw, 15);
  ctx.fillStyle = tok("--fg"); ctx.textAlign = "left"; ctx.fillText(label, w - lw - 7, yd - 6);
  ctx.fillStyle = tok("--muted"); ctx.textAlign = "center"; ctx.font = `10px ${tok("--mono")}`; ctx.fillText("ms per block · blocks in order →", w / 2, h - 4);
}
function wireRealtime() {
  $("#btn-stream").addEventListener("click", runStream);
  $("#btn-compare").addEventListener("click", runCompare);
  const pick = () => ($('#rt-ab [aria-pressed="true"]').dataset.rt === "in" ? $("#rt-audio-in") : $("#rt-audio-stream"));
  const icon = (p) => { $(".rt-play-i").hidden = p; $(".rt-pause-i").hidden = !p; $("#rt-play").setAttribute("aria-label", p ? "Pause" : "Play"); };
  $$("#rt-ab button").forEach((b) => b.addEventListener("click", () => {
    const prev = pick(), playing = !prev.paused, t = prev.currentTime; prev.pause();
    pressOne("#rt-ab", b);
    const nx = pick(); try { nx.currentTime = t || 0; } catch { /* not ready */ } if (playing) nx.play().catch(() => {});
  }));
  $("#rt-play").addEventListener("click", () => { const a = pick(); if (a.paused) { a.play().catch(() => {}); icon(true); } else { a.pause(); icon(false); } });
  ["#rt-audio-in", "#rt-audio-stream"].forEach((s) => $(s).addEventListener("ended", () => icon(false)));
}

/* ── about ─────────────────────────────────── */
const COLS = { si_sdr_i: "SI-SDR gain", si_sdr: "SI-SDR", snr_i: "SNR gain", snr: "SNR", pesq: "PESQ", stoi: "STOI", estoi: "ESTOI", lsd: "LSD", rtf: "RTF", n: "Clips", label: "Model", method: "Method" };
const HI = new Set(["si_sdr_i", "si_sdr", "snr_i", "snr", "pesq", "stoi", "estoi"]), LO = new Set(["lsd", "rtf"]);
async function ensureAbout() {
  if (S.tables || !S.engines.length) return;
  $("#model-cards").innerHTML = S.engines.map((e) => {
    const b = e.benchmark || {}, st = (k, l, d) => `<div class="mc-stat"><b class="${b[k] < 0 ? "neg" : ""}">${b[k] == null ? "—" : k === "si_sdr_i" ? signed(b[k], 2) : (+b[k]).toFixed(d)}</b><span>${l}</span></div>`;
    return `<article class="model-card ${e.default ? "locked" : ""}"><h3>${esc(e.label)} <span class="tag ${e.default ? "tag-ok" : ""}">${e.default ? "chosen model" : e.family === "neural" ? "neural" : "classical"}</span></h3>
      <p>${esc(e.summary)}</p><div class="mc-stats">${st("si_sdr_i", "SI-SDR gain", 2)}${st("pesq", "PESQ", 2)}${st("stoi", "STOI", 3)}</div><span class="mc-set">${esc(b.set || "")}</span></article>`;
  }).join("");
  try {
    S.tables = (await api("/api/research")).tables;
    const vbOrig = S.tables.find((t) => t.key === "voicebank")?.rows.find((r) => r._baseline);
    if (vbOrig) {
      $("#model-cards").insertAdjacentHTML("afterbegin", `<article class="model-card orig-card"><h3>Original (no processing) <span class="tag">baseline</span></h3>
        <p>The noisy audio exactly as recorded. Every model's score is measured against this: a gain of 0 dB means no improvement.</p>
        <div class="mc-stats"><div class="mc-stat"><b>+0.00</b><span>SI-SDR gain</span></div><div class="mc-stat"><b>${(+vbOrig.pesq).toFixed(2)}</b><span>PESQ</span></div><div class="mc-stat"><b>${(+vbOrig.stoi).toFixed(3)}</b><span>STOI</span></div></div>
        <span class="mc-set">VoiceBank 824</span></article>`);
    }
    $("#table-tabs").innerHTML = S.tables.map((t, i) => `<button type="button" role="tab" aria-selected="${i === 0}" data-t="${t.key}">${esc(t.title.split(" — ")[0])}</button>`).join("");
    $$("#table-tabs button").forEach((b) => b.addEventListener("click", () => { $$("#table-tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b))); showTable(b.dataset.t); }));
    showTable(S.tables[0].key);
  } catch (e) { $("#research-table").innerHTML = `<p class="hint">Could not load the result tables: ${esc(e.message)}</p>`; }
}
function showTable(key) {
  const t = S.tables.find((x) => x.key === key);
  let cols = t.columns.filter((c) => !(c === "method" && t.columns.includes("label")));
  cols = cols.filter((c) => t.rows.some((r) => r[c] !== "" && r[c] !== "nan"));
  const best = {};
  cols.forEach((c) => { const v = t.rows.filter((r) => !r._baseline).map((r) => parseFloat(r[c])).filter(isFinite); if (v.length && HI.has(c)) best[c] = Math.max(...v); if (v.length && LO.has(c)) best[c] = Math.min(...v); });
  const f = (c, v) => {
    const x = parseFloat(v);
    if (!isFinite(x) || c === "label" || c === "method") return esc(v === "nan" ? "—" : v);
    const d = c === "n" ? 0 : ["stoi", "estoi", "rtf"].includes(c) ? 3 : 2;
    return (Math.abs(x) < 0.5 * 10 ** -d ? 0 : x).toFixed(d);  // no "-0.00"
  };
  $("#research-table").innerHTML = `<h2>${esc(t.title)}</h2><p class="hint">${esc(t.note)}</p>
    <div class="table-wrap"><table class="tbl"><thead><tr>${cols.map((c) => `<th>${COLS[c] || esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${t.rows.map((r) => `<tr class="${r._baseline ? "orig-row" : ""}">${cols.map((c) => `<td class="${!r._baseline && parseFloat(r[c]) === best[c] ? "best" : ""}">${f(c, r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="hint">Grey italic rows are the original, unprocessed audio for comparison. Green marks the best value in each column.</p>`;
}
function wireAbout() {
  $$("#about-tabs button").forEach((b) => b.addEventListener("click", () => {
    $$("#about-tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $$("[data-aboutpage]").forEach((p) => { p.hidden = p.dataset.aboutpage !== b.dataset.about; });
  }));
}

/* ── init ──────────────────────────────────── */
let glowRAF = null;
window.addEventListener("pointermove", (e) => {
  if (glowRAF || REDUCED.matches) return;
  glowRAF = requestAnimationFrame(() => {
    document.documentElement.style.setProperty("--mx", `${e.clientX}px`);
    document.documentElement.style.setProperty("--my", `${e.clientY}px`);
    glowRAF = null;
  });
}, { passive: true });
async function init() {
  wireTheme(); wireInput(); wireDrawer(); wirePlayer(); wireRealtime(); wireAbout();
  $("#btn-enhance").addEventListener("click", enhance);
  updateClipCard();
  route();
  pollStatus();
  setInterval(pollStatus, 8000);
  try { await Promise.all([loadEngines(), loadPresets(), loadSamples()]); }
  catch (e) { toast(`Could not reach the SONA server: ${e.message}`, true, 8000); }
  refreshMemory();
  updateSummary();
  if (S.route === "about") ensureAbout();
}
document.addEventListener("DOMContentLoaded", init);
