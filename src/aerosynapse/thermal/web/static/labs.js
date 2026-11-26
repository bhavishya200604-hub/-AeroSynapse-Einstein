/* THERMAL LAB, Phase 10.4-10.7: flow lab, data factory, PINN training on real data, downloads.
   Every number comes from the API (src/web/labs.py, src/web/server.py); this file collects inputs and draws.
   The particle tracers of the flow lab are a visualisation of the solver's velocity field (bilinear interpolation). */
"use strict";

// ------------------------------------------------------------------------------------------------ requests and downloads
const LAST = {}; // last request and time per route: "Results (ZIP)" exports what is on screen
const _api0 = api;
// eslint-disable-next-line no-global-assign
api = async function (path, body) { if (body !== undefined) LAST[path] = { body, at: Date.now() }; return _api0(path, body); };

async function download(path, body, fallback) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.error || `HTTP ${r.status}`); }
  const blob = await r.blob();
  const m = /filename="([^"]+)"/.exec(r.headers.get("Content-Disposition") || "");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = m ? m[1] : fallback;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 3000);
}
const ROUTE_OF = { solve: "/api/solve", converge: "/api/converge", inverse: "/api/inverse", flow: "/api/flow" };
async function exportKind(kind, btn) {
  let body;
  if (kind === "dataset") { if (!DS.id) throw new Error("generate a dataset first"); body = { kind, id: DS.id }; }
  else if (kind === "train") { if (!TR.id || TR.state !== "done") throw new Error("export a training once it has finished"); body = { kind, id: TR.id }; }
  else if (kind === "real") {
    const f = LAST["/api/real/fit"], l = LAST["/api/real/lumped"];
    if (!f && !l) throw new Error("run a fit first");
    body = (!l || (f && f.at > l.at)) ? { kind: "real_fit", request: f.body } : { kind: "lumped", request: l.body };
  } else {
    const last = LAST[ROUTE_OF[kind]];
    if (!last) throw new Error("run a computation in this tab first");
    body = { kind, request: last.body };
  }
  const old = btn.textContent; btn.disabled = true; btn.textContent = "⏳ preparing…";
  try { await download("/api/export", body, `thermal_lab_${kind}.zip`); toast("Downloaded: data, figures and reproduce.py"); }
  finally { btn.disabled = false; btn.textContent = old; }
}
function bindExports(root = document) {
  $$("[data-export]", root).forEach((b) => { if (!b.dataset.bound) { b.dataset.bound = "1"; b.addEventListener("click", () => exportKind(b.dataset.export, b).catch((e) => toast(e.message, true))); } });
}

// section headers for the first tabs (title, purpose, download), so every tab reads the same way
const HEADS = {
  sim: ["Simulator", "verified solvers", "Four heat-transfer cases solved now by the project's verified solvers, with the trained PINN ensembles side by side and validity warnings outside their training range.", "solve"],
  verify: ["Verification", "order of accuracy", "Grid convergence against exact solutions proves the solvers; the parameter sweep shows where each PINN can be trusted.", "converge"],
  inverse: ["Inverse lab", "identifiability", "Identify thermal diffusivity, conductivity and heat-transfer coefficient from synthetic sensors; the Fisher information decides what can be estimated before any fit.", "inverse"],
  real: ["Real data", "NOAA soil profiles", "Fit soil thermal diffusivity to measured temperatures (or your own table), with uncertainty, hold-out validation against a no-physics baseline and the full convergence log.", "real"],
};
function addHeads() {
  Object.entries(HEADS).forEach(([tab, [title, tag, text, kind]]) => {
    const sec = $(`#tab-${tab}`);
    if (!sec || $(".section-head", sec)) return;
    const h = document.createElement("div");
    h.className = "section-head";
    h.innerHTML = `<div><h2>${title} <span class="tag">${tag}</span></h2><p class="muted">${text}</p></div>
      <div class="row-tight"><button class="btn" data-export="${kind}">⤓ Results (ZIP)</button></div>`;
    sec.prepend(h);
  });
}

// ------------------------------------------------------------------------------------------------ colour maps (canvas)
const VIRIDIS = [[0, "rgb(68,1,84)"], [0.25, "rgb(59,82,139)"], [0.5, "rgb(33,145,140)"], [0.75, "rgb(94,201,98)"], [1, "rgb(253,231,37)"]];
const DIVERGE = [[0, "rgb(5,48,97)"], [0.25, "rgb(67,147,195)"], [0.5, "rgb(247,247,247)"], [0.75, "rgb(214,96,77)"], [1, "rgb(103,0,31)"]];
function lut(stops) {
  const s = stops.map(([t, c]) => [t, c.match(/\d+/g).map(Number)]);
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 0;
    while (k < s.length - 2 && t > s[k + 1][0]) k++;
    const [t0, c0] = s[k], [t1, c1] = s[k + 1], w = (t - t0) / ((t1 - t0) || 1);
    for (let j = 0; j < 3; j++) out[3 * i + j] = c0[j] + w * (c1[j] - c0[j]);
  }
  return out;
}
const LUT = { inferno: lut(INFERNO), viridis: lut(VIRIDIS), diverge: lut(DIVERGE) };
const css_grad = (stops) => `linear-gradient(90deg, ${stops.map(([t, c]) => `${c} ${t * 100}%`).join(", ")})`;

// ------------------------------------------------------------------------------------------------ flow lab
const FL = { r: null, frame: 0, playing: true, field: "theta", tick: 0, parts: [], raf: 0, scale: {}, showP: true };
const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function flowBody() {
  const v = (id) => Number($(id).value);
  return { heating: $("#fl-heating").value, lid: $("#fl-lid").value, lid_speed: v("#fl-speed"), Re: v("#fl-Re"), Pr: v("#fl-Pr"),
    Ri: v("#fl-Ri"), Bi: v("#fl-Bi"), grid: v("#fl-grid"), t_end: v("#fl-tend"), frames: v("#fl-frames") };
}
function flowDefaults() {
  const d = { "#fl-heating": "lid_hot", "#fl-lid": "cosh", "#fl-speed": 1, "#fl-Re": 400, "#fl-Pr": 0.71, "#fl-Ri": 0, "#fl-Bi": 1, "#fl-grid": 32, "#fl-tend": 8, "#fl-frames": 24 };
  Object.entries(d).forEach(([k, v]) => { $(k).value = v; });
}
function speedOf(r, k) { return r.u[k].map((row, j) => row.map((u, i) => Math.hypot(u, r.v[k][j][i]))); }
function fieldOf(k) {
  const r = FL.r;
  if (FL.field === "theta") return { z: r.theta[k], lut: "inferno", cs: INFERNO };
  if (FL.field === "speed") return { z: speedOf(r, k), lut: "viridis", cs: VIRIDIS };
  if (FL.field === "vorticity") return { z: r.vorticity[k], lut: "diverge", cs: DIVERGE, sym: true };
  return { z: r.psi[k], lut: "diverge", cs: DIVERGE, sym: true };
}
function fieldRange(name) {
  const r = FL.r, key = name;
  if (FL.scale[key]) return FL.scale[key];
  const vals = [];
  for (let k = 0; k < r.t.length; k++) { fieldOf(k).z.forEach((row) => row.forEach((v) => vals.push(v))); }
  vals.sort((a, b) => a - b);
  const q = (p) => vals[Math.min(vals.length - 1, Math.max(0, Math.round(p * (vals.length - 1))))];
  let lo = q(0.005), hi = q(0.995);
  if (name === "vorticity" || name === "psi") { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m; }
  if (hi - lo < 1e-12) hi = lo + 1;
  return (FL.scale[key] = [lo, hi]);
}
function drawField() {
  const r = FL.r; if (!r) return;
  const cv = $("#fl-canvas"), ctx = cv.getContext("2d");
  const f = fieldOf(FL.frame), z = f.z, ny = z.length, nx = z[0].length, [lo, hi] = fieldRange(FL.field);
  const off = FL.off && FL.off.width === nx && FL.off.height === ny ? FL.off : (FL.off = Object.assign(document.createElement("canvas"), { width: nx, height: ny }));
  const octx = off.getContext("2d"), img = octx.createImageData(nx, ny), L = LUT[f.lut];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const c = Math.max(0, Math.min(255, Math.round(255 * (z[j][i] - lo) / (hi - lo))));
    const p = 4 * ((ny - 1 - j) * nx + i); // row 0 of the field is the bottom wall
    img.data[p] = L[3 * c]; img.data[p + 1] = L[3 * c + 1]; img.data[p + 2] = L[3 * c + 2]; img.data[p + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(off, 0, 0, cv.width, cv.height);
  $("#fl-cbar").style.background = css_grad(f.cs);
  $("#fl-cmin").textContent = fmt(lo, 3); $("#fl-cmax").textContent = fmt(hi, 3);
  $("#fl-time").textContent = `t = ${fmt(r.t[FL.frame], 3)}`;
  $("#fl-frame").value = FL.frame;
}
function velAt(x, y, k) {
  const r = FL.r, xs = r.xv, ys = r.yv, n = xs.length, m = ys.length;
  const fx = Math.max(0, Math.min(n - 1.001, (x - xs[0]) / (xs[1] - xs[0])));
  const fy = Math.max(0, Math.min(m - 1.001, (y - ys[0]) / (ys[1] - ys[0])));
  const i = Math.floor(fx), j = Math.floor(fy), a = fx - i, b = fy - j;
  const U = r.u[k], V = r.v[k];
  const bil = (A) => (1 - a) * (1 - b) * A[j][i] + a * (1 - b) * A[j][i + 1] + (1 - a) * b * A[j + 1][i] + a * b * A[j + 1][i + 1];
  return [bil(U), bil(V)];
}
function seedParticles(n = 900) {
  FL.parts = Array.from({ length: n }, () => ({ x: Math.random(), y: Math.random(), age: Math.floor(Math.random() * 200) }));
  const ctx = $("#fl-particles").getContext("2d"); ctx.clearRect(0, 0, 640, 640);
}
function stepParticles() {
  const r = FL.r, cv = $("#fl-particles"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.globalCompositeOperation = "destination-out"; ctx.fillStyle = "rgba(0,0,0,0.10)"; ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "source-over";
  if (!FL.showP) { ctx.clearRect(0, 0, W, H); return; }
  const k = FL.frame, vmax = FL.vmax || 1, gain = 0.0035 / vmax;
  ctx.strokeStyle = FL.field === "theta" ? "rgba(235,245,255,0.85)" : "rgba(20,20,20,0.75)"; ctx.lineWidth = 1.1;
  ctx.beginPath();
  for (const p of FL.parts) {
    const [u, v] = velAt(p.x, p.y, k);
    const x2 = p.x + gain * u, y2 = p.y + gain * v;
    ctx.moveTo(p.x * W, (1 - p.y) * H); ctx.lineTo(x2 * W, (1 - y2) * H);
    p.x = x2; p.y = y2; p.age++;
    if (p.age > 240 || p.x < 0.002 || p.x > 0.998 || p.y < 0.002 || p.y > 0.998 || Math.hypot(u, v) < 1e-4 * vmax) {
      p.x = 0.02 + 0.96 * Math.random(); p.y = 0.02 + 0.96 * Math.random(); p.age = 0;
    }
  }
  ctx.stroke();
}
function flowLoop() {
  FL.raf = 0;
  if (!FL.r || !$("#tab-flow").classList.contains("active")) return;
  FL.tick++;
  if (FL.playing && FL.tick % 9 === 0) { FL.frame = (FL.frame + 1) % FL.r.t.length; drawField(); }
  stepParticles();
  FL.raf = requestAnimationFrame(flowLoop);
}
function startLoop() { if (!FL.raf) FL.raf = requestAnimationFrame(flowLoop); }
function drawPsi() {
  const r = FL.r, k = FL.frame;
  draw("fl-psi", [{ z: r.psi[k], x: r.x, y: r.y, type: "contour", colorscale: DIVERGE, ncontours: 24, contours: { coloring: "lines" }, line: { width: 1.4 }, showscale: false, hoverinfo: "z" }],
    { xaxis: { title: "x", range: [0, 1], constrain: "domain" }, yaxis: { title: "y", range: [0, 1], scaleanchor: "x" }, margin: { l: 46, r: 10, t: 10, b: 40 }, showlegend: false });
}
function renderFlow(r) {
  FL.r = r; FL.scale = {}; FL.frame = r.t.length - 1;
  let vm = 0; r.u.forEach((U, k) => U.forEach((row, j) => row.forEach((u, i) => { vm = Math.max(vm, Math.hypot(u, r.v[k][j][i])); }))); FL.vmax = vm || 1;
  $("#fl-frame").max = r.t.length - 1;
  drawField(); seedParticles();
  const p = r.params;
  $("#fl-chips").innerHTML = [["Re", p.Re], ["Pr", p.Pr], ["Pe = Re·Pr", p.Re * p.Pr], ["Ri", p.Ri], ["grid", `${r.grid}²`], ["time steps", r.steps],
    ["max |div u|", fmt(r.max_div, 2)], ["solve", `${fmt(r.wall_s, 3)} s`]].map(([k, v]) => `<span class="chip">${k}<b>${typeof v === "number" ? (Number.isInteger(v) ? v : fmt(v, 4)) : v}</b></span>`).join("");
  const tr = [{ x: r.t, y: r.nu_top, name: "Nu top", line: { color: ROLE.solver, width: 2 } }, { x: r.t, y: r.nu_bottom, name: "Nu bottom", line: { color: ROLE.pinn, width: 2 } }];
  if (r.heating === "side_heated") tr.push({ x: r.t, y: r.nu_left, name: "Nu left", line: { color: SERIES[3], width: 2 } }, { x: r.t, y: r.nu_right, name: "Nu right", line: { color: SERIES[4], width: 2 } });
  tr.push({ x: r.t, y: r.ke, name: "kinetic energy", xaxis: "x", yaxis: "y2", line: { color: ROLE.data, width: 2 } });
  draw("fl-nu", tr, { grid: { rows: 2, columns: 1, roworder: "top to bottom" }, xaxis: { title: "t", anchor: "y2" }, yaxis: { title: "Nu", domain: [0.55, 1] }, yaxis2: { title: "KE", domain: [0, 0.42] }, margin: { l: 56, r: 10, t: 30, b: 40 } });
  drawPsi();
  FL.playing = !reduceMotion; $("#fl-play").textContent = FL.playing ? "❚❚" : "▶";
  startLoop();
}
async function runFlow() {
  if (FL.loading) return FL.loading; // one simulation at a time from this page
  FL.loading = busy("#fl-busy", () => api("/api/flow", flowBody()));
  try { const r = await FL.loading; renderFlow(r); return r; } finally { FL.loading = null; }
}

// ------------------------------------------------------------------------------------------------ data factory
const DS = { id: null, r: null };
const LAYOUTS = { 1: ["uniform", "random", "clustered_right", "clustered_left"], 2: ["grid", "sobol", "wall_clustered", "centrelines"] };
const SPLIT_COL = { train: ROLE.solver, val: ROLE.data, interp: ROLE.pinn, extrap: SERIES[3] };
function dsCase() { return S.info.data_cases[$("#ds-case").value]; }
function dsFill(reset = true) {
  DS.filled = true;
  const key = $("#ds-case").value, c = dsCase(), is2d = key === "caseC";
  const params = Object.entries(c.parameters);
  $("#ds-ranges").innerHTML = `<label class="field">parameter ranges (the design is log-uniform in each)</label><div class="range-pair">` +
    params.map(([n, v]) => typeof v === "object"
      ? `<b>${n}</b><input type="number" data-lo="${n}" value="${v.lo}"><input type="number" data-hi="${n}" value="${v.hi}">`
      : `<b>${n}</b><input type="number" data-fixed="${n}" value="${v}"><span class="faint">fixed</span>`).join("") + "</div>";
  if (!reset) return;
  let sz = { ...c.sizes };
  const cap = is2d ? S.info.labs.points_2d : S.info.labs.points_1d, tot = Object.values(sz).reduce((a, b) => a + b, 0);
  if (tot > cap) { const f = cap / tot; sz = Object.fromEntries(Object.entries(sz).map(([k, v]) => [k, Math.max(k === "train" ? 1 : 0, Math.floor(v * f))])); }
  ["train", "val", "interp", "extrap"].forEach((k) => { $(`#ds-${k}`).value = sz[k]; });
  $("#ds-res-label").textContent = is2d ? "grid (cells / side)" : "grid points nx";
  $("#ds-res").value = is2d ? Math.min(24, c.resolution.grid) : c.resolution.nx;
  $("#ds-nt").value = c.times.n;
  $("#ds-layout").innerHTML = LAYOUTS[is2d ? 2 : 1].map((l) => `<option>${l}</option>`).join("");
  $("#ds-ns").value = is2d ? 16 : 8;
  const d = S.info.data_defaults;
  $("#ds-method").value = d.design.method; $("#ds-ulim").value = d.design.unit_train_limit; $("#ds-seed").value = d.seed;
  $("#ds-noise").value = d.noise.model; $("#ds-levels").value = d.noise.levels.join(", ");
}
function dsBody() {
  const key = $("#ds-case").value, is2d = key === "caseC", ranges = {}, fixed = {};
  $$("#ds-ranges [data-lo]").forEach((e) => { ranges[e.dataset.lo] = { lo: Number(e.value), hi: Number($(`#ds-ranges [data-hi="${e.dataset.lo}"]`).value) }; });
  $$("#ds-ranges [data-fixed]").forEach((e) => { fixed[e.dataset.fixed] = Number(e.value); });
  const b = { case: key, ranges, fixed, sizes: Object.fromEntries(["train", "val", "interp", "extrap"].map((k) => [k, Number($(`#ds-${k}`).value)])),
    method: $("#ds-method").value, u_lim: Number($("#ds-ulim").value), seed: Number($("#ds-seed").value), n_times: Number($("#ds-nt").value),
    layout: $("#ds-layout").value, n_sensors: Number($("#ds-ns").value), noise_model: $("#ds-noise").value,
    noise_levels: $("#ds-levels").value.split(/[,\s]+/).filter(Boolean).map(Number) };
  b[is2d ? "grid" : "nx"] = Number($("#ds-res").value);
  return b;
}
function renderDataset(r) {
  DS.id = r.id; DS.r = r;
  const n = r.design.length, mb = r.bytes / 2 ** 20;
  $("#ds-chips").innerHTML = [`<span class="chip">design points<b>${n}</b></span>`,
    ...Object.entries(r.shapes).map(([k, v]) => `<span class="chip">${k}<b>${v.join("×")}</b></span>`),
    `<span class="chip">size<b>${fmt(mb, 3)} MB</b></span>`, `<span class="chip">generation<b>${fmt(r.wall_s, 3)} s</b></span>`,
    `<span class="chip">split leakage<b>${r.leakage.all_ok ? "none ✓" : "CHECK"}</b></span>`].join("");
  const names = r.dims.map((d) => d.name);
  const tr = Object.keys(SPLIT_COL).map((sp) => {
    const pts = r.design.filter((d) => d.split === sp);
    return { x: pts.map((d) => d.params[names[0]]), y: pts.map((d) => names.length > 1 ? d.params[names[1]] : sp), mode: "markers", type: "scatter", name: `${sp} (${pts.length})`,
      marker: { color: SPLIT_COL[sp], size: names.length > 2 ? pts.map((d) => 7 + 5 * Math.log10(d.params[names[2]] / r.dims[2].lo + 1)) : 10, line: { width: 1, color: css("--bg") || "#000" } },
      hovertext: pts.map((d) => Object.entries(d.params).map(([k, v]) => `${k} = ${fmt(v, 4)}`).join("<br>")), hoverinfo: "text" };
  });
  draw("ds-design", tr, { xaxis: { title: names[0], type: r.dims[0].scale === "log" ? "log" : "linear" },
    yaxis: names.length > 1 ? { title: names[1], type: r.dims[1].scale === "log" ? "log" : "linear" } : { title: "split", type: "category" },
    title: names.length > 2 ? `marker size: ${names[2]}` : "" });
  const pv = r.preview, pp = Object.entries(r.preview_point.params).map(([k, v]) => `${k} = ${fmt(v, 3)}`).join(", ");
  $("#ds-prev-title").textContent = `Ground truth, training point ${pp}`;
  if (pv.y) draw("ds-prev", [heat(pv.theta, pv.x, pv.y, "θ")], { xaxis: { title: "x" }, yaxis: { title: "y", scaleanchor: "x" }, margin: { l: 46, r: 10, t: 10, b: 40 } });
  else draw("ds-prev", pv.theta.map((row, k) => ({ x: pv.x, y: row, mode: "lines", name: `t = ${fmt(pv.t[k], 2)}`, line: { width: 1.6, color: `hsl(${200 + 140 * k / Math.max(1, pv.t.length - 1)},70%,60%)` } })),
    { xaxis: { title: "x" }, yaxis: { title: "θ" }, showlegend: false });
  const s = r.sensors, last = (a) => Array.isArray(a[0]) ? a[a.length - 1] : a;
  if (Array.isArray(s.positions[0])) {
    draw("ds-sens", [{ x: s.positions.map((p) => p[0]), y: s.positions.map((p) => p[1]), mode: "markers", type: "scatter", name: "sensors",
      marker: { size: 11, color: last(s.clean), colorscale: INFERNO, showscale: true, colorbar: { thickness: 10, title: { text: "θ" } } } }],
    { xaxis: { title: "x", range: [0, 1] }, yaxis: { title: "y", range: [0, 1], scaleanchor: "x" }, showlegend: false, margin: { l: 46, r: 10, t: 10, b: 40 } });
  } else {
    const tr2 = [{ x: s.positions, y: last(s.clean), mode: "lines+markers", name: "clean", line: { color: css("--ink") || "#fff", width: 1.5 } }];
    s.noisy.forEach((lv, k) => { if (s.levels[k] > 0) tr2.push({ x: s.positions, y: last(lv), mode: "markers", name: `noise ${s.levels[k]}`, marker: { size: 7, color: SERIES[k % SERIES.length] } }); });
    draw("ds-sens", tr2, { xaxis: { title: `x (${s.layout}, last snapshot)` }, yaxis: { title: "θ" }, legend: { orientation: "h", y: 1.28, x: 0 }, margin: { l: 56, r: 10, t: 64, b: 46 } });
  }
}
async function runDataset() {
  if (DS.loading) return DS.loading;
  DS.loading = busy("#ds-busy", () => api("/api/dataset", dsBody()));
  try { const r = await DS.loading; renderDataset(r); return r; } finally { DS.loading = null; }
}

// ------------------------------------------------------------------------------------------------ PINN training
const TR = { id: null, state: null, timer: 0, cov: null };
function trainInit() {
  // one shared promise: every caller waits for the same set-up (a second call must not run ahead of the first)
  if (TR.ready) return TR.ready;
  TR.ready = (async () => {
    const sd = S.info.soil_defaults;
    TR.top = sd.model.top_m; TR.bottom = sd.model.bottom_m;
    $("#tr-station").innerHTML = S.info.stations.map((s) => `<option value="${s.station}|${s.year}" ${s.default ? "selected" : ""}>${s.station.replaceAll("_", " ")} (${s.year})</option>`).join("");
    try { const r = await api("/api/real/stations", {}); TR.cov = Object.fromEntries(r.stations.map((s) => [`${s.station}|${s.year}`, s])); } catch (_) { /* coverage optional */ }
    trainSensors();
    $("#tr-station").addEventListener("change", trainSensors);
    if (S.info.mode === "public") { $("#tr-adam").value = Math.min(2500, S.info.labs.train_adam); $("#tr-lbfgs").value = Math.min(200, S.info.labs.train_lbfgs); }
  })();
  return TR.ready;
}
function trainSensors() {
  const c = TR.cov && TR.cov[$("#tr-station").value];
  const depths = (c ? c.depths_m : [0.05, 0.1, 0.2, 0.5, 1.0]).filter((z) => z > TR.top + 1e-9 && z < TR.bottom - 1e-9);
  $("#tr-sensors").innerHTML = `<span class="chip sel on fixed">${fmt(TR.top * 100, 3)} cm (top)</span>` +
    depths.map((z, i) => `<span class="chip sel ${i === depths.length - 1 ? "on" : ""}" data-z="${z}">${fmt(z * 100, 3)} cm</span>`).join("") +
    `<span class="chip sel on fixed">${fmt(TR.bottom * 100, 3)} cm (bottom)</span>`;
  $$("#tr-sensors [data-z]").forEach((e) => e.addEventListener("click", () => { e.classList.toggle("on"); trainHoldouts(); }));
  trainHoldouts();
}
function trainHoldouts() {
  const off = $$("#tr-sensors [data-z]").filter((e) => !e.classList.contains("on")).map((e) => Number(e.dataset.z));
  $("#tr-holdout").innerHTML = `<option value="">none</option>` + off.map((z, i) => `<option value="${z}" ${i === 0 ? "selected" : ""}>${fmt(z * 100, 3)} cm</option>`).join("");
}
function trainBody() {
  const [st, yr] = $("#tr-station").value.split("|"), v = (id) => Number($(id).value);
  return { source: "station", station: st, year: Number(yr), start: $("#tr-start").value, days: v("#tr-days"), top_m: TR.top, bottom_m: TR.bottom,
    train_inner: $$("#tr-sensors [data-z].on").map((e) => Number(e.dataset.z)), holdout: $("#tr-holdout").value === "" ? null : Number($("#tr-holdout").value),
    width: v("#tr-width"), depth: v("#tr-depth"), harmonics: v("#tr-harm"), n_colloc: v("#tr-colloc"), warmup: v("#tr-warm"),
    adam_steps: v("#tr-adam"), lr: v("#tr-lr"), lbfgs_steps: v("#tr-lbfgs"), w_pde: v("#tr-wpde"), seed: v("#tr-seed") };
}
function trainCurves(st) {
  const h = st.history, x = h.map((r) => r.step);
  draw("tr-loss", [{ x, y: h.map((r) => r.data), name: "data misfit", line: { color: ROLE.data, width: 2 } }, { x, y: h.map((r) => r.pde), name: "heat-equation residual", line: { color: ROLE.pinn, width: 2 } }],
    { xaxis: { title: "step" }, yaxis: { title: "loss (scaled)", type: "log", exponentformat: "power" }, margin: { l: 60, r: 10, t: 30, b: 40 } });
  const tr = [{ x, y: h.map((r) => r.alpha), name: "PINN α", line: { color: ROLE.pinn, width: 2 } }];
  if (st.result && st.result.classical) tr.push({ x: [x[0], x[x.length - 1]], y: [st.result.classical.alpha, st.result.classical.alpha], name: "classical α (same sensors)", line: { color: ROLE.solver, dash: "dash", width: 2 } });
  draw("tr-alpha", tr, { xaxis: { title: "step" }, yaxis: { title: "α [m²/s]", exponentformat: "e" }, margin: { l: 70, r: 10, t: 30, b: 40 } });
}
function trainResult(st) {
  const r = st.result, s = st.series, nz = s.z.length, chips = [`<span class="chip">PINN α<b>${fmt(r.alpha, 3)} m²/s</b></span>`];
  if (r.classical) chips.push(`<span class="chip">classical α<b>${fmt(r.classical.alpha, 3)} m²/s</b></span>`,
    `<span class="chip">held-out RMSE PINN<b>${fmt(r.rmse[String(st.holdout)], 3)} K</b></span>`, `<span class="chip">held-out RMSE classical<b>${fmt(r.classical.rmse_holdout, 3)} K</b></span>`);
  chips.push(`<span class="chip">parameters<b>${r.n_parameters}</b></span>`, `<span class="chip">training<b>${fmt(r.wall_s, 3)} s</b></span>`);
  $("#tr-chips").innerHTML = chips.join("");
  const data = [], lay = { grid: { rows: nz, columns: 1, pattern: "coupled", roworder: "top to bottom" }, xaxis: { title: "hours since the window start" }, showlegend: true, margin: { l: 60, r: 10, t: 30, b: 46 } };
  s.z.forEach((z, j) => {
    const ax = j === 0 ? "" : String(j + 1), held = st.holdout !== null && Math.abs(z - st.holdout) < 1e-9;
    data.push({ x: s.t_h, y: s.obs.map((row) => row[j]), name: "measured", legendgroup: "m", showlegend: j === 0, xaxis: "x", yaxis: "y" + ax, line: { color: ROLE.data, width: 1.4 } });
    data.push({ x: s.t_h, y: s.pinn.map((row) => row[j]), name: "PINN", legendgroup: "p", showlegend: j === 0, xaxis: "x", yaxis: "y" + ax, line: { color: ROLE.pinn, width: 1.4 } });
    if (held && s.classical_holdout) data.push({ x: s.t_h, y: s.classical_holdout, name: "classical (held-out sensor)", xaxis: "x", yaxis: "y" + ax, line: { color: ROLE.solver, dash: "dash", width: 1.4 } });
    lay["yaxis" + ax] = { title: { text: `${fmt(z * 100, 3)} cm${held ? " · held out" : ""}`, font: { size: 11 } } };
  });
  draw("tr-sens", data, lay);
}
async function pollTrain() {
  const st = await _api0("/api/train/status", { id: TR.id });
  TR.state = st.state;
  $("#tr-bar").style.width = `${Math.min(100, 100 * st.step / Math.max(1, st.total))}%`;
  if (st.history.length) {
    const h = st.history[st.history.length - 1];
    if (st.state !== "done") $("#tr-chips").innerHTML = [["state", st.state], ["step", `${h.step} / ${st.total}`], ["phase", h.phase], ["α", `${fmt(h.alpha, 3)} m²/s`], ["data", fmt(h.data, 3)], ["residual", fmt(h.pde, 3)], ["elapsed", `${fmt(h.wall_s, 3)} s`]]
      .map(([k, v]) => `<span class="chip">${k}<b>${v}</b></span>`).join("");
    trainCurves(st);
  }
  if (["done", "error", "cancelled"].includes(st.state)) {
    clearInterval(TR.timer); TR.timer = 0; $("#tr-run").disabled = false; $("#tr-cancel").disabled = true;
    if (st.state === "done") { $("#tr-bar").style.width = "100%"; trainResult(st); toast("Training finished."); }
    else toast(st.state === "error" ? st.error : "Training cancelled.", st.state === "error", 9000);
  }
  return st;
}
async function runTrain() {
  await trainInit();
  const r = await api("/api/train/start", trainBody());
  TR.id = r.id; TR.state = r.state;
  $("#tr-run").disabled = true; $("#tr-cancel").disabled = false; $("#tr-bar").style.width = "0";
  $("#tr-chips").innerHTML = `<span class="chip">data<b>${r.data}</b></span><span class="chip">state<b>queued</b></span>`;
  if (TR.timer) clearInterval(TR.timer);
  TR.timer = setInterval(() => pollTrain().catch((e) => { clearInterval(TR.timer); TR.timer = 0; $("#tr-run").disabled = false; toast(e.message, true); }), 1000);
  return r;
}
function trainDefaults() {
  const d = { "#tr-start": "2023-07-01", "#tr-days": 14, "#tr-width": 32, "#tr-depth": 3, "#tr-harm": 2, "#tr-colloc": 1500, "#tr-warm": 500, "#tr-adam": S.info.mode === "public" ? 2500 : 3000,
    "#tr-lr": 0.003, "#tr-lbfgs": S.info.mode === "public" ? 200 : 300, "#tr-wpde": 1, "#tr-seed": 0 };
  Object.entries(d).forEach(([k, v]) => { $(k).value = v; });
  const def = S.info.stations.find((s) => s.default); if (def) $("#tr-station").value = `${def.station}|${def.year}`;
  trainSensors();
}

// ------------------------------------------------------------------------------------------------ wiring
const _show0 = showTab;
// eslint-disable-next-line no-global-assign
showTab = function (name) {
  _show0(name);
  if (name === "flow") { if (FL.r) { drawField(); startLoop(); } else runFlow().catch(() => {}); }
  if (name === "data" && !DS.r) { if (!DS.filled) dsFill(); runDataset().catch(() => {}); }
  if (name === "train") trainInit();
};
window.LABS = {
  flowDefault: async () => { flowDefaults(); await runFlow(); },
  dataDefault: async () => { $("#ds-case").value = "caseB"; dsFill(); await runDataset(); },
  trainDefault: async () => { await trainInit(); trainDefaults(); await runTrain(); },
};

(function wire() {
  addHeads(); bindExports();
  $$(".lab-card[data-go]").forEach((b) => b.addEventListener("click", () => { showTab(b.dataset.go); window.scrollTo({ top: 0 }); }));
  $("#fl-run").addEventListener("click", () => runFlow().catch(() => {}));
  $("#fl-default").addEventListener("click", () => window.LABS.flowDefault().catch(() => {}));
  $("#fl-play").addEventListener("click", () => { FL.playing = !FL.playing; $("#fl-play").textContent = FL.playing ? "❚❚" : "▶"; if (!FL.playing && FL.r) drawPsi(); startLoop(); });
  $("#fl-frame").addEventListener("input", (e) => { FL.frame = Number(e.target.value); FL.playing = false; $("#fl-play").textContent = "▶"; drawField(); drawPsi(); });
  $("#fl-show-p").addEventListener("change", (e) => { FL.showP = e.target.checked; });
  $$("#fl-field button").forEach((b) => b.addEventListener("click", () => {
    $$("#fl-field button").forEach((x) => x.classList.toggle("on", x === b)); FL.field = b.dataset.f; drawField();
  }));
  $("#ds-case").addEventListener("change", () => dsFill());
  $("#ds-run").addEventListener("click", () => runDataset().catch(() => {}));
  $("#ds-default").addEventListener("click", () => window.LABS.dataDefault().catch(() => {}));
  $("#tr-run").addEventListener("click", () => runTrain().catch((e) => toast(e.message, true)));
  $("#tr-cancel").addEventListener("click", () => TR.id && _api0("/api/train/status", { id: TR.id, cancel: true }).catch(() => {}));
  $("#tr-default").addEventListener("click", () => window.LABS.trainDefault().catch((e) => toast(e.message, true)));
  $("#dl-code").addEventListener("click", () => download("/api/download/code", undefined, "thermal_lab_code.zip").then(() => toast("Downloaded the project's code.")).catch((e) => toast(e.message, true)));
  if (reduceMotion) { FL.showP = false; $("#fl-show-p").checked = false; }
  // wait for /api/info (loaded by app.js) before filling the data factory and the footer
  const ready = setInterval(() => {
    if (!S.info) return;
    clearInterval(ready);
    if (!DS.filled) dsFill();
    $("#foot-mode").textContent = `Thermal Lab · ${S.info.mode === "public" ? "public server (Hugging Face Space)" : "local server"} · build ${S.info.build || "–"}`;
  }, 100);
})();
