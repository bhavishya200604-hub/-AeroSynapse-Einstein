/* THERMAL LAB front end. Everything numerical comes from the local API (src/web/server.py): verified solvers, saved
   PINN ensembles, estimators. This file only collects inputs, converts physical units for display and draws. */
"use strict";

// ------------------------------------------------------------------------------------------------ utilities
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "–" :
  (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? Number(v).toExponential(d - 1) : Number(v).toPrecision(d);
const ROLE = { exact: null, solver: "#56B4E9", pinn: "#E69F00", data: "#2FD3A0", base: "#9AA3B5" };
const SERIES = ["#56B4E9", "#E69F00", "#2FD3A0", "#CC79A7", "#F0E442", "#FF7A45", "#7AA6FF"];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function toast(msg, err = false, ms = 5200) {
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), ms);
}
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  // failures come with an HTTP error status; a successful result may itself contain a field named "error"
  // (the PINN - solver difference of /api/solve)
  if (!r.ok) throw new Error((typeof j.error === "string" && j.error) || `HTTP ${r.status}`);
  return j;
}
async function busy(id, fn) {
  const b = $(id);
  b && b.classList.add("on");
  try { return await fn(); } catch (e) { toast(e.message, true, 9000); throw e; } finally { b && b.classList.remove("on"); }
}
function layout(extra = {}) {
  const ink = css("--ink") || "#eef", muted = css("--muted") || "#aab", line = "rgba(140,160,210,0.16)";
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, Segoe UI, system-ui, sans-serif", color: muted, size: 12 },
    margin: { l: 62, r: 18, t: 30, b: 48 },
    legend: { orientation: "h", y: 1.12, x: 0, font: { color: ink } },
    xaxis: { gridcolor: line, zerolinecolor: line, linecolor: line },
    yaxis: { gridcolor: line, zerolinecolor: line, linecolor: line },
    hoverlabel: { bgcolor: css("--panel-solid") || "#223", bordercolor: line, font: { color: ink } },
  }, extra);
}
const PCONF = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
// Plotly.js 3 accepts titles only as {text: ...}: normalise every axis / figure title given as a string
function titles(lay) {
  Object.keys(lay).forEach((k) => {
    if (/^(x|y)axis\d*$/.test(k) && lay[k] && typeof lay[k].title === "string") lay[k].title = { text: lay[k].title };
  });
  if (typeof lay.title === "string") lay.title = { text: lay.title };
  return lay;
}
const draw = (id, data, lay) => Plotly.react(id, data, titles(layout(lay)), PCONF);
function band(x, mean, sd, name, color) {
  const up = mean.map((m, i) => m + 2 * sd[i]), lo = mean.map((m, i) => m - 2 * sd[i]);
  return [
    { x, y: up, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x, y: lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: color + "33", name: name + " ±2σ", hoverinfo: "skip" },
    { x, y: mean, mode: "lines", line: { color, width: 2.4 }, name },
  ];
}
// matplotlib's inferno (perceptually uniform), as rgb() stops: Plotly.js has no built-in "Inferno"
const INFERNO = [[0, "rgb(0,0,4)"], [0.1, "rgb(22,11,57)"], [0.2, "rgb(66,10,104)"], [0.3, "rgb(106,23,110)"], [0.4, "rgb(147,38,103)"],
  [0.5, "rgb(188,55,84)"], [0.6, "rgb(221,81,58)"], [0.7, "rgb(243,120,25)"], [0.8, "rgb(252,165,10)"], [0.9, "rgb(246,215,70)"], [1, "rgb(252,255,164)"]];
function heat(z, x, y, title, opts = {}) {
  const t = { z, type: "heatmap", colorscale: opts.cs || INFERNO, colorbar: { title: { text: title, side: "right" }, thickness: 12, outlinewidth: 0 },
    zsmooth: "best", hovertemplate: "x %{x:.3g}<br>y %{y:.3g}<br>%{z:.4g}<extra></extra>" };
  if (x && y && z.length === y.length && z[0].length === x.length) { t.x = x; t.y = y; }
  if (opts.zmid !== undefined) t.zmid = opts.zmid;
  return t;
}

// ------------------------------------------------------------------------------------------------ state
const S = { info: null, case: "B_slab", units: "dimless", vals: {}, cond: {}, phys: {}, last: null, real: { src: "station", model: "cond", table: null } };
const DEFAULT_CASE = "B_slab";
const FLUIDS = { // approximate properties near 300 K (inputs, edit as needed)
  air: { nu: 1.57e-5, alpha: 2.21e-5, k: 0.026 }, water: { nu: 8.6e-7, alpha: 1.47e-7, k: 0.61 }, "engine oil": { nu: 5.5e-4, alpha: 8.6e-8, k: 0.145 },
};
const PHYS = {
  A_steady: {
    inputs: [["U", "flow speed U", "m s⁻¹", 1e-3], ["L", "length L", "m", 0.1], ["alpha", "thermal diffusivity α", "m² s⁻¹", 1e-5],
      ["TL", "temperature at x = 0", "°C", 20], ["TR", "temperature at x = L", "°C", 80]],
    map: (p) => ({ params: { Pe: p.U * p.L / p.alpha }, cond: { theta_left: 0, theta_right: 1, source: 0 },
      T: (th) => p.TL + (p.TR - p.TL) * th, X: (x) => x * p.L, groups: [["Pe = U L / α", p.U * p.L / p.alpha]], xl: "x [m]" }),
  },
  B_slab: {
    inputs: [["L", "half-thickness L", "m", 0.05], ["k", "conductivity k", "W m⁻¹ K⁻¹", 1.0], ["h", "heat-transfer coefficient h", "W m⁻² K⁻¹", 20],
      ["rho", "density ρ", "kg m⁻³", 2300], ["c", "specific heat c", "J kg⁻¹ K⁻¹", 880], ["q", "internal source q‴", "W m⁻³", 1000],
      ["T0", "initial temperature T₀", "°C", 22.5], ["Tinf", "ambient temperature T∞", "°C", 20], ["tend", "simulated time", "s", 2530],
      ["m", "side-loss parameter m (dimensionless)", "–", 1]],
    map: (p) => {
      const a = p.k / (p.rho * p.c), dT = p.q > 0 ? p.q * p.L * p.L / p.k : (p.T0 - p.Tinf || 1);
      const Bi = p.h * p.L / p.k, Fo = a * p.tend / (p.L * p.L);
      return { params: { Bi }, cond: { Q: p.q > 0 ? 1 : 0, m: p.m, theta0: (p.T0 - p.Tinf) / dT, T: Fo },
        T: (th) => p.Tinf + dT * th, X: (x) => x * p.L, time: (t) => t * p.L * p.L / a, xl: "x [m]",
        groups: [["Bi = h L / k", Bi], ["α = k / (ρ c)  [m² s⁻¹]", a], ["Fourier time α t / L²", Fo], ["ΔT scale [K]", dT], ["θ₀", (p.T0 - p.Tinf) / dT]] };
    },
  },
  D_graetz: {
    fluid: true,
    inputs: [["H", "channel height H", "m", 0.01], ["U", "mean velocity U", "m s⁻¹", 0.045], ["nu", "kinematic viscosity ν", "m² s⁻¹", 1.5e-5],
      ["alpha", "thermal diffusivity α", "m² s⁻¹", 1.5e-5], ["k", "fluid conductivity k", "W m⁻¹ K⁻¹", 0.026], ["h", "outer wall coefficient h", "W m⁻² K⁻¹", 13],
      ["Tin", "inlet temperature", "°C", 20], ["Tamb", "ambient temperature", "°C", 80]],
    map: (p) => ({ params: { Re: p.U * p.H / p.nu, Pr: p.nu / p.alpha, Bi: p.h * p.H / p.k }, cond: {},
      T: (th) => p.Tin + (p.Tamb - p.Tin) * th, X: (x) => x * p.H, hconv: (Nu) => Nu * p.k / p.H, xl: "x [m]",
      groups: [["Re = U H / ν", p.U * p.H / p.nu], ["Pr = ν / α", p.nu / p.alpha], ["Bi = h H / k", p.h * p.H / p.k]] }),
  },
  cavity: {
    fluid: true,
    inputs: [["L", "cavity side L", "m", 0.1], ["U", "lid speed U", "m s⁻¹", 0.0157], ["nu", "kinematic viscosity ν", "m² s⁻¹", 1.57e-5],
      ["alpha", "thermal diffusivity α", "m² s⁻¹", 2.21e-5], ["Tlid", "lid temperature", "°C", 60], ["Tbot", "bottom temperature", "°C", 20]],
    map: (p) => ({ params: { Re: p.U * p.L / p.nu, Pr: p.nu / p.alpha }, cond: {},
      T: (th) => p.Tbot + (p.Tlid - p.Tbot) * th, X: (x) => x * p.L, xl: "x [m]",
      groups: [["Re = U L / ν", p.U * p.L / p.nu], ["Pr = ν / α", p.nu / p.alpha]] }),
  },
};
const TAGLINE = { A_steady: "boundary layer, 1D", B_slab: "transient conduction, 1D", D_graetz: "channel flow + heat, 2D", cavity: "lid-driven, 2D" };

// ------------------------------------------------------------------------------------------------ tabs, theme
function showTab(name) {
  $$("nav.tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$("section.tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + name));
  history.replaceState(null, "", "#" + name);
  if (name === "real" && !S.real.loaded) loadStations();
  setTimeout(() => $$(`#tab-${name} .js-plotly-plot`).forEach((p) => Plotly.Plots.resize(p)), 60);
}
$$("nav.tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
$("#theme").addEventListener("click", () => {
  const r = document.documentElement;
  r.dataset.theme = r.dataset.theme === "light" ? "dark" : "light";
  try { localStorage.setItem("thermal-theme", r.dataset.theme); } catch (_) { /* private mode */ }
  if (S.last) renderSim(S.last);
});
try { const t = localStorage.getItem("thermal-theme"); if (t) document.documentElement.dataset.theme = t; } catch (_) { /* ignore */ }

// ------------------------------------------------------------------------------------------------ overview
function renderOverview() {
  const st = [
    ["1", "Verify", "Solvers reproduce manufactured solutions at their formal order."],
    ["2", "Validate", "Nusselt numbers match series, asymptotes and benchmarks; soil models meet real NOAA data."],
    ["3", "Identify", "Fisher information decides what can be estimated before any fit."],
    ["4", "Learn", "Parametric PINN ensembles answer queries, with spread and validity warnings."],
    ["5", "Decide", "Break-even queries: when a surrogate pays for its training."],
  ];
  $("#pipeline").innerHTML = st.map(([n, t, p]) => `<div class="stage"><div class="n grad-text">${n}</div><div class="t">${t}</div><p>${p}</p></div>`).join("");
  $("#findings").innerHTML = S.info.findings.map((f) => `<div class="finding"><div class="v grad-text">${f.value}</div>
    <div class="w">${f.what}</div><div class="r">${f.paper} · ${f.registry.join(", ") || f.macro}</div></div>`).join("");
  const chips = S.info.cases.map((c) => `<span class="chip">${c.title.split(" (")[0]}<b>${c.pinn ? "PINN ✓" : "solver only"}</b></span>`);
  chips.push(`<span class="chip">NOAA stations on disk<b>${S.info.stations.length}</b></span>`);
  $("#status-chips").innerHTML = chips.join("");
}

// ------------------------------------------------------------------------------------------------ simulator
const caseInfo = (k) => S.info.cases.find((c) => c.key === k);
function renderCaseCards() {
  $("#case-cards").innerHTML = S.info.cases.map((c) => `<button class="case-card ${c.key === S.case ? "on" : ""}" data-k="${c.key}">
    <b>${c.title.split(" (")[0].replace("Steady convection-diffusion", "Steady conv.–diffusion")}</b>
    <span class="tagline">${TAGLINE[c.key] || ""} · ${c.pinn ? "solver + PINN" : "solver"}</span></button>`).join("");
  $$(".case-card").forEach((b) => b.addEventListener("click", () => { selectCase(b.dataset.k); runSim(); }));
}
function selectCase(k, reset = true) {
  S.case = k;
  const c = caseInfo(k);
  if (reset || !S.vals[k]) {
    S.vals[k] = Object.fromEntries(c.params.map((p) => [p.name, p.default]));
    S.cond[k] = { ...c.conditions };
    S.phys[k] = Object.fromEntries(PHYS[k].inputs.map(([n, , , v]) => [n, v]));
  }
  renderCaseCards();
  renderControls();
  $("#sim-title").textContent = c.title;
  $("#sim-eq").innerHTML = "";
  if (window.katex) { try { katex.render(c.equation, $("#sim-eq"), { throwOnError: false }); } catch (_) { $("#sim-eq").textContent = c.equation; } }
  else $("#sim-eq").textContent = c.equation;
}
function logPos(p, v) { return p.log ? Math.log10(v) : v; }
function renderControls() {
  const c = caseInfo(S.case), phys = S.units === "phys";
  $("#param-box").style.display = phys ? "none" : "";
  $("#phys-box").style.display = phys ? "" : "none";
  // dimensionless parameters
  $("#param-box").innerHTML = c.params.map((p) => {
    const v = S.vals[S.case][p.name], lo = logPos(p, p.lo), hi = logPos(p, p.hi);
    let bar = "";
    if (p.train && c.pinn) {
      const a = (logPos(p, p.train[0]) - lo) / (hi - lo) * 100, b = (logPos(p, p.train[1]) - lo) / (hi - lo) * 100;
      bar = `<div class="train-bar" title="PINN training range"><span style="left:${a}%;width:${b - a}%"></span></div>`;
    }
    return `<div class="param"><div class="head"><b>${p.name}</b><span class="val" id="pv-${p.name}">${fmt(v)}</span></div>
      <div class="range-note">${p.help}</div>
      <input type="range" data-p="${p.name}" min="${lo}" max="${hi}" step="${(hi - lo) / 400}" value="${logPos(p, v)}">${bar}
      <div class="row"><input type="number" data-pn="${p.name}" value="${v}" step="any"><span class="faint" style="flex:2">accepted ${fmt(p.lo)}–${fmt(p.hi)}${p.train && c.pinn ? ` · PINN trained on ${fmt(p.train[0])}–${fmt(p.train[1])} (green)` : ""}</span></div></div>`;
  }).join("");
  $$("#param-box input[type=range]").forEach((r) => {
    const p = c.params.find((q) => q.name === r.dataset.p);
    r.addEventListener("input", () => {
      const v = p.log ? 10 ** Number(r.value) : Number(r.value);
      S.vals[S.case][p.name] = v; $(`#pv-${p.name}`).textContent = fmt(v); $(`[data-pn=${p.name}]`).value = Number(v.toPrecision(4));
    });
    r.addEventListener("change", () => { if ($("#sim-live").checked) runSim(); });
  });
  $$("#param-box input[type=number]").forEach((n) => n.addEventListener("change", () => {
    S.vals[S.case][n.dataset.pn] = Number(n.value); renderControls(); if ($("#sim-live").checked) runSim();
  }));
  // physical inputs
  const P = PHYS[S.case], pv = S.phys[S.case];
  let html = "";
  if (P.fluid) html += `<label class="field">fluid preset (approximate values near 300 K)</label><div class="seg" id="fluid-seg">${Object.keys(FLUIDS).map((f) => `<button data-f="${f}">${f}</button>`).join("")}</div>`;
  html += `<div class="grid-auto" style="grid-template-columns:1fr 1fr;gap:6px 10px">` + P.inputs.map(([n, lab, unit]) =>
    `<div><label class="field">${lab} [${unit}]</label><input type="number" step="any" data-ph="${n}" value="${pv[n]}"></div>`).join("") + `</div>`;
  html += `<div class="param" id="groups"></div>`;
  $("#phys-box").innerHTML = html;
  $$("#phys-box [data-ph]").forEach((i) => i.addEventListener("change", () => { pv[i.dataset.ph] = Number(i.value); physGroups(); if ($("#sim-live").checked) runSim(); }));
  $$("#fluid-seg button").forEach((b) => b.addEventListener("click", () => {
    const f = FLUIDS[b.dataset.f]; ["nu", "alpha", "k"].forEach((k) => { if (k in pv) pv[k] = f[k]; }); renderControls(); runSim();
  }));
  physGroups();
  // conditions
  $("#cond-box").innerHTML = Object.entries(S.cond[S.case]).map(([k, v]) =>
    `<label class="field">${k}</label><input type="number" step="any" data-c="${k}" value="${v}">`).join("") +
    (S.case === "A_steady" ? `<label class="field">scheme</label><select data-c="scheme">${["central", "upwind1", "upwind2", "quick"].map((s) => `<option ${S.cond[S.case].scheme === s ? "selected" : ""}>${s}</option>`).join("")}</select>` : "") +
    (S.case === "A_steady" || S.case === "B_slab" ? `<label class="field">grid cells N</label><input type="number" data-c="N" value="${S.cond[S.case].N || 40}">` : "") +
    (S.case === "B_slab" ? `<label class="field">output times n_out</label><input type="number" data-c="n_out" value="${S.cond[S.case].n_out || 6}">` : "");
  $$("#cond-box [data-c]").forEach((i) => i.addEventListener("change", () => {
    S.cond[S.case][i.dataset.c] = i.tagName === "SELECT" ? i.value : Number(i.value); if ($("#sim-live").checked) runSim();
  }));
}
function physGroups() {
  const m = PHYS[S.case].map(S.phys[S.case]);
  const c = caseInfo(S.case);
  const rows = m.groups.map(([n, v]) => {
    const p = c.params.find((q) => n.startsWith(q.name + " "));
    let tag = "";
    if (p && p.train && c.pinn) tag = (v >= p.train[0] && v <= p.train[1]) ? ` <span style="color:var(--ok)">● in PINN range</span>` : ` <span style="color:var(--stop)">● outside PINN range</span>`;
    return `<div class="row" style="justify-content:space-between"><span class="muted">${n}</span><b class="mono">${fmt(v, 4)}${tag}</b></div>`;
  });
  $("#groups").innerHTML = `<div class="head"><b>dimensionless groups</b></div>` + rows.join("");
}
$$("#units-seg button").forEach((b) => b.addEventListener("click", () => {
  $$("#units-seg button").forEach((x) => x.classList.toggle("on", x === b)); S.units = b.dataset.u; renderControls(); runSim();
}));

function simRequest() {
  const c = caseInfo(S.case);
  let params = { ...S.vals[S.case] }, opts = { ...S.cond[S.case] }, map = null;
  if (S.units === "phys") {
    map = PHYS[S.case].map(S.phys[S.case]);
    params = { ...params, ...map.params };
    opts = { ...opts, ...map.cond };
  }
  // the PINNs were trained for the default conditions only: other conditions -> solver only, said plainly
  const condKeys = Object.keys(c.conditions).filter((k) => !["N", "cells_per_unit", "n_out"].includes(k));
  const changed = condKeys.filter((k) => opts[k] !== undefined && Math.abs(Number(opts[k]) - Number(c.conditions[k])) > 1e-9);
  return { body: { case: S.case, params, opts, mode: c.pinn && !changed.length ? "compare" : "solver" }, map, changed };
}
let simSeq = 0;
async function runSim() {
  const seq = ++simSeq, req = simRequest();
  const r = await busy("#sim-busy", () => api("/api/solve", req.body));
  if (seq !== simSeq) return;
  r.__map = req.map; r.__changed = req.changed;
  S.last = r;
  renderSim(r);
}
function renderSim(r) {
  const k = S.case, c = caseInfo(k), map = r.__map, s = r.solver, pinn = r.pinn;
  const T = map ? map.T : (v) => v, X = map ? map.X : (v) => v, unit = map ? "T [°C]" : "θ", xl = map ? map.xl : "x*";
  const conv = (a) => Array.isArray(a[0]) ? a.map((row) => row.map(T)) : a.map(T);
  // chips
  const chips = [["solver", fmt(s.wall_s * 1000, 3) + " ms"]];
  if (pinn) chips.push(["PINN ensemble", fmt(r.pinn_s * 1000, 3) + " ms"], ["speed-up", fmt(r.speedup, 3) + "×"], ["PINN vs solver rel. L2", fmt(r.rel_l2, 3)]);
  if (s.err_inf !== undefined) chips.push(["solver vs exact max |err|", fmt(s.err_inf, 3)]);
  if (k === "D_graetz") {
    chips.push(["Nu solver", fmt(s.Nu, 4)]);
    if (pinn) chips.push(["Nu PINN", `${fmt(pinn.Nu, 4)} ± ${fmt(pinn.Nu_sd, 2)}`], ["Nu error", fmt(100 * r.Nu_rel_err, 3) + " %"]);
    if (map && map.hconv) chips.push(["h from Nu [W m⁻² K⁻¹]", fmt(map.hconv(s.Nu), 4)]);
    chips.push(["Newton iterations", s.newton_iterations]);
  }
  if (k === "cavity") chips.push(["Nu lid", fmt(s.Nu_lid, 4)], ["ψ min", fmt(s.psi_min, 4)], ["converged", s.converged ? "yes" : "no"], ["max |div u|", fmt(s.max_div, 2)]);
  $("#sim-chips").innerHTML = chips.map(([a, b]) => `<span class="chip">${a}<b>${b}</b></span>`).join("");
  const vb = (r.validity || []).map((v) => `<div class="badge ${v.level}">${v.text}</div>`);
  if (r.__changed && r.__changed.length && c.pinn)
    vb.push(`<div class="badge warn">Conditions ${r.__changed.join(", ")} differ from the PINN's training conditions (${c.pinn_detail}); only the verified solver is shown.</div>`);
  if (!c.pinn) vb.push(`<div class="badge warn">${c.pinn_note}</div>`);
  $("#sim-validity").innerHTML = vb.join("");
  $("#field-seg").innerHTML = "";
  $("#time-box").style.display = "none";
  if (k === "A_steady") {
    const data = [];
    if (s.exact) data.push({ x: s.x_fine.map(X), y: s.exact.map(T), mode: "lines", name: "exact", line: { color: css("--ink"), dash: "dash", width: 1.6 } });
    data.push({ x: s.x.map(X), y: s.theta.map(T), mode: "lines+markers", name: `solver (${s.scheme}, N = ${s.N})`, line: { color: ROLE.solver, width: 1.5 }, marker: { size: 4 } });
    if (pinn) data.push(...band(pinn.x.map(X), pinn.mean.map(T), pinn.sd.map((v) => map ? v * Math.abs(T(1) - T(0)) : v), `PINN ensemble (${pinn.n_members})`, ROLE.pinn));
    draw("sim-plot", data, { xaxis: { title: xl }, yaxis: { title: unit }, hovermode: "x unified" });
    $("#sim-plot2-title").textContent = pinn ? "PINN − solver" : "solver − exact";
    const e = pinn ? r.error : null;
    draw("sim-plot2", [{ x: (pinn ? pinn.x : s.x).map(X), y: e || s.theta.map((v, i) => v - (s.exact ? s.exact[Math.round(i * (s.x_fine.length - 1) / (s.x.length - 1))] : 0)), mode: "lines", line: { color: ROLE.pinn, width: 1.6 }, name: "error" }],
      { xaxis: { title: xl }, yaxis: { title: "θ difference" } });
  } else if (k === "B_slab") {
    const nt = s.t.length, box = $("#time-box"), sl = $("#time-slider");
    box.style.display = ""; sl.max = nt - 1; if (Number(sl.value) > nt - 1) sl.value = nt - 1;
    const tlab = (i) => map && map.time ? `${fmt(map.time(s.t[i]), 4)} s (t* = ${fmt(s.t[i], 3)})` : `t* = ${fmt(s.t[i], 3)}`;
    const drawT = () => {
      const i = Number(sl.value); $("#time-label").textContent = tlab(i);
      const data = [{ x: s.x.map(X), y: s.exact[i].map(T), mode: "lines", name: "exact (eigen-series)", line: { color: css("--ink"), dash: "dash", width: 1.6 } },
        { x: s.x.map(X), y: s.theta[i].map(T), mode: "markers", name: `solver (CN, N = ${s.N})`, marker: { color: ROLE.solver, size: 5 } }];
      if (pinn) data.push(...band(pinn.x.map(X), pinn.mean[i].map(T), pinn.sd[i].map((v) => map ? v * Math.abs(T(1) - T(0)) : v), `PINN ensemble (${pinn.n_members})`, ROLE.pinn));
      draw("sim-plot", data, { xaxis: { title: xl }, yaxis: { title: unit }, hovermode: "x unified" });
    };
    sl.oninput = drawT; drawT();
    $("#sim-plot2-title").textContent = "temperature field in space and time (solver)";
    draw("sim-plot2", [heat(conv(s.theta), s.x.map(X), s.t.map((t) => map && map.time ? map.time(t) : t), unit)],
      { xaxis: { title: xl }, yaxis: { title: map ? "t [s]" : "t*" } });
  } else {
    const fields = [["θ solver", () => conv(s.theta), s.x, s.y]];
    if (pinn) {
      fields.push(["θ PINN", () => conv(pinn.mean), pinn.x, pinn.y], ["PINN − solver", () => r.error, pinn.x, pinn.y, { cs: "RdBu", zmid: 0, rev: true }],
        ["PINN spread (sd)", () => pinn.sd, pinn.x, pinn.y, { cs: "Viridis" }]);
    }
    fields.push(["u", () => s.u, s.x, s.y, { cs: "Viridis" }], ["v", () => s.v, s.x, s.y, { cs: "RdBu", zmid: 0 }]);
    if (k === "cavity") {
      const n = s.psi.length, xn = [...Array(n).keys()].map((i) => i / (n - 1));
      fields.push(["ψ stream function", () => s.psi, xn, xn, { cs: "Viridis" }], ["ω vorticity", () => s.vorticity, xn, xn, { cs: "RdBu", zmid: 0 }]);
    }
    S.field = Math.min(S.field || 0, fields.length - 1);
    $("#field-seg").innerHTML = fields.map(([n], i) => `<button data-i="${i}" class="${i === S.field ? "on" : ""}">${n}</button>`).join("");
    const drawF = () => {
      const [n, zf, x, y, o] = fields[S.field];
      const isT = n.startsWith("θ");
      draw("sim-plot", [heat(zf(), x.map(X), y.map(X), isT ? unit : n, o || {})],
        { xaxis: k === "cavity" ? { title: xl, scaleanchor: "y", constrain: "domain" } : { title: xl },
          yaxis: { title: (map ? "y [m]" : "y*") + (k === "D_graetz" ? "  (stretched ×8)" : "") } });
    };
    $$("#field-seg button").forEach((b) => b.addEventListener("click", () => { S.field = Number(b.dataset.i); $$("#field-seg button").forEach((x) => x.classList.toggle("on", x === b)); drawF(); }));
    drawF();
    // second plot: profiles across the domain
    if (k === "D_graetz") {
      $("#sim-plot2-title").textContent = "temperature across the channel at three stations (solver solid, PINN dashed)";
      const data = [];
      [0.25, 0.5, 0.875].forEach((f, j) => {
        const ix = Math.round(f * (s.x.length - 1));
        data.push({ x: s.theta.map((row) => T(row[ix])), y: s.y.map(X), mode: "lines", name: `solver x = ${fmt(X(s.x[ix]), 3)}`, line: { color: SERIES[j], width: 2 } });
        if (pinn) { const ip = Math.round(f * (pinn.x.length - 1)); data.push({ x: pinn.mean.map((row) => T(row[ip])), y: pinn.y.map(X), mode: "lines", name: "PINN", line: { color: SERIES[j], dash: "dash", width: 2 }, showlegend: j === 0 }); }
      });
      draw("sim-plot2", data, { xaxis: { title: unit }, yaxis: { title: map ? "y [m]" : "y*" } });
    } else {
      $("#sim-plot2-title").textContent = "centreline velocities: u(x = ½, y) and v(x, y = ½)";
      const n = s.x.length, mid = Math.floor(n / 2);
      draw("sim-plot2", [{ x: s.y.map((_, j) => s.u[j][mid]), y: s.y, mode: "lines", name: "u(½, y)", line: { color: ROLE.solver, width: 2 } },
        { x: s.x, y: s.v[mid], mode: "lines", name: "v(x, ½)", line: { color: ROLE.pinn, width: 2 }, xaxis: "x2", yaxis: "y2" }],
        { grid: { rows: 1, columns: 2, pattern: "independent" }, xaxis: { title: "u" }, yaxis: { title: "y*" }, xaxis2: { title: "x*" }, yaxis2: { title: "v" } });
    }
  }
}
$("#sim-run").addEventListener("click", runSim);
$("#sim-default").addEventListener("click", () => { S.units = "dimless"; $$("#units-seg button").forEach((x) => x.classList.toggle("on", x.dataset.u === "dimless")); selectCase(DEFAULT_CASE); runSim(); });

// ------------------------------------------------------------------------------------------------ verify
function renderVerParam() {
  const c = caseInfo($("#ver-case").value), p = c.params[0];
  $("#ver-param").innerHTML = `<label class="field">${p.name} (${p.help})</label><input type="number" id="ver-p" value="${p.default}" step="any">`;
}
$("#ver-case").addEventListener("change", renderVerParam);
async function runVerify() {
  const key = $("#ver-case").value, c = caseInfo(key), p = c.params[0];
  const r = await busy("#ver-busy", () => api("/api/converge", { case: key, params: { [p.name]: Number($("#ver-p").value) } }));
  const data = [], chips = [];
  Object.entries(r.schemes).forEach(([sch, d], j) => {
    data.push({ x: r.h, y: d.err_inf, mode: "lines+markers", name: sch, line: { color: SERIES[j], width: 2 }, marker: { size: 7 } });
    chips.push(`<span class="chip">observed order (${sch})<b>${fmt(d.observed_order, 3)}</b></span>`);
  });
  const h0 = r.h[r.h.length - 1], e0 = Math.min(...Object.values(r.schemes).map((d) => d.err_inf[d.err_inf.length - 1]));
  [1, 2].forEach((o) => data.push({ x: r.h, y: r.h.map((h) => e0 * (h / h0) ** o), mode: "lines", name: `slope ${o}`, line: { color: css("--faint"), dash: o === 1 ? "dot" : "dash", width: 1 } }));
  $("#ver-chips").innerHTML = chips.join("");
  draw("ver-plot", data, { xaxis: { title: "grid spacing h", type: "log" }, yaxis: { title: "max |error| vs exact", type: "log", exponentformat: "power" } });
}
async function runSweep() {
  const key = $("#ver-case").value;
  if (!caseInfo(key).pinn) { toast("no PINN for this case on this machine", true); return; }
  const r = await busy("#sw-busy", () => api("/api/sweep", { case: key, n: 28 }));
  const shapes = r.train ? [{ type: "rect", xref: "x", yref: "paper", x0: r.train[0], x1: r.train[1], y0: 0, y1: 1, fillcolor: "rgba(47,211,160,0.12)", line: { width: 0 } }] : [];
  draw("sw-plot", [
    { x: r.value, y: r.member_max, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: r.value, y: r.member_min, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: ROLE.pinn + "30", name: "range of the members" },
    { x: r.value, y: r.rel_l2, mode: "lines+markers", name: "ensemble mean: rel. L2 error", line: { color: ROLE.pinn, width: 2.2 } },
    { x: r.value, y: r.rms_sd, mode: "lines", name: "ensemble spread (rms sd)", line: { color: ROLE.solver, dash: "dash", width: 2 } },
  ], { xaxis: { title: caseInfo(key).params[0].name, type: "log" }, yaxis: { title: "error / spread", type: "log", exponentformat: "power" }, shapes,
    annotations: r.train ? [{ x: Math.sqrt(r.train[0] * r.train[1]), xref: "x", y: 1, yref: "paper", text: "training range", showarrow: false, font: { color: css("--ok") } }] : [] });
}
$("#ver-run").addEventListener("click", runVerify);
$("#ver-sweep").addEventListener("click", runSweep);

// ------------------------------------------------------------------------------------------------ inverse (synthetic)
const qOf = () => (Number($("#inv-q").value) <= 0.001 ? 0 : 10 ** Number($("#inv-q").value));
function invLabels() { $("#inv-q-l").textContent = qOf() === 0 ? "0 (no source)" : fmt(qOf(), 3); $("#inv-n-l").textContent = $("#inv-n").value + " %"; }
["#inv-q", "#inv-n"].forEach((s) => $(s).addEventListener("input", invLabels));
async function runInverse() {
  const unknown = $$(".inv-u").filter((c) => c.checked).map((c) => c.value);
  const r = await busy("#inv-busy", () => api("/api/inverse", { q_vol: qOf(), noise: Number($("#inv-n").value) / 100, n_sensors: Number($("#inv-s").value), unknown, seed: Number($("#inv-seed").value) }));
  let msg = `<div class="badge ${r.identifiable ? "ok" : "stop"}">${r.message}${r.rank !== undefined ? ` · Fisher rank ${r.rank} of ${r.n_unknown}` : ""}</div>`;
  if (r.null_directions && r.null_directions.length) msg += `<div class="faint mono">null direction in (log α, log k, log h): [${r.null_directions[0].map((v) => fmt(v, 3)).join(", ")}]</div>`;
  if (r.Bi !== undefined) msg += `<div class="chips"><span class="chip">Biot number Bi = hL/k<b>${fmt(r.Bi, 4)}</b></span><span class="chip">χ² per reading<b>${fmt(r.chi2_per_reading, 3)}</b></span><span class="chip">fit time<b>${fmt(r.wall_s * 1000, 3)} ms</b></span></div>`;
  $("#inv-msg").innerHTML = msg;
  $("#inv-table").innerHTML = r.rows.length ? `<tr><th>parameter</th><th>estimate</th><th>95 % interval</th><th>truth</th><th>error</th></tr>` +
    r.rows.map((w) => `<tr><td>${w.parameter}</td><td>${fmt(w.estimate, 4)}</td><td>[${fmt(w.ci95[0], 3)}, ${fmt(w.ci95[1], 3)}]</td><td>${fmt(w.truth, 3)}</td><td>${fmt(w.error_pct, 3)} %</td></tr>`).join("") : "";
  const d = r.data, data = [];
  if (r.sensor_x) {
    r.sensor_x.forEach((xs, j) => {
      const idx = d.r.map((v, i) => [v, i]).filter(([v]) => Math.abs(v - xs) < 1e-12).map(([, i]) => i);
      const col = SERIES[j % SERIES.length];
      data.push({ x: idx.map((i) => d.t[i]), y: idx.map((i) => d.T[i]), mode: "markers", marker: { size: 4, color: col }, name: `sensor x = ${fmt(xs, 3)} m`, legendgroup: j });
      data.push({ x: r.times, y: r.fitted.map((row) => row[j]), mode: "lines", line: { color: col, width: 1.6 }, showlegend: false, legendgroup: j });
    });
  } else {
    data.push({ x: d.t, y: d.T, mode: "markers", marker: { size: 3, color: ROLE.data }, name: "sensor readings" });
  }
  draw("inv-plot", data, { xaxis: { title: "t [s]" }, yaxis: { title: "T [°C]" }, legend: { orientation: "h", y: -0.25 } });
}
$("#inv-run").addEventListener("click", runInverse);
$("#inv-default").addEventListener("click", () => { $("#inv-q").value = 5; $("#inv-n").value = 2; $("#inv-s").value = 8; $("#inv-seed").value = 0; $$(".inv-u").forEach((c) => { c.checked = true; }); invLabels(); runInverse(); });

// ------------------------------------------------------------------------------------------------ real data
async function loadStations() {
  S.real.loaded = true;
  const sel = $("#st-select");
  sel.innerHTML = S.info.stations.map((s) => `<option value="${s.station}|${s.year}" ${s.default ? "selected" : ""}>${s.station.replaceAll("_", " ")} (${s.year})</option>`).join("");
  try {
    const r = await api("/api/real/stations", {});
    S.real.cov = Object.fromEntries(r.stations.map((s) => [`${s.station}|${s.year}`, s]));
    showCoverage();
  } catch (e) { toast(e.message, true); }
}
function showCoverage() {
  const s = S.real.cov && S.real.cov[$("#st-select").value];
  if (!s) return;
  $("#st-coverage").innerHTML = "sensors: " + s.depths_m.map((z, i) => `${Math.round(z * 100)} cm <b style="color:${s.missing_fraction[i] > 0.5 ? "var(--stop)" : s.missing_fraction[i] > 0.05 ? "var(--warn)" : "var(--ok)"}">${(100 * (1 - s.missing_fraction[i])).toFixed(0)} %</b>`).join(" · ") +
    ` <span class="faint">(${fmt(s.latitude, 4)}°, ${fmt(s.longitude, 4)}°)</span>`;
}
$("#st-select").addEventListener("change", showCoverage);
$$("#src-seg button").forEach((b) => b.addEventListener("click", () => {
  $$("#src-seg button").forEach((x) => x.classList.toggle("on", x === b)); S.real.src = b.dataset.s;
  $("#src-station").style.display = S.real.src === "station" ? "" : "none"; $("#src-upload").style.display = S.real.src === "upload" ? "" : "none";
}));
$$("#model-seg button").forEach((b) => b.addEventListener("click", () => {
  $$("#model-seg button").forEach((x) => x.classList.toggle("on", x === b)); S.real.model = b.dataset.m;
  $("#model-cond").style.display = S.real.model === "cond" ? "" : "none"; $("#model-lumped").style.display = S.real.model === "lumped" ? "" : "none";
}));
$("#fetch-go").addEventListener("click", async () => {
  const station = $("#fetch-name").value.trim(), year = Number($("#fetch-year").value);
  const r = await busy("#real-busy", () => api("/api/real/fetch", { station, year }));
  toast(`${r.downloaded ? "downloaded" : "already present"}: ${station} ${year} (${fmt(r.bytes / 1e6, 3)} MB)`);
  S.info = await api("/api/info"); S.real.loaded = false; await loadStations(); $("#st-select").value = `${station}|${year}`; showCoverage();
});
// table upload and column mapping
function parseHeader(text) {
  const first = text.split(/\r?\n/).find((l) => l.trim()) || "";
  const delim = first.includes(",") ? "," : first.includes(";") ? ";" : first.includes("\t") ? "\t" : /\s+/;
  return first.split(delim).map((s) => s.trim().replace(/^"|"$/g, ""));
}
function guessPos(name) {
  const m = name.match(/(\d+(?:\.\d+)?)\s*(mm|cm|m)?/i);
  if (!m) return "";
  const v = Number(m[1]), u = (m[2] || "").toLowerCase();
  return u === "mm" ? v / 1000 : u === "cm" ? v / 100 : u === "m" ? v : v > 3 ? v / 100 : v;
}
function setTable(text, name) {
  S.real.table = text;
  const cols = parseHeader(text), tcol = cols.find((c) => /time|date|^t$|hour|sec/i.test(c)) || cols[0];
  const rows = text.split(/\r?\n/).filter((l) => l.trim()).length - 1;
  $("#colmap").innerHTML = `<p class="faint">${name}: ${rows} rows, ${cols.length} columns</p>
    <label class="field">time column</label><div class="row"><select id="tcol">${cols.map((c) => `<option ${c === tcol ? "selected" : ""}>${c}</option>`).join("")}</select>
    <select id="tunit" style="max-width:120px"><option value="auto">auto</option><option value="datetime">date-time</option><option value="s">seconds</option><option value="min">minutes</option><option value="h">hours</option><option value="d">days</option></select></div>
    <label class="field">temperature columns: use · position [m] (profile model) · lumped: one column</label>
    <div class="colmap">${cols.filter((c) => c !== tcol).map((c) => `<input type="checkbox" data-use="${c}" checked><span>${c}</span><input type="number" step="any" data-pos="${c}" value="${guessPos(c)}" placeholder="m">`).join("")}</div>
    <label class="field">ambient-temperature column (lumped model, optional)</label><select id="acol"><option value="">— fit a constant ambient —</option>${cols.map((c) => `<option>${c}</option>`).join("")}</select>`;
  const used = cols.filter((c) => c !== tcol).map(guessPos).filter((v) => v !== "").sort((a, b) => a - b);
  if (used.length >= 3) { $("#m-top").value = used[0]; $("#m-bot").value = used[used.length - 1]; $("#m-inner").value = used.slice(1, -1).join(", "); }
}
const drop = $("#drop"), file = $("#file");
drop.addEventListener("click", () => file.click());
["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", (ev) => { const f = ev.dataTransfer.files[0]; if (f) f.text().then((t) => setTable(t, f.name)); });
file.addEventListener("change", () => { const f = file.files[0]; if (f) f.text().then((t) => setTable(t, f.name)); });
$("#ex-profile").addEventListener("click", async () => {
  const r = await api("/api/real/example", { kind: "profile" }); setTable(r.csv, r.name); toast(r.note);
  S.real.model = "cond"; $$("#model-seg button").forEach((x) => x.classList.toggle("on", x.dataset.m === "cond")); $("#model-cond").style.display = ""; $("#model-lumped").style.display = "none";
  $("#up-days").value = 7; $("#m-edges").value = "";
});
$("#ex-cool").addEventListener("click", async () => {
  const r = await api("/api/real/example", { kind: "cooling" }); setTable(r.csv, r.name); toast(r.note);
  S.real.model = "lumped"; $$("#model-seg button").forEach((x) => x.classList.toggle("on", x.dataset.m === "lumped")); $("#model-cond").style.display = "none"; $("#model-lumped").style.display = "";
  $("#up-days").value = 1;
  $("#acol").value = "T_ambient_C";  // the example measures the ambient: use it rather than fitting a constant
  const amb = $('[data-use="T_ambient_C"]'); if (amb) amb.checked = false;
});
const nums = (s) => s.split(/[,;\s]+/).map(Number).filter((v) => !Number.isNaN(v) && s.trim() !== "");

async function runReal() {
  const upload = S.real.src === "upload";
  if (upload && !S.real.table) { toast("upload a table first (or load an example)", true); return; }
  if (S.real.model === "lumped") return runLumped();
  const body = {
    source: upload ? "upload" : "station", top_m: Number($("#m-top").value), bottom_m: Number($("#m-bot").value),
    inner_m: nums($("#m-inner").value), layer_edges_m: nums($("#m-edges").value), depths: $("#m-depths").checked, offsets: $("#m-offsets").checked,
    holdout: $("#m-hold").checked,
  };
  if (upload) {
    Object.assign(body, { csv: S.real.table, time_col: $("#tcol").value, time_unit: $("#tunit").value, start_offset_h: Number($("#up-start").value), days: Number($("#up-days").value),
      positions: Object.fromEntries($$("[data-use]").filter((c) => c.checked).map((c) => [c.dataset.use, Number($(`[data-pos="${c.dataset.use}"]`).value)])) });
  } else {
    const [station, year] = $("#st-select").value.split("|");
    Object.assign(body, { station, year: Number(year), start: $("#st-start").value, days: Number($("#st-days").value) });
  }
  const r = await busy("#real-busy", () => api("/api/real/fit", body));
  renderReal(r);
}
function renderReal(r) {
  const f = r.fit, chips = [["converged", f.converged ? "✓ yes" : "✗ no"], ["strategy", r.strategy_used], ["stationarity", fmt(f.stationarity, 2)],
    ["train RMSE [K]", r.rmse.map((v) => fmt(v, 3)).join(" / ")], ["residual autocorrelation", fmt(r.covariance.lag1_autocorrelation, 3)], ["time", fmt(r.wall_s, 3) + " s"]];
  if (r.holdout) chips.push(["hold-out RMSE [K]", r.holdout.rmse_model.map((v) => fmt(v, 3)).join(" / ")], ["skill vs linear-in-depth", r.holdout.skill.map((v) => fmt(v, 2)).join(" / ")]);
  $("#real-chips").innerHTML = chips.map(([a, b]) => `<span class="chip">${a}<b>${b}</b></span>`).join("");
  let msg = `<div class="badge ${f.converged ? "ok" : "stop"}">${f.converged ? "Converged" : "Not converged"} — ${f.message}${f.at_bound ? " · a parameter sits at its bound: the optimum lies outside the plausible range (model rejected, not forced)" : ""}</div>`;
  if (r.holdout_note) msg += `<div class="badge warn">${r.holdout_note}</div>`;
  if (r.meta && r.meta.source) msg += `<p class="faint">${r.meta.source}${r.meta.file ? " · " + r.meta.file : ""}${r.stamp0 ? " · window from " + r.stamp0 : ""}</p>`;
  $("#real-msg").innerHTML = msg;
  $("#real-table").innerHTML = `<tr><th>parameter</th><th>estimate</th><th>95 % Gauss–Newton</th><th>95 % HAC (autocorrelation-robust)</th></tr>` +
    r.rows.map((w) => `<tr><td>${w.parameter.replace("alpha", "α")}${w.parameter.startsWith("alpha") ? " [m² s⁻¹]" : ""}</td><td>${fmt(w.estimate, 4)}</td><td>[${fmt(w.ci95_gn[0], 3)}, ${fmt(w.ci95_gn[1], 3)}]</td><td>[${fmt(w.ci95_hac[0], 3)}, ${fmt(w.ci95_hac[1], 3)}]</td></tr>`).join("");
  $("#real-plot-title").textContent = r.holdout ? "Measured vs model (training window, then hold-out)" : "Measured vs model (training window)";
  // measured vs model, one row per inner sensor
  const s = r.series, n = s.inner.length, data = [], lay = { grid: { rows: n, columns: 1, pattern: "independent" }, hovermode: "x unified", height: Math.max(360, 230 * n) };
  s.inner.forEach((z, i) => {
    const k = s.z.findIndex((v) => Math.abs(v - z) < 1e-9), ax = i ? String(i + 1) : "";
    data.push({ x: s.t_h, y: s.T.map((row) => row[k]), mode: "lines", name: "measured", line: { color: ROLE.data, width: 2.2 }, xaxis: "x" + ax, yaxis: "y" + ax, showlegend: i === 0, legendgroup: "m" });
    data.push({ x: s.t_h, y: s.pred.map((row) => row[i]), mode: "lines", name: "model (training)", line: { color: ROLE.pinn, width: 1.5 }, xaxis: "x" + ax, yaxis: "y" + ax, showlegend: i === 0, legendgroup: "p" });
    if (r.holdout) {
      const h = r.holdout;
      data.push({ x: h.t_h, y: h.obs.map((row) => row[i]), mode: "lines", name: "measured", line: { color: ROLE.data, width: 2.2 }, xaxis: "x" + ax, yaxis: "y" + ax, showlegend: false, legendgroup: "m" });
      data.push({ x: h.t_h, y: h.pred.map((row) => row[i]), mode: "lines", name: "model (hold-out)", line: { color: ROLE.solver, width: 1.5 }, xaxis: "x" + ax, yaxis: "y" + ax, showlegend: i === 0, legendgroup: "h" });
      data.push({ x: h.t_h, y: h.baseline.map((row) => row[i]), mode: "lines", name: "linear in depth (no physics)", line: { color: ROLE.base, width: 1, dash: "dot" }, xaxis: "x" + ax, yaxis: "y" + ax, showlegend: i === 0, legendgroup: "b" });
    }
    lay["xaxis" + ax] = { title: i === n - 1 ? "hours from the start of the window" : "", gridcolor: "rgba(140,160,210,0.16)" };
    lay["yaxis" + ax] = { title: `T at ${fmt(z, 3)} m [°C]`, gridcolor: "rgba(140,160,210,0.16)" };
  });
  if (r.holdout) lay.shapes = s.inner.map((_, i) => ({ type: "rect", xref: "x" + (i ? i + 1 : ""), yref: (i ? "y" + (i + 1) : "y") + " domain", x0: r.holdout.t_h[0], x1: r.holdout.t_h[r.holdout.t_h.length - 1], y0: 0, y1: 1, fillcolor: "rgba(120,140,200,0.10)", line: { width: 0 } }));
  draw("real-plot", data, lay);
  renderLog(r.attempts);
  renderIdent(r.identifiability);
  const hp = r.harmonic && r.harmonic.pairs ? r.harmonic.pairs : [];
  $("#real-harm").innerHTML = `<p class="faint">First diurnal harmonic between neighbouring sensors (homogeneous half-space theory)</p><table class="data"><tr><th>depths [m]</th><th>α amplitude</th><th>α phase</th><th>lag [h]</th></tr>` +
    hp.map((p) => `<tr><td>${fmt(p.z1, 2)}–${fmt(p.z2, 2)}</td><td>${fmt(p.alpha_amplitude, 3)}</td><td>${fmt(p.alpha_phase, 3)}</td><td>${fmt(p.phase_lag_h, 3)}</td></tr>`).join("") + `</table>`;
}
function renderLog(attempts) {
  $("#real-log").innerHTML = attempts.map((a, i) => `<div class="step ${a.converged ? "good" : ""}"><i></i><span>${i + 1}. ${a.strategy}</span><span class="mono">stat ${fmt(a.stationarity, 2)} · cost ${fmt(a.cost, 3)}</span></div>`).join("") || "<p class='faint'>—</p>";
}
function renderIdent(id) {
  draw("real-ident", [{ x: id.singular_values.map((_, i) => `σ${i + 1}`), y: id.singular_values, type: "bar", marker: { color: id.singular_values.map((v, i) => (i < id.rank ? ROLE.solver : css("--stop"))) } }],
    { yaxis: { type: Math.max(...id.singular_values) / Math.max(Math.min(...id.singular_values), 1e-300) > 20 ? "log" : "linear",
      title: "singular value (scaled J)", exponentformat: "power", rangemode: "tozero" }, margin: { l: 60, r: 10, t: 34, b: 30 },
      title: { text: `rank ${id.rank} / ${id.n} · condition ${fmt(id.condition, 3)}`, font: { size: 13, color: css("--ink") } } });
}
async function runLumped() {
  const amb = $("#acol") ? $("#acol").value : "";
  const temp = $$("[data-use]").filter((c) => c.checked && c.dataset.use !== amb).map((c) => c.dataset.use)[0];
  const body = { csv: S.real.table, time_col: $("#tcol").value, time_unit: $("#tunit").value, temp_col: temp, ambient_col: amb, rho: Number($("#l-rho").value) || null, c: Number($("#l-c").value) || null, V_over_A: Number($("#l-va").value) || null };
  const r = await busy("#real-busy", () => api("/api/real/lumped", body));
  const f = r.fit;
  $("#real-chips").innerHTML = [["converged", f.converged ? "✓ yes" : "✗ no"], ["strategy", r.strategy_used], ["RMSE [K]", fmt(f.rmse, 3)], ...(r.h ? [["h [W m⁻² K⁻¹]", fmt(r.h, 4)]] : [])].map(([a, b]) => `<span class="chip">${a}<b>${b}</b></span>`).join("");
  $("#real-msg").innerHTML = `<div class="badge ${f.converged ? "ok" : "stop"}">${f.converged ? "Converged" : "Not converged"} — ${f.message}</div>` + (r.h_note ? `<p class="faint">${r.h_note}</p>` : "");
  $("#real-table").innerHTML = `<tr><th>parameter</th><th>estimate</th><th>95 % HAC</th></tr>` + r.rows.map((w) => `<tr><td>${w.parameter}</td><td>${fmt(w.estimate, 4)}</td><td>[${fmt(w.ci95_hac[0], 3)}, ${fmt(w.ci95_hac[1], 3)}]</td></tr>`).join("");
  $("#real-plot-title").textContent = "Measured temperature and the Newton-cooling fit";
  draw("real-plot", [{ x: r.series.t_s, y: r.series.T, mode: "markers", marker: { color: ROLE.data, size: 4 }, name: "measured" },
    { x: r.series.t_s, y: r.series.pred, mode: "lines", line: { color: ROLE.pinn, width: 2 }, name: "Newton cooling fit" }], { xaxis: { title: "t [s]" }, yaxis: { title: "T [°C]" }, height: 420 });
  renderLog(r.attempts); renderIdent(r.identifiability); $("#real-harm").innerHTML = "<p class='faint'>—</p>";
}
function realDefaults() {
  S.real.src = "station"; S.real.model = "cond";
  $$("#src-seg button").forEach((x) => x.classList.toggle("on", x.dataset.s === "station")); $("#src-station").style.display = ""; $("#src-upload").style.display = "none";
  $$("#model-seg button").forEach((x) => x.classList.toggle("on", x.dataset.m === "cond")); $("#model-cond").style.display = ""; $("#model-lumped").style.display = "none";
  const d = S.info.soil_defaults;
  $("#st-start").value = d.fit.start; $("#st-days").value = d.fit.window_days;
  $("#m-top").value = d.model.top_m; $("#m-bot").value = d.model.bottom_m; $("#m-inner").value = d.model.inner_m.join(", ");
  $("#m-edges").value = (d.compare.three_layers.layer_edges_m || []).join(", "); $("#m-depths").checked = false; $("#m-offsets").checked = false; $("#m-hold").checked = true;
  const def = S.info.stations.find((s) => s.default); if (def) $("#st-select").value = `${def.station}|${def.year}`;
  showCoverage();
}
$("#real-run").addEventListener("click", runReal);
$("#real-default").addEventListener("click", async () => { if (!S.real.loaded) await loadStations(); realDefaults(); runReal(); });

// ------------------------------------------------------------------------------------------------ learn
const GLOSS = [
  ["Verification", "Checking that the equations are solved correctly, e.g. that the error falls at the formal order of the scheme (method of manufactured solutions)."],
  ["Validation", "Checking that the equations describe reality, which needs measured data — here the NOAA soil profiles."],
  ["PINN (physics-informed neural network)", "A network trained to make the residuals of the governing equations and boundary conditions small at collocation points."],
  ["Parametric PINN", "A PINN with physical parameters (Pe, Bi, Re, Pr) as inputs: one network for a whole family of solutions."],
  ["Deep ensemble", "Several networks from different random starts; the mean is the prediction, the spread an estimate of model uncertainty."],
  ["Out-of-distribution query", "A query outside the training range; errors grow quickly there, and the ensemble spread and physics residual flag it."],
  ["Identifiability", "Whether data of a given design determine a parameter: structural (full-rank Fisher information) and practical (useful precision)."],
  ["Fisher information / Cramér–Rao bound", "F = JᵀΣ⁻¹J measures the information in the data; its inverse bounds the variance of any unbiased estimator."],
  ["Complex-step derivative", "dr/dp = Im r(p + ih)/h with h = 10⁻²⁰: exact to round-off, computed through the implicit solver itself."],
  ["Levenberg–Marquardt", "A damped Gauss–Newton method for nonlinear least squares; here in log-parameters with bounds."],
  ["Stationarity measure", "max |J_k·r| / (|J_k||r|): the cosine between the residual and each sensitivity; zero at a converged least-squares fit."],
  ["HAC covariance", "Newey–West heteroscedasticity-and-autocorrelation-consistent covariance: honest intervals when residuals are correlated in time."],
  ["Hold-out validation", "Fitting on one window and predicting the next without refitting; the fair way to compare models of different complexity."],
  ["Thermal diffusivity α", "k/(ρc) [m² s⁻¹]: how fast temperature changes spread; the soil-temperature fits estimate it."],
  ["Break-even query count", "The number of queries after which training plus querying a surrogate costs less than solving every query."],
];
function renderLearn() {
  $("#gloss").innerHTML = GLOSS.map(([t, d]) => `<dt>${t}</dt><dd>${d}</dd>`).join("");
  const eqs = S.info.cases.map((c) => `<p class="muted" style="margin:12px 0 4px">${c.title}</p><div class="eq" data-tex="${c.equation.replaceAll('"', "&quot;")}">${c.equation}</div>`);
  eqs.push(`<p class="muted" style="margin:12px 0 4px">Real soil (Real data tab)</p><div class="eq" data-tex="\\partial_t T = \\partial_z\\big(\\alpha(z)\\,\\partial_z T\\big),\\ T(z_{top},t),\\,T(z_{bot},t)\\ \\text{measured}">soil</div>`);
  $("#eqs").innerHTML = eqs.join("");
  const render = () => $$("#eqs [data-tex]").forEach((e) => { try { katex.render(e.dataset.tex, e, { throwOnError: false, displayMode: true }); } catch (_) { /* keep text */ } });
  if (window.katex) render(); else window.addEventListener("load", () => window.katex && render());
}

// ------------------------------------------------------------------------------------------------ full demo
$("#run-demo").addEventListener("click", async () => {
  const b = $("#run-demo"); b.disabled = true;
  const steps = [
    ["sim", "Simulator: default heated slab, solver vs PINN", async () => { S.units = "dimless"; $$("#units-seg button").forEach((x) => x.classList.toggle("on", x.dataset.u === "dimless")); selectCase(DEFAULT_CASE); await runSim(); }],
    ["verify", "Verify: grid convergence of the steady problem", async () => { $("#ver-case").value = "A_steady"; renderVerParam(); await runVerify(); await runSweep(); }],
    ["inverse", "Inverse lab: identify α, k, h", async () => { $("#inv-default").click(); await new Promise((r) => setTimeout(r, 1500)); }],
    ["real", "Real data: Lincoln NE soil, three-layer model", async () => { if (!S.real.loaded) await loadStations(); realDefaults(); await runReal(); }],
    ["flow", "Flow lab: transient heated lid-driven cavity", async () => { await window.LABS.flowDefault(); }],
    ["data", "Data factory: training data of the heated slab", async () => { await window.LABS.dataDefault(); }],
    ["train", "PINN training on real soil data (keeps running in its tab)", async () => { await window.LABS.trainDefault(); }],
  ];
  try {
    for (const [tab, msg, fn] of steps) { showTab(tab); toast("▶ " + msg, false, 3500); await fn(); await new Promise((r) => setTimeout(r, 900)); }
    toast("Full demo finished.");
  } catch (_) { /* error already shown */ } finally { b.disabled = false; }
});

// ------------------------------------------------------------------------------------------------ start
(async function init() {
  try {
    S.info = await api("/api/info");
  } catch (e) { toast("cannot reach the local server: " + e.message, true, 20000); return; }
  renderOverview();
  selectCase(DEFAULT_CASE);
  renderVerParam();
  invLabels();
  renderLearn();
  const tab = location.hash.replace("#", "");
  if (["overview", "sim", "verify", "inverse", "real", "learn", "flow", "data", "train"].includes(tab)) showTab(tab);
  runSim().catch(() => {});
})();
