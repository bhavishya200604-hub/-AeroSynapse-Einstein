"""Phase 10.2: screenshots of THERMAL LAB (the web demo) in headless Edge / Chrome over the DevTools protocol.

    python -m aerosynapse.thermal.web.server --no-browser &          # the server must be running (default port 8765)
    python -m aerosynapse.thermal.experiments.p10_2_walkthrough [--url http://127.0.0.1:8765] [--only 02_simulator_graetz]

Each shot runs the page's own functions (exactly what a click does), waits until no panel is busy and every plot has
rendered, and saves docs/figures/web/<name>.png; results/p10_2_walkthrough_<tag>/metrics.json records what each shot
shows and how long the page took.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from aerosynapse.thermal.experiments.p8_walkthrough import BROWSERS, CDP
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

OUT = ROOT / "docs" / "figures" / "web"
SETUP_PHYS = "S.units='phys'; $$('#units-seg button').forEach(x=>x.classList.toggle('on', x.dataset.u==='phys'));"
SETUP_DIM = "S.units='dimless'; $$('#units-seg button').forEach(x=>x.classList.toggle('on', x.dataset.u==='dimless'));"
# (name, height, JavaScript that brings the page into the state, plots that must have rendered, what it shows)
SHOTS = [
    ("01_overview", 1250, "showTab('overview');", [], "chain of trust and key findings with registry rows"),
    (
        "02_simulator_graetz",
        1250,
        f"showTab('sim'); {SETUP_PHYS} selectCase('D_graetz'); await runSim();",
        ["sim-plot", "sim-plot2"],
        "Graetz channel in physical units (°C, m, W): solver field, groups, PINN check",
    ),
    (
        "03_simulator_slab",
        1250,
        f"showTab('sim'); {SETUP_DIM} selectCase('B_slab'); await runSim();",
        ["sim-plot", "sim-plot2"],
        "heated slab: solver, exact series and the 8-member PINN ensemble (± 2 sd)",
    ),
    (
        "04_extrapolation",
        1250,
        f"showTab('sim'); {SETUP_DIM} selectCase('B_slab'); S.vals.B_slab.Bi = 30; renderControls(); await runSim();",
        ["sim-plot"],
        "Bi = 30, outside the PINN training range: STOP badge",
    ),
    (
        "05_cavity",
        1250,
        f"showTab('sim'); {SETUP_DIM} selectCase('cavity'); await runSim(); S.field = 5; renderSim(S.last);",
        ["sim-plot", "sim-plot2"],
        "heated lid-driven cavity: stream function",
    ),
    (
        "06_verify",
        1400,
        "showTab('verify'); $('#ver-case').value='A_steady'; renderVerParam(); await runVerify(); "
        "$('#ver-case').value='B_slab'; renderVerParam(); await runSweep();",
        ["ver-plot", "sw-plot"],
        "observed orders 2 and 1; PINN error and spread across Bi",
    ),
    (
        "07_inverse_no_source",
        1150,
        "showTab('inverse'); $('#inv-q').value = 0; invLabels(); await runInverse();",
        ["inv-plot"],
        "no heat source: Fisher rank 2 of 3, k and h not identifiable",
    ),
    (
        "08_real_noaa",
        1900,
        "showTab('real'); await loadStations(); realDefaults(); await runReal();",
        ["real-plot", "real-ident"],
        "NOAA Lincoln NE soil: three-layer fit, hold-out, convergence log, HAC intervals",
    ),
    (
        "09_real_upload_cooling",
        1500,
        "showTab('real'); $$('#src-seg button')[1].click(); $('#ex-cool').click(); "
        "for (let i = 0; i < 300 && !S.real.table; i++) await new Promise(r => setTimeout(r, 200)); await runReal();",
        ["real-plot"],
        "your table: a cooling curve fitted with Newton cooling (synthetic example, tau recovered)",
    ),
    ("10_learn", 1100, "showTab('learn');", [], "equations of every case and the glossary"),
    # Phase 10.4-10.6 laboratories (D149)
    (
        "11_flow_lab",
        1500,
        "showTab('flow'); await window.LABS.flowDefault(); FL.playing = false; FL.frame = FL.r.t.length - 1; drawField();"
        " FL.showP = true; $('#fl-show-p').checked = true; for (let i = 0; i < 80; i++) stepParticles();",
        ["fl-nu", "fl-psi"],
        "flow lab: transient heated lid-driven cavity, temperature with tracer particles, wall Nusselt numbers",
    ),
    (
        "12_flow_side_heated",
        1500,
        "showTab('flow'); $('#fl-heating').value = 'side_heated'; $('#fl-lid').value = 'none'; $('#fl-Ri').value = 10;"
        " $('#fl-Re').value = 100; await runFlow(); FL.playing = false; FL.frame = FL.r.t.length - 1; drawField();"
        " FL.showP = true; $('#fl-show-p').checked = true; for (let i = 0; i < 80; i++) stepParticles();",
        ["fl-nu", "fl-psi"],
        "flow lab: differentially heated cavity (natural convection, Ri = 10), heat in at the hot wall, out at the cold",
    ),
    (
        "13_data_factory",
        1400,
        "showTab('data'); await window.LABS.dataDefault();",
        ["ds-design", "ds-prev", "ds-sens"],
        "data factory: Sobol design with leakage-free splits, ground truth and noisy sensors of the heated slab",
    ),
    (
        "14_pinn_training",
        1750,
        "showTab('train'); await trainInit(); trainDefaults(); $('#tr-adam').value = 1500; $('#tr-lbfgs').value = 100;"
        " await runTrain(); for (let i = 0; i < 1500 && !['done', 'error', 'cancelled'].includes(TR.state); i++)"
        " await new Promise(r => setTimeout(r, 1000));",
        ["tr-loss", "tr-alpha", "tr-sens"],
        "PINN trained on NOAA soil temperatures (screenshot run: 1500 Adam + 100 L-BFGS steps), 10 cm held out",
    ),
]
READY = """(() => {{
  const busy = document.querySelectorAll('.busy.on').length;
  const need = {need};
  const pending = need.filter(id => !document.querySelector('#' + id + ' .main-svg')).length;
  return JSON.stringify({{ready: !busy && !pending, busy, pending}});
}})()"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--browser", default=None)
    ap.add_argument("--port", type=int, default=9334)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    exe = a.browser or next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge")
    if not exe:
        raise SystemExit("no Chromium browser found; pass --browser")
    urllib.request.urlopen(f"{a.url}/api/info", timeout=60).read()  # the server must be running
    OUT.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="p10_edge_")
    proc = subprocess.Popen(
        [
            exe,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--remote-debugging-port={a.port}",
            f"--user-data-dir={prof}",
            "--window-size=1440,1000",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shots = [s for s in SHOTS if not a.only or s[0] in a.only]
    try:
        for _ in range(60):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/json/list", timeout=2).read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except (OSError, StopIteration):
                time.sleep(0.5)
        else:
            raise SystemExit("the browser did not open its DevTools port")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "reduce"}])
        records = []
        with Experiment(f"p10_2_walkthrough_{a.tag}", {"seed": 0, "url": a.url, "browser": exe}) as ex:
            for name, height, js, need, what in shots:
                cdp.call(
                    "Emulation.setDeviceMetricsOverride",
                    width=1440,
                    height=int(height),
                    deviceScaleFactor=1,
                    mobile=False,
                )
                cdp.call("Page.navigate", url=a.url + "/")
                t0 = time.time()
                for _ in range(120):  # the page has loaded its case list
                    if cdp.js("typeof S !== 'undefined' && !!S.info"):
                        break
                    time.sleep(0.5)
                res = cdp.call(
                    "Runtime.evaluate",
                    expression=f"(async () => {{ {js} return 'ok'; }})()",
                    awaitPromise=True,
                    returnByValue=True,
                    timeout=int(a.timeout * 1000),
                )
                err = res.get("exceptionDetails")
                state = {}
                while time.time() - t0 < a.timeout:
                    state = json.loads(cdp.js(READY.format(need=json.dumps(need))) or "{}")
                    if state.get("ready"):
                        break
                    time.sleep(0.5)
                cdp.js("window.scrollTo(0, 0)")
                time.sleep(1.2)  # plotly transitions
                png = base64.b64decode(cdp.call("Page.captureScreenshot", format="png")["data"])
                (OUT / f"{name}.png").write_bytes(png)
                rec = {
                    "name": name,
                    "what": what,
                    "ready": bool(state.get("ready")),
                    "seconds": round(time.time() - t0, 1),
                    "error": (err or {}).get("text"),
                    "file": f"docs/figures/web/{name}.png",
                }
                records.append(rec)
                print(f"{name:26s} ready {rec['ready']!s:5} {rec['seconds']:6.1f} s {rec['error'] or ''}", flush=True)
            ex.metrics.update(
                {"shots": records, "all_ready": all(r["ready"] and not r["error"] for r in records), "git": git_hash()}
            )
    finally:
        proc.terminate()
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
