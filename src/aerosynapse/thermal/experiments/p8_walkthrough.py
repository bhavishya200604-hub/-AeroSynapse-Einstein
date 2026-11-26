"""Phase 8 gate: walkthrough screenshots of the running app (docs/figures/app/*.png), taken by a headless Chromium
browser (Microsoft Edge or Chrome) over the DevTools protocol: open each deep link, press the page's run button where
there is one, wait until Streamlit has finished running and the plots are drawn, capture.

    python -m streamlit run src/app/main.py --server.headless true --server.address 127.0.0.1   (in another terminal)
    python -m aerosynapse.thermal.experiments.p8_walkthrough [--url http://127.0.0.1:8501] [--browser path\\to\\msedge.exe]

Writes docs/figures/app/<name>.png and results/p8_walkthrough_<tag>/metrics.json (what each shot shows, the time it
took, and whether the page reported an exception).
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

OUT = ROOT / "docs" / "figures" / "app"
# (file name, case, section, charts the page must show, button text to press or None, window height, what)
SHOTS = [
    (
        "01_overview",
        "B_slab",
        "Overview",
        0,
        None,
        1000,
        "case overview: equation, PINN source, validity of the request",
    ),
    ("02_solver_slab", "B_slab", "Solver", 1, None, 1000, "verified Crank-Nicolson solver against the exact series"),
    ("03_pinn_slab", "B_slab", "PINN", 1, None, 1000, "saved 8-member ensemble: mean, members and +- 2 sd"),
    ("04_comparison_slab", "B_slab", "Comparison", 2, None, 1250, "solver vs PINN, error, query times"),
    (
        "05_extrapolation_slab",
        "B_slab",
        "Comparison&Bi=30",
        2,
        None,
        1250,
        "Bi = 30, outside the training range: flagged",
    ),
    ("06_uncertainty_slab", "B_slab", "Uncertainty & validity", 1, None, 1100, "error and spread across Bi"),
    (
        "07_inverse",
        "B_slab",
        "Inverse",
        1,
        None,
        1150,
        "least-squares identification of alpha, k, h with 95 % intervals",
    ),
    ("08_convergence", "A_steady", "Convergence", 1, None, 1000, "grid refinement: observed orders 2 and 1"),
    ("09_training_monitor", "B_slab", "Training monitor", 1, None, 1000, "saved loss history of one ensemble member"),
    (
        "10_graetz_comparison",
        "D_graetz",
        "Comparison",
        3,
        "Run solver and networks",
        1500,
        "2D channel: solver vs PINN",
    ),
    ("11_cavity_solver", "cavity", "Solver", 2, "Run solver", 1000, "heated cavity: steady Newton solver"),
    ("12_cavity_high_re", "cavity", "Overview&Re=12000", 0, None, 1000, "Re = 12000: the guard refuses a valid answer"),
    ("13_export", "B_slab", "Export", 0, None, 900, "downloads and the equivalent CLI command"),
]
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
DONE_JS = """(() => {
  const running = document.querySelector('[data-testid="stStatusWidget"]');
  const skel = document.querySelectorAll('[data-testid="stSkeleton"]').length;
  const spin = document.querySelectorAll('[data-testid="stSpinner"]').length;
  const title = (document.body.innerText || '').includes('Physics-informed neural networks for heat transfer');
  const charts = [...document.querySelectorAll('[data-testid="stPlotlyChart"]')];
  const plots = charts.length;
  const pending = charts.filter(c => !c.querySelector('.main-svg')).length;
  return JSON.stringify({ready: !running && !skel && !spin && title && !pending, plots: plots,
                         exception: document.querySelectorAll('[data-testid="stException"]').length});
})()"""


class CDP:
    """Minimal Chrome DevTools Protocol client (one page target)."""

    def __init__(self, ws_url: str):
        from websockets.sync.client import connect

        self.ws = connect(ws_url, max_size=64 * 2**20)
        self.i = 0

    def call(self, method: str, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.i:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    def js(self, expr: str):
        return self.call("Runtime.evaluate", expression=expr, returnByValue=True)["result"].get("value")


def wait_ready(cdp: CDP, timeout: float, plots: int = 0) -> dict:
    t0, state, stable = time.time(), {}, 0
    while time.time() - t0 < timeout:
        state = json.loads(cdp.js(DONE_JS) or "{}")
        stable = stable + 1 if state.get("ready") and state.get("plots", 0) >= plots else 0
        if stable >= 3:  # ready on three consecutive polls (Streamlit reruns can start right after a render)
            return state | {"seconds": round(time.time() - t0, 1)}
        time.sleep(0.5)
    return state | {"seconds": round(time.time() - t0, 1), "timeout": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--browser", default=None)
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    exe = a.browser or next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge")
    if not exe:
        raise SystemExit("no Chromium browser found; pass --browser")
    urllib.request.urlopen(f"{a.url}/_stcore/health", timeout=10).read()  # the app must be running
    OUT.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="p8_edge_")
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
        records = []
        with Experiment(f"p8_walkthrough_{a.tag}", {"seed": 0, "url": a.url, "browser": exe}) as ex:
            for name, case, section, n_plots, button, height, what in shots:
                cdp.call(
                    "Emulation.setDeviceMetricsOverride",
                    width=1440,
                    height=int(height),
                    deviceScaleFactor=1,
                    mobile=False,
                )
                sec, _, extra = section.partition("&")  # extra: parameter values in the deep link, e.g. Bi=30
                q = f"?case={case}&section={urllib.parse.quote(sec.split()[0].lower())}" + (
                    f"&{extra}" if extra else ""
                )
                cdp.call("Page.navigate", url=a.url + "/" + q)
                time.sleep(1.0)
                st = wait_ready(cdp, a.timeout, 0 if button else n_plots)
                if button:
                    # press until the page shows its charts (a press that lands before the button's handler is
                    # attached is lost); the app keeps a request for the session, so pressing again is harmless
                    for _ in range(3):
                        cdp.js(
                            "(() => { const b = [...document.querySelectorAll('button')]"
                            f".find(b => b.innerText.includes('{button}')); if (b) b.click(); }})()"
                        )
                        time.sleep(2.0)
                        st = wait_ready(cdp, a.timeout / 3, n_plots)
                        if not st.get("timeout"):
                            break
                time.sleep(1.5)  # plotly transitions
                png = base64.b64decode(cdp.call("Page.captureScreenshot", format="png")["data"])
                (OUT / f"{name}.png").write_bytes(png)
                rec = {
                    "name": name,
                    "case": case,
                    "section": section,
                    "button": button,
                    "shows": what,
                    "file": f"docs/figures/app/{name}.png",
                    "bytes": len(png),
                    **st,
                }
                records.append(rec)
                print(
                    f"[{name}] {st.get('seconds')} s, plots {st.get('plots')}, exception {st.get('exception')}"
                    + (" TIMEOUT" if st.get("timeout") else ""),
                    flush=True,
                )
            ex.metrics["shots"] = records
            ex.metrics["failures"] = [r["name"] for r in records if r.get("timeout") or r.get("exception")]
            ex.metrics["git"] = git_hash()
    finally:
        proc.terminate()
        shutil.rmtree(prof, ignore_errors=True)
    bad = [r["name"] for r in records if r.get("timeout") or r.get("exception")]
    if bad:
        raise SystemExit(f"shots with a timeout or an exception: {bad}")


if __name__ == "__main__":
    main()
