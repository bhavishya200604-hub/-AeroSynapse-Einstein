"""Phase 7 Step 7.4: smoke tests for every figure function, on saved inputs.

    python -m aerosynapse.thermal.experiments.p7_4_smoke --capture      # record the inputs (about 1.5 h; no training, see below)
    python -m aerosynapse.thermal.experiments.p7_4_smoke                # replay every fixture, write the report

Capture: every saved-results re-render command of Step 7.3 (experiments/p7_3_rerender.py, smoke profile; the
full-profile Phase 1 reruns call the same figure functions and are skipped) and the three Phase 6B gap-run
analyses run under aerosynapse.thermal.viz.capture, which pickles the arguments of each call of a figure function into
tests/fixtures/figures/ (MANIFEST.json lists every captured call; argument files are content-addressed in blobs/,
those above LIMIT_MB in tests/fixtures/figures_large/blobs/, git-ignored). Figures are
drawn into a scratch folder; the files the commands rewrite under results/ and docs/ are restored from git
afterwards and new untracked files there removed (data/ is never touched). No network is trained.
Replay: tests/unit/test_figure_smoke.py calls each figure function alone on its fixture (figures into a temporary
folder). This script runs that test file and writes results/p7_4_smoke_<tag>/metrics.json and
docs/tables/phase7_4_smoke.md: figure functions found, functions with a fixture, replays passed / failed /
skipped (a skip names the git-ignored artefact it needs).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

FIX = ROOT / "tests" / "fixtures" / "figures"
FIX_LARGE = ROOT / "tests" / "fixtures" / "figures_large"
LIMIT_MB = 1.0
# figure functions outside the Step 7.3 catalogue commands (analyses of saved runs; no training)
EXTRA = [
    "python -m aerosynapse.thermal.experiments.p6b_g1_stats",
    "python -m aerosynapse.thermal.experiments.p6b_g2_conflict --analysis-only",
    "python -m aerosynapse.thermal.experiments.p6b_g3_budget --analysis-only",
]


def capture_commands() -> list[str]:
    from aerosynapse.thermal.experiments.p7_3_rerender import commands

    return [c["cmd"] for c in commands() if not c["full"]] + EXTRA


def _git(*args: str) -> list[str]:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    return r.stdout.split("\n")


def _snapshot() -> dict:
    """Contents of the tracked files under results/ and docs/ that already hold UNCOMMITTED work before a capture."""
    files = [x[3:] for x in _git("status", "--porcelain", "--untracked-files=no", "--", "results", "docs") if x]
    return {f: (ROOT / f).read_bytes() if (ROOT / f).exists() else None for f in files}


def _restore(snap: dict) -> list[str]:
    """Undo what the captured commands changed under results/ and docs/: a file the run modified goes back to git HEAD,
    unless it already held uncommitted work, which is written back exactly (never discarded)."""
    changed = {x[3:] for x in _git("status", "--porcelain", "--untracked-files=no", "--", "results", "docs") if x}
    out = []
    for f in sorted(changed | set(snap)):
        if f in snap:
            if snap[f] is not None and (not (ROOT / f).exists() or (ROOT / f).read_bytes() != snap[f]):
                (ROOT / f).write_bytes(snap[f])
                out.append(f)
        else:
            subprocess.run(["git", "checkout", "--", f], cwd=ROOT, check=True)
            out.append(f)
    return out


def label(cmd: str) -> str:
    """Fixture label of a command: script, then the tag / profile it runs on, then its mode."""
    words = cmd.split()[2:]
    script = words[0].split(".", 1)[1]
    opt = {k: v for k, v in zip(words, words[1:], strict=False) if k in ("--tag", "--profile")}
    mode = "+residuals" if "--residuals" in words else ""
    return f"{script}@{opt.get('--tag') or opt.get('--profile') or 'default'}{mode}"


def capture(tag: str) -> dict:
    scratch = ROOT / "results" / f"p7_4_capture_{tag}"
    raw, logs_dir = scratch / "raw", scratch / "logs"
    for d in (FIX, FIX_LARGE, raw, logs_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    before = set(_git("ls-files", "--others", "--exclude-standard", "results", "docs"))
    snap = _snapshot()
    env = {"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    import os

    env = {**os.environ, **env}
    runs = []
    for i, cmd in enumerate(capture_commands()):
        mod, *args = cmd.split()[2:]
        script = mod.split(".", 1)[1]
        t0 = time.time()
        log = logs_dir / f"{i:02d}_{label(cmd)}.log"
        with open(log, "w", encoding="utf-8") as fh:
            code = subprocess.run(
                [sys.executable, "-m", "aerosynapse.thermal.viz.capture", str(raw), label(cmd), script, *args],
                cwd=ROOT,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            ).returncode
        runs.append({"cmd": cmd, "label": label(cmd), "exit": code, "wall_s": round(time.time() - t0, 1)})
        print(f"[{i + 1}] exit {code} {time.time() - t0:7.1f} s  {cmd}", flush=True)
    # restore what the commands rewrote (their numbers equal the committed ones, Step 7.3 gate G3); uncommitted work
    # that existed before the capture is written back exactly
    changed = _restore(snap)
    new = set(_git("ls-files", "--others", "--exclude-standard", "results", "docs")) - before
    removed = []
    for f in sorted(x for x in new if x):
        if f.startswith(f"results/p7_4_capture_{tag}/"):
            continue
        (ROOT / f).unlink(missing_ok=True)
        removed.append(f)
    # manifest: fixture -> function and blob; blobs above LIMIT_MB stay out of git
    manifest, failed = {}, []
    for f in sorted(raw.glob("*.capture.json")):
        lg = json.loads(f.read_text(encoding="utf-8"))
        manifest.update(lg["fixtures"])
        failed += lg["failed"]
        shutil.move(f, logs_dir / f.name)
    for d in (FIX, FIX_LARGE):
        (d / "blobs").mkdir(parents=True, exist_ok=True)
    for b in sorted((raw / "blobs").glob("*.pkl.gz")):
        large = b.stat().st_size > LIMIT_MB * 2**20
        shutil.move(b, (FIX_LARGE if large else FIX) / "blobs" / b.name)
    for e in manifest.values():
        f = FIX / "blobs" / f"{e['blob']}.pkl.gz"
        e["committed"] = f.exists()
        e["bytes"] = (f if f.exists() else FIX_LARGE / "blobs" / f"{e['blob']}.pkl.gz").stat().st_size
    (FIX / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    shutil.rmtree(raw)
    return {
        "runs": runs,
        "restored_from_git": len(changed),
        "removed_new_files": removed,
        "fixtures": len(manifest),
        "blobs_committed": len(list((FIX / "blobs").glob("*.pkl.gz"))),
        "blobs_large_ignored": len(list((FIX_LARGE / "blobs").glob("*.pkl.gz"))),
        "pickle_failures": failed,
        "uncaptured_arguments": {k: e["uncaptured"] for k, e in manifest.items() if e["uncaptured"]},
    }


def add(cmds: list[str], tag: str) -> dict:
    """Capture further commands (e.g. the figure function of a new script) and MERGE their fixtures into the existing
    manifest, without redoing the full capture; results/ and docs/ are restored as in capture()."""
    import os

    scratch = ROOT / "results" / f"p7_4_add_{tag}"
    raw, logs_dir = scratch / "raw", scratch / "logs"
    for d in (raw, logs_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    before = set(_git("ls-files", "--others", "--exclude-standard", "results", "docs"))
    snap = _snapshot()
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    runs = []
    for i, cmd in enumerate(cmds):
        mod, *args = cmd.split()[2:]
        script = mod.split(".", 1)[1]
        t0 = time.time()
        with open(logs_dir / f"{i:02d}_{label(cmd)}.log", "w", encoding="utf-8") as fh:
            code = subprocess.run(
                [sys.executable, "-m", "aerosynapse.thermal.viz.capture", str(raw), label(cmd), script, *args],
                cwd=ROOT,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            ).returncode
        runs.append({"cmd": cmd, "label": label(cmd), "exit": code, "wall_s": round(time.time() - t0, 1)})
        print(f"[{i + 1}] exit {code} {time.time() - t0:7.1f} s  {cmd}", flush=True)
    changed = _restore(snap)
    for f in sorted(set(_git("ls-files", "--others", "--exclude-standard", "results", "docs")) - before):
        if f and not f.startswith(f"results/p7_4_add_{tag}/"):
            (ROOT / f).unlink(missing_ok=True)
    manifest = json.loads((FIX / "MANIFEST.json").read_text(encoding="utf-8"))
    new, failed = {}, []
    for f in sorted(raw.glob("*.capture.json")):
        lg = json.loads(f.read_text(encoding="utf-8"))
        new.update(lg["fixtures"])
        failed += lg["failed"]
        shutil.move(f, logs_dir / f.name)
    for b in sorted((raw / "blobs").glob("*.pkl.gz")):
        large = b.stat().st_size > LIMIT_MB * 2**20
        (FIX_LARGE if large else FIX).joinpath("blobs").mkdir(parents=True, exist_ok=True)
        shutil.move(b, (FIX_LARGE if large else FIX) / "blobs" / b.name)
    for e in new.values():
        f = FIX / "blobs" / f"{e['blob']}.pkl.gz"
        e["committed"] = f.exists()
        e["bytes"] = (f if f.exists() else FIX_LARGE / "blobs" / f"{e['blob']}.pkl.gz").stat().st_size
    manifest.update(new)
    (FIX / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    shutil.rmtree(raw)
    return {"runs": runs, "added": sorted(new), "pickle_failures": failed, "restored_from_git": len(changed)}


def replay(tag: str) -> dict:
    out = ROOT / "results" / f"p7_4_smoke_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    t0 = time.time()
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_figure_smoke.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "slow or not slow",  # every replay, the slow-marked ones included
            f"--junitxml={junit}",
            "-rs",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    wall = time.time() - t0
    (out / "pytest.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    import xml.etree.ElementTree as ET

    cases = []
    for tc in ET.parse(junit).getroot().iter("testcase"):
        status = "passed"
        msg = ""
        for child in tc:
            if child.tag in ("failure", "error", "skipped"):
                status = {"failure": "failed", "error": "failed", "skipped": "skipped"}[child.tag]
                msg = (child.get("message") or "")[:200]
        m = re.search(r"\[(.*)\]$", tc.get("name", ""))
        cases.append(
            {
                "test": tc.get("name"),
                "id": m.group(1) if m else "",
                "status": status,
                "time_s": float(tc.get("time", 0)),
                "message": msg,
            }
        )
    return {"cases": cases, "pytest_exit": r.returncode, "wall_s": round(wall, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--add", nargs="*", default=None, help="capture these commands and merge them into the manifest")
    a = ap.parse_args()
    from aerosynapse.thermal.viz.capture import figure_functions, inline_figures

    if a.add:
        with Experiment(f"p7_4_add_{a.tag}", {"seed": 0, "commands": a.add}) as ex:
            ex.metrics.update(add(a.add, a.tag) | {"git": git_hash()})
        print("added:", ex.metrics["added"], "failures:", ex.metrics["pickle_failures"])
        return
    if a.capture:
        with Experiment(f"p7_4_capture_{a.tag}", {"seed": 0, "limit_mb": LIMIT_MB}) as ex:
            ex.metrics.update(capture(a.tag) | {"git": git_hash()})
        print("capture done:", {k: len(v) if isinstance(v, list) else v for k, v in ex.metrics.items()})
        return
    funcs = [f"{m}.{n}" for m, ns in figure_functions().items() for n in ns]
    manifest = json.loads((FIX / "MANIFEST.json").read_text(encoding="utf-8"))
    fn_of = {k: f"{e['module']}.{e['function']}" for k, e in manifest.items()}
    covered = set(fn_of.values())
    blobs = {e["blob"]: (e["bytes"], e["committed"]) for e in manifest.values()}
    res = replay(a.tag)
    # record each replay's time: tests/unit/test_figure_smoke.py marks those above SLOW_S as slow
    for c in res["cases"]:
        if c["id"] in manifest:
            manifest[c["id"]]["replay_s"] = round(c["time_s"], 2)
    (FIX / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    by = {s: [c for c in res["cases"] if c["status"] == s] for s in ("passed", "failed", "skipped")}
    tests = [c for c in res["cases"] if c["id"] in fn_of]
    replayed_ok = {fn_of[c["id"]] for c in tests if c["status"] == "passed"}
    no_fixture = sorted(set(funcs) - covered)
    cap = json.loads((ROOT / f"results/p7_4_capture_{a.tag}/metrics.json").read_text(encoding="utf-8"))
    ok = not by["failed"] and not no_fixture and not inline_figures() and not cap["pickle_failures"] and bool(tests)
    with Experiment(f"p7_4_smoke_{a.tag}", {"seed": 0}) as ex:
        ex.metrics.update(
            {
                "figure_functions": len(funcs),
                "functions_with_fixture": len(set(funcs) & covered),
                "functions_without_fixture": no_fixture,
                "functions_replayed_ok": len(set(funcs) & replayed_ok),
                "inline_figure_scripts": inline_figures(),
                "fixtures": len(manifest),
                "fixtures_on_committed_blobs": sum(e["committed"] for e in manifest.values()),
                "blobs": len(blobs),
                "blobs_committed": sum(c for _, c in blobs.values()),
                "blobs_mb": round(sum(b for b, _ in blobs.values()) / 2**20, 2),
                "blobs_committed_mb": round(sum(b for b, c in blobs.values() if c) / 2**20, 2),
                "uncaptured_arguments": cap.get("uncaptured_arguments", {}),
                "passed": len(by["passed"]),
                "failed": [c["test"] + ": " + c["message"] for c in by["failed"]],
                "skipped": [c["test"] + ": " + c["message"] for c in by["skipped"]],
                "replay_wall_s": res["wall_s"],
                "slowest": sorted(((c["time_s"], c["id"]) for c in tests), reverse=True)[:5],
                "capture_commands": len(cap["runs"]),
                "capture_failed_commands": [r["cmd"] for r in cap["runs"] if r["exit"] != 0],
                "pickle_failures": cap["pickle_failures"],
                "pass": ok,
                "git": git_hash(),
            }
        )
        m = ex.metrics
        L = [
            "# Phase 7 Step 7.4 — smoke tests for every figure function (auto-generated by experiments/p7_4_smoke.py)",
            "",
            f"git {git_hash()[:7]}; overall **{'PASS' if ok else 'FAIL'}**. Inputs captured from the saved-results "
            "re-render commands (no training); each function is then called alone on its fixture "
            "(tests/unit/test_figure_smoke.py).",
            "",
            "| quantity | value |",
            "|---|---|",
            f"| figure functions (src/viz public plotting functions + script figure functions) | {m['figure_functions']} |",
            f"| with a captured fixture | {m['functions_with_fixture']} |",
            f"| replayed without error | {m['functions_replayed_ok']} |",
            f"| scripts drawing inside main() | {len(m['inline_figure_scripts'])} |",
            f"| fixtures (calls captured); on committed inputs | {m['fixtures']}; {m['fixtures_on_committed_blobs']} |",
            f"| distinct input files: all / committed (≤ {LIMIT_MB:g} MB; larger ones git-ignored) | {m['blobs']} "
            f"({m['blobs_mb']} MB) / {m['blobs_committed']} ({m['blobs_committed_mb']} MB) |",
            f"| replays passed / failed / skipped | {m['passed']} / {len(m['failed'])} / {len(m['skipped'])} |",
            f"| replay wall time | {m['replay_wall_s']} s |",
            f"| capture commands (failed) | {m['capture_commands']} ({len(m['capture_failed_commands'])}) |",
            f"| captures that failed / arguments stored as placeholders | {len(m['pickle_failures'])} / "
            f"{len(m['uncaptured_arguments'])} |",
        ]
        for title, key in (
            ("Functions without a fixture", "functions_without_fixture"),
            ("Failed", "failed"),
            ("Skipped", "skipped"),
            ("Capture commands that failed", "capture_failed_commands"),
            ("Pickle failures", "pickle_failures"),
            (
                "Arguments stored as placeholders (unused by the figure code: the replay would fail otherwise)",
                "uncaptured_arguments",
            ),
        ):
            if m[key]:
                items = m[key].items() if isinstance(m[key], dict) else m[key]
                L += ["", f"## {title}", ""] + [f"- {x}" for x in items]
        (ROOT / "docs/tables/phase7_4_smoke.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print({k: (len(v) if isinstance(v, list) else v) for k, v in ex.metrics.items() if k != "git"})


if __name__ == "__main__":
    main()
