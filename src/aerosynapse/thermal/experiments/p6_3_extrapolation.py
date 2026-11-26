"""Step 6.3: extrapolation and validity — inside vs outside the trained range, truth-free validity indicators, and an
explicit, LABELLED study beyond the laminar regime.

(A) Parameter extrapolation: the Step 6.2 parametric members (deep ensemble + MC-dropout, no retraining) are evaluated
    from decades_below to decades_above outside their training range. Per value: error of the ensemble mean (vs the
    analytic truth), 95 % band coverage, distance from the range [decades], and truth-free indicators — ensemble spread,
    physics residual of the ensemble mean (PDE + IC + BC), MC-dropout spread, their combination. Thresholds = a
    quantile of each indicator over random IN-RANGE values; scored as detectors of 'error > tolerance'.
(B) Time extrapolation (transient problems): t up to factor x T at in-range parameter values, per time slice.
(C) Beyond the laminar regime (labelled): verified steady FV Navier–Stokes, lid-driven cavity, Re continuation on
    several grids (GCI), Ghia comparison where tabulated, linear stability of every steady solution (rightmost
    eigenvalue); Re >= label_from_Re carries 'no laminar-validity claim'.

    python experiments/p6_3_extrapolation.py [--profile smoke] [--config my.yaml] [--set key=value] [--tag name]
                                             [--part all|parametric|cavity] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from aerosynapse.thermal.experiments import p6_2_uq as P62
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import uq as UQ
from aerosynapse.thermal.pinn import validity as V
from aerosynapse.thermal.pinn.metrics import predict_grid
from aerosynapse.thermal.pinn.networks import clear_dropout_masks, sample_dropout_masks
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import validity_figs as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_3_extrapolation"


# --------------------------------------------------------------------------- members of the Step 6.2 run
def members_dir(es: dict, tag: str) -> Path:
    m = es.get("members_from", "auto")
    return ROOT / "results" / f"p6_2_uq_{tag}" if m in (None, "auto") else Path(m)


def load_members(run: Path, problem: str) -> tuple[dict, list]:
    """(6.2 spec as run, [(member record, problem object, network)]) for one problem; specs rebuilt from the run's
    saved config so the conditions are exactly those the members were trained with."""
    spec62 = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if problem not in spec62["uq_study"]["problems"]:
        return spec62, []  # not trained in that 6.2 run: skipped
    rng = spec62["uq_study"]["problems"][problem]["range"]
    out = []
    for m in met["members"]:
        if m["problem"] != problem:
            continue
        s = SP.deep_merge(spec62, P62.problem_override(problem, rng))
        s = SP.deep_merge(s, SP.deep_merge(m["over"], {"seed": int(m["seed"])}))
        prob, net = P62.load_member(s, problem, Path(m["dir"]) / "checkpoint.pt")
        out.append((m, prob, net))
    return spec62, out


# --------------------------------------------------------------------------- (A, B) parametric extrapolation
def _mc(net, prob, x, t, vals, samples: int, seed: int = 12345):
    gen, s1, s2 = torch.Generator().manual_seed(seed), 0.0, 0.0
    for _ in range(samples):
        sample_dropout_masks(net, gen)
        u = np.stack([predict_grid(net, prob, x, t, v)[0] for v in vals])
        s1, s2 = s1 + u, s2 + u * u
    clear_dropout_masks(net)
    mean = s1 / samples
    return mean, np.sqrt(np.maximum(s2 - samples * mean * mean, 0.0) / (samples - 1))


def evaluate_values(es, spec62, members, vals, t_eval, t_res, nx) -> dict:
    """Per parameter value: truth, ensemble mean/sd, errors, coverage, residual indicator, MC-dropout sd."""
    ens = [(m, p, n) for m, p, n in members if m["kind"] == "ensemble"]
    mcd = [(m, p, n) for m, p, n in members if m["kind"] == "mc_dropout"]
    prob = ens[0][1]
    x = np.linspace(0.0, 1.0, nx)
    xr = np.linspace(0.0, 1.0, int(es["residual_grid"]["nx"]))
    truth = np.stack([prob.reference(x, t_eval, v) for v in vals])
    U = np.stack([np.stack([predict_grid(n, p, x, t_eval, v)[0] for v in vals]) for _, p, n in ens])
    mu, sd = U.mean(axis=0), U.std(axis=0, ddof=1)
    out = {
        "value": list(map(float, vals)),
        "rel_l2": [],
        "member_rel_l2_min": [],
        "member_rel_l2_max": [],
        "rms_sd": [],
        "coverage_95": [],
        "residual": [],
        "residual_parts": [],
    }
    for i, v in enumerate(vals):
        y = truth[i]
        nrm = max(np.linalg.norm(y), 1e-300)
        out["rel_l2"].append(float(np.linalg.norm(mu[i] - y) / nrm))
        me = [float(np.linalg.norm(U[k, i] - y) / nrm) for k in range(len(ens))]
        out["member_rel_l2_min"].append(min(me))
        out["member_rel_l2_max"].append(max(me))
        out["rms_sd"].append(float(np.sqrt(np.mean(sd[i] ** 2))))
        out["coverage_95"].append(UQ.coverage(mu[i], sd[i], y, [0.95])[0.95])
        res = V.mean_residuals([V.residual_vectors(n, p, float(v), xr, t_res) for _, p, n in ens])
        ind = V.physics_indicator(res)
        out["residual"].append(ind["total"])
        out["residual_parts"].append(ind)
    if es.get("mc_dropout", True):
        samples = int(spec62["uq_study"]["mc_dropout"]["samples"])
        for m, p, n in mcd:
            _, s_ = _mc(n, p, x, t_eval, vals, samples)
            out[f"sd_{m['name']}"] = [float(np.sqrt(np.mean(s_[i] ** 2))) for i in range(len(vals))]
    return out


def score(es: dict, cal: dict, ev: dict, dist: np.ndarray) -> dict:
    """Thresholds from the calibration values; every indicator scored against error > tol on the evaluated values."""
    q = float(es["threshold_quantile"])
    names = ["rms_sd", "residual"] + [k for k in ev if k.startswith("sd_mcd")]
    thr = {k: float(np.quantile(cal[k], q)) for k in names}
    ind = {k: np.asarray(ev[k]) for k in names}
    cal_comb = np.sqrt(np.asarray(cal["rms_sd"]) / thr["rms_sd"] * np.asarray(cal["residual"]) / thr["residual"])
    ind["combined"] = np.sqrt(ind["rms_sd"] / thr["rms_sd"] * ind["residual"] / thr["residual"])
    thr["combined"] = float(np.quantile(cal_comb, q))
    ind["distance"], thr["distance"] = dist, 0.0
    err = np.asarray(ev["rel_l2"])
    out = {}
    for k, s in ind.items():
        out[k] = {
            "threshold": thr[k],
            "spearman": V.spearman(s, err),
            "detection": [V.detection(s, err, float(tol), thr[k]) for tol in es["error_tolerances"]],
        }
    return out


def parametric_part(es: dict, spec: dict, run: Path) -> dict:
    res = {}
    for prob_name, pc in es["problems"].items():
        spec62, members = load_members(run, prob_name)
        if not members:
            continue
        lo, hi = map(float, spec62["uq_study"]["problems"][prob_name]["range"])
        vals = np.geomspace(lo / 10 ** float(pc["decades_below"]), hi * 10 ** float(pc["decades_above"]), int(pc["n"]))
        prob = members[0][1]
        nt = int(spec62["evaluation"]["nt"])
        t_eval = (prob.T * np.arange(1, nt + 1) / nt) if prob.has_time else None
        ntr = int(es["residual_grid"]["nt"])
        t_res = np.linspace(0.0, prob.T, ntr) if prob.has_time else None
        g = np.random.default_rng(int(es["calibration"]["seed"]))
        cal_vals = np.exp(g.uniform(np.log(lo), np.log(hi), int(es["calibration"]["n"])))
        t0 = time.perf_counter()
        ev = evaluate_values(es, spec62, members, vals, t_eval, t_res, int(es["eval_nx"]))
        cal = evaluate_values(es, spec62, members, cal_vals, t_eval, t_res, int(es["eval_nx"]))
        dist = V.distance(vals, lo, hi)
        r = {
            "range": [lo, hi],
            "param_name": prob.param_name,
            "values": ev,
            "distance": dist.tolist(),
            "calibration": {k: cal[k] for k in cal if k != "residual_parts"},
            "scores": score(es, cal, ev, dist),
            "wall_s": time.perf_counter() - t0,
        }
        if prob.has_time:
            r["time"] = time_part(es, spec62, members, lo, hi)
        res[prob_name] = r
        print(f"{prob_name}: {len(vals)} values + {len(cal_vals)} calibration in {r['wall_s']:.0f} s", flush=True)
    return res


def time_part(es, spec62, members, lo, hi) -> dict:
    """Per time slice up to factor x T at in-range parameter values: slice error of the ensemble mean, slice spread,
    slice residual (PDE at that t + both boundaries); thresholds from the in-range slices (t <= T)."""
    tm = es["time"]
    ens = [(m, p, n) for m, p, n in members if m["kind"] == "ensemble"]
    prob = ens[0][1]
    T = prob.T
    ts = np.linspace(T * float(tm["factor"]) / int(tm["n"]), T * float(tm["factor"]), int(tm["n"]))
    vals = np.geomspace(lo, hi, int(tm["values"]))
    x = np.linspace(0.0, 1.0, int(es["eval_nx"]))
    xr = np.linspace(0.0, 1.0, int(es["residual_grid"]["nx"]))
    rows = []
    for v in vals:
        y = prob.reference(x, ts, float(v))
        U = np.stack([predict_grid(n, p, x, ts, float(v))[0] for _, p, n in ens])
        mu, sd = U.mean(axis=0), U.std(axis=0, ddof=1)
        res = V.mean_residuals([V.residual_vectors(n, p, float(v), xr, ts) for _, p, n in ens])
        pde = res["pde"].reshape(len(ts), -1)
        for k, tk in enumerate(ts):
            r_k = np.sqrt(np.mean(pde[k] ** 2) + res["bc_left"][k] ** 2 + res["bc_right"][k] ** 2)
            rows.append(
                {
                    "value": float(v),
                    "t": float(tk),
                    "t_over_T": float(tk / T),
                    "rel_l2": float(np.linalg.norm(mu[k] - y[k]) / max(np.linalg.norm(y[k]), 1e-300)),
                    "rms_sd": float(np.sqrt(np.mean(sd[k] ** 2))),
                    "residual": float(r_k),
                    "coverage_95": UQ.coverage(mu[k], sd[k], y[k], [0.95])[0.95],
                }
            )
    inside = [r for r in rows if r["t_over_T"] <= 1.0 + 1e-12]
    q = float(es["threshold_quantile"])
    scores = {}
    err = np.array([r["rel_l2"] for r in rows])
    for k in ("rms_sd", "residual"):
        thr = float(np.quantile([r[k] for r in inside], q))
        s = np.array([r[k] for r in rows])
        scores[k] = {
            "threshold": thr,
            "spearman": V.spearman(s, err),
            "detection": [V.detection(s, err, float(tol), thr) for tol in es["error_tolerances"]],
        }
    return {"T": T, "factor": float(tm["factor"]), "rows": rows, "scores": scores}


# --------------------------------------------------------------------------- (C) beyond the laminar regime
def cavity_task(n: int, Res: list[float], shifts: list[float], n_eig: int) -> dict:
    """One grid (own process): steady Newton continuation over Re, cavity quantities and the rightmost eigenvalue at
    every Re; a Re that cannot be reached is recorded as such (the remaining ones are skipped)."""
    from aerosynapse.thermal.experiments.p1_7_benchmarks import cavity_quantities, continuation, ghia_compare
    from aerosynapse.thermal.analysis import benchmarks as bm
    from aerosynapse.thermal.numerics import fvm2d as fv
    from aerosynapse.thermal.numerics import ns_projection as ns
    from aerosynapse.thermal.numerics.ns_stability import rightmost
    from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS

    s = ns.MACNavierStokes2D(fv.Grid2D(n, n), float(Res[0]), {"top": ns.wall(u=1.0)})
    N = SteadyNewtonNS(s)
    x = N.pack(s.u, s.v, None, None)
    log: list = []

    def setter(Re: float) -> None:
        s.Re = Re

    prev, out = 10.0, []
    for Re in Res:
        t0 = time.perf_counter()
        try:
            x, _ = continuation(N, x, setter, [float(Re)], prev, log)
        except RuntimeError as e:
            out.append({"Re": float(Re), "N": n, "converged": False, "error": str(e)})
            break
        prev = float(Re)
        N.x = x
        N.store()
        q = cavity_quantities(s)
        st = rightmost(N, x, shifts, k=n_eig)
        rec = {
            "Re": float(Re),
            "N": n,
            "converged": True,
            **{k: v for k, v in q.items() if not k.startswith("_")},
            "stability": st,
            "wall_s": time.perf_counter() - t0,
        }
        if int(Re) in bm.GHIA_U:
            rec["ghia"] = {
                k: {kk: vv for kk, vv in d.items() if kk != "dev_at_points"}
                for k, d in ghia_compare(q["_su"], q["_sv"], int(Re)).items()
            }
        out.append(rec)
    return {"N": n, "records": out, "newton_steps": len(log)}


def cavity_part(es: dict) -> dict:
    from aerosynapse.thermal.experiments.p1_7_benchmarks import gci_record
    from aerosynapse.thermal.numerics.ns_stability import crossing

    c = es["cavity"]
    jobs = [
        (
            MOD,
            "cavity_task",
            dict(
                n=int(n),
                Res=[float(r) for r in c["Re"]],
                shifts=[float(w) for w in c["shifts_imag"]],
                n_eig=int(c["n_eig"]),
            ),
        )
        for n in c["N"]
    ]
    outs = run_tasks(
        jobs,
        {"isolate": True, **es.get("execution", {})},
        lambda i, r, sec: print(
            f"cavity N {r['N']}: {sum(x['converged'] for x in r['records'])} Re solved, {sec:.0f} s", flush=True
        ),
    )
    grids = {int(o["N"]): o for o in outs}
    Ns = sorted(grids)
    per_re = {}
    for Re in c["Re"]:
        recs = [next((r for r in grids[n]["records"] if r["Re"] == float(Re) and r["converged"]), None) for n in Ns]
        ok = [r for r in recs if r]
        entry = {
            "Re": float(Re),
            "label": "no laminar-validity claim (model-equation solution only)"
            if float(Re) >= float(c["label_from_Re"])
            else "laminar steady regime (verified solver)",
            "per_grid": {
                str(r["N"]): {k: r[k] for k in ("psi_min", "u_min", "v_max", "v_min", "max_div")}
                | {
                    "growth_rate": r["stability"]["growth_rate"],
                    "frequency": r["stability"]["frequency"],
                    "unstable": r["stability"]["unstable"],
                }
                for r in ok
            },
        }
        if len(ok) >= 3:
            entry["gci_psi_min"] = gci_record([r["N"] for r in ok], [r["psi_min"] for r in ok])
        g = next((r.get("ghia") for r in reversed(ok) if r.get("ghia")), None)
        if g:
            entry["ghia_finest"] = g
        per_re[str(int(Re))] = entry
    crit = {}
    for n in Ns:
        recs = [r for r in grids[n]["records"] if r["converged"]]
        crit[str(n)] = crossing([r["Re"] for r in recs], [r["stability"]["growth_rate"] for r in recs])
    return {
        "grids": {str(n): grids[n] for n in Ns},
        "per_Re": per_re,
        "critical_Re": crit,
        "label_from_Re": float(c["label_from_Re"]),
    }


# --------------------------------------------------------------------------- driver, report, figures
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--part", default="all", choices=["all", "parametric", "cavity"])
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    cli = SP.parse_set(a.sets)
    spec = SP.load_spec(a.config or None, a.profile, cli)
    es = spec["extrapolation_study"]
    custom = bool(a.config or a.sets or a.tag != "default" or a.profile != "smoke")
    odir = ROOT / "results" / f"p6_3_extrapolation_{a.tag}"
    table = odir / "extrapolation.md" if custom else ROOT / "docs" / "tables" / "phase6_3_extrapolation.md"
    if a.figures_only:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        met["figures"] = figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        print(f"wrote {table}")
        return
    run = members_dir(es, a.tag)
    with Experiment(f"p6_3_extrapolation_{a.tag}", spec) as ex:
        ex.metrics["members_from"] = str(run)
        if a.part in ("all", "parametric"):
            if not (run / "metrics.json").exists():
                raise FileNotFoundError(f"no Step 6.2 run at {run}: run experiments/p6_2_uq.py --tag {a.tag} first")
            ex.metrics["parametric"] = parametric_part(es, spec, run)
        if a.part in ("all", "cavity") and es["cavity"].get("enabled", False):
            ex.metrics["cavity"] = cavity_part(es)
        ex.metrics["git"] = git_hash()
        ex.metrics["figures"] = figures(ex.metrics, a.tag, ex.dir, docs=not custom)
        report(ex.metrics, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def _bracket(records: list[dict]) -> tuple | None:
    """(last stable Re, first unstable Re, growth rate and frequency there) along the ladder; None if never unstable."""
    ok = [r for r in records if r["converged"]]
    for a, b in zip(ok, ok[1:], strict=False):
        if a["stability"]["growth_rate"] < 0 <= b["stability"]["growth_rate"]:
            return a["Re"], b["Re"], b["stability"]["growth_rate"], b["stability"]["frequency"]
    return None


def _f(v, fmt=".3g") -> str:
    return "—" if v is None else format(v, fmt)


def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.0f} %"


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Step 6.3 extrapolation and validity — `{tag}` (auto-generated by experiments/p6_3_extrapolation.py)",
        "",
        "Verification only (analytic truth for the 1D networks; verified FV solver for the cavity). Distance = decades "
        "outside the training range. Indicators are truth-free; each threshold is the configured quantile of the "
        "indicator over random IN-RANGE values; TPR = share of bad predictions flagged, FPR = share of good ones "
        "flagged, AUC = threshold-free ranking quality.",
        "",
    ]
    for prob, r in (met.get("parametric") or {}).items():
        v, d = r["values"], np.asarray(r["distance"])
        L += [f"## {prob}: {r['param_name']} trained on [{r['range'][0]:g}, {r['range'][1]:g}]", ""]
        L += [
            "| distance [decades] | values | ensemble rel L2 (median, max) | members max | 95 % coverage (median) |",
            "|---|---|---|---|---|",
        ]
        e, cov, mx = np.asarray(v["rel_l2"]), np.asarray(v["coverage_95"]), np.asarray(v["member_rel_l2_max"])
        for lo_, hi_, lab in ((0, 0, "0 (inside)"), (1e-9, 0.5, "0-0.5"), (0.5, 1.0, "0.5-1"), (1.0, 99, "> 1")):
            m = (d == 0) if hi_ == 0 else ((d > lo_) & (d <= hi_))
            if m.any():
                L.append(
                    f"| {lab} | {int(m.sum())} | {np.median(e[m]):.3g}, {e[m].max():.3g} | {mx[m].max():.3g} | {_pct(np.median(cov[m]))} |"
                )
        L += [
            "",
            "| indicator | threshold | Spearman vs error | "
            + " | ".join(
                f"tol {t['tol']:g}: TPR / FPR / AUC (bad n)" for t in next(iter(r["scores"].values()))["detection"]
            )
            + " |",
            "|---|---|---|" + "---|" * len(next(iter(r["scores"].values()))["detection"]),
        ]
        for k, s in r["scores"].items():
            L.append(
                f"| {k} | {s['threshold']:.3g} | {s['spearman']:.2f} | "
                + " | ".join(
                    f"{_pct(t['tpr'])} / {_pct(t['fpr'])} / {_f(t['auc'], '.2f')} ({t['n_bad']})"
                    for t in s["detection"]
                )
                + " |"
            )
        if "time" in r:
            tm = r["time"]
            rows = tm["rows"]
            L += [
                "",
                f"Time extrapolation (t up to {tm['factor']:g} T at {len({x['value'] for x in rows})} in-range values):",
                "",
            ]
            L += [
                "| t / T | ensemble rel L2 (median over values) | 95 % coverage | spread | residual |",
                "|---|---|---|---|---|",
            ]
            for lo_, hi_ in ((0, 1.0), (1.0, 2.0), (2.0, 99)):
                sel = [x for x in rows if lo_ < x["t_over_T"] <= hi_ + 1e-12]
                if sel:
                    L.append(
                        f"| ({lo_:g}, {min(hi_, tm['factor']):g}] | {np.median([x['rel_l2'] for x in sel]):.3g} | {_pct(np.median([x['coverage_95'] for x in sel]))} | {np.median([x['rms_sd'] for x in sel]):.2g} | {np.median([x['residual'] for x in sel]):.2g} |"
                    )
            for k, s in tm["scores"].items():
                L.append(
                    f"- {k}: Spearman {s['spearman']:.2f}; "
                    + "; ".join(
                        f"tol {t['tol']:g}: TPR {_pct(t['tpr'])}, FPR {_pct(t['fpr'])}, AUC {_f(t['auc'], '.2f')}"
                        for t in s["detection"]
                    )
                )
        L.append("")
    cav = met.get("cavity")
    if cav:
        Ns = sorted(cav["grids"], key=int)
        L += ["## Lid-driven cavity beyond the laminar regime (LABELLED; verified steady FV solver, uniform lid)", ""]
        L += [
            "| Re | label | psi_min per grid ("
            + " / ".join(Ns)
            + ") | GCI psi_min | growth rate per grid | Ghia max |dev| u / v (finest) |",
            "|---|---|---|---|---|---|",
        ]
        for key, e in cav["per_Re"].items():
            pg = e["per_grid"]
            psi = " / ".join(_f(pg[n]["psi_min"], ".5f") if n in pg else "not reached" for n in Ns)
            gr = " / ".join(
                _f(pg[n]["growth_rate"], ".2e")
                + (f" (ω {pg[n]['frequency']:.2f})" if pg[n]["frequency"] > 1e-8 else "")
                if n in pg
                else "—"
                for n in Ns
            )
            gc = e.get("gci_psi_min")
            gtxt = (
                f"{100 * gc['GCI_fine21']:.2g} % (p {gc['p']:.2f}, {gc['type']})"
                if gc and "GCI_fine21" in gc
                else (gc or {}).get("type", "—")
            )
            gh = e.get("ghia_finest")
            ghtxt = f"{gh['u']['max_abs_dev']:.3g} / {gh['v']['max_abs_dev']:.3g}" if gh else "not tabulated"
            L.append(f"| {key} | {e['label']} | {psi} | {gtxt} | {gr} | {ghtxt} |")
        L += [
            "",
            "Linear stability of the steady solution (rightmost eigenvalue; the Re ladder only BRACKETS the change):",
        ]
        for n in Ns:
            b = _bracket(cav["grids"][n]["records"])
            v = cav["critical_Re"].get(n)
            L.append(
                f"- N {n}: "
                + (
                    f"stable at Re {b[0]:g}, unstable at Re {b[1]:g} (rightmost mode there: growth {b[2]:.3g}, ω {b[3]:.2f}); "
                    + (
                        f"linear interpolation of the rightmost growth rates ≈ {v:.0f} (approximate: the rightmost "
                        "mode can change between ladder points)"
                        if v
                        else "no interpolated value"
                    )
                    if b
                    else "stable at every Re of the ladder"
                )
            )
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    paths = []
    if met.get("parametric"):
        paths += save_fig(F.extrapolation(met["parametric"]), f"p6_3_extrapolation_{tag}", odir, docs=docs)
        paths += save_fig(F.indicators(met["parametric"]), f"p6_3_indicators_{tag}", odir, docs=docs)
        if any("time" in r for r in met["parametric"].values()):
            paths += save_fig(F.time_extrapolation(met["parametric"]), f"p6_3_time_{tag}", odir, docs=docs)
    if met.get("cavity"):
        paths += save_fig(F.cavity(met["cavity"]), f"p6_3_cavity_{tag}", odir, docs=docs)
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
