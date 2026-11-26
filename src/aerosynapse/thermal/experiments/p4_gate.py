"""Phase 4 gate: (G1) 2D PINN results vs the Phase 1 solver on the interpolation-test parameters (compiled from the
saved Phase 4 study runs), (G2) the ablation table (Step 4.7), (G3) benchmark comparison with new physics-only PINN
runs: Ghia et al. cavity centreline profiles (Re 100, uniform lid), the Graetz constant-wall-temperature Nusselt
number (parallel plates, with axial conduction) and the de Vahl Davis natural-convection Nusselt number (Ra 1e4).
C_D: no literature value exists for the square cylinder of D_obstacle (D69) and the Schäfer–Turek circle is not a
supported PINN geometry, so C_D is compared with the verified solver only (G1); St: not applicable (steady problems).

Everything is set in `gate` (configs/pinn/defaults.yaml). Benchmark PINNs use no data (comparing a network trained
on solver data with the solver would be circular).

    python experiments/p4_gate.py --profile smoke [--config my.yaml] [--set key=value] [--tag name]
    python experiments/p4_gate.py --compile-only        # G1 + G2 from saved runs, no training
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from aerosynapse.thermal.experiments.p4_report import NOT_CAPTURED, _fmt, ablation_aggregate
from aerosynapse.thermal.analysis import benchmarks as bm
from aerosynapse.thermal.physics import graetz as gz
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.networks import resolve_device
from aerosynapse.thermal.pinn.ref2d import cavity_reference
from aerosynapse.thermal.pinn.trainer import Trainer
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import apply_style, save_fig

MOD = "aerosynapse.thermal.experiments.p4_gate"
SEED_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]


# --- G1: compile the interpolation-test results of the saved study runs ----------------------------------------------
def compile_rows(sources: list[dict]) -> list[dict]:
    rows = []
    for src in sources:
        rdir = ROOT / "results" / src["run"]
        if not (rdir / "metrics.json").exists():
            rows.append({"label": src["label"], "run": src["run"], "missing": True})
            continue
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        recs = [
            r
            for r in met["records"]
            if r.get("split") == "interp"
            and all(r.get(k) == v for k, v in (src.get("where") or {}).items())
            and (not src.get("values") or r.get(src["group"]) in src["values"])
        ]
        for g in dict.fromkeys(r[src["group"]] for r in recs):
            rs = [r for r in recs if r[src["group"]] == g]
            e = [r["rel_l2_max"] for r in rs]
            row = {
                "label": src["label"],
                "run": src["run"],
                "git": met.get("git", "?")[:7],
                "group": g,
                "n": len(rs),
                "seeds": len({r["seed"] for r in rs}),
                "err_mean": float(np.mean(e)),
                "err_min": min(e),
                "err_max": max(e),
                "ref_unc": max(r.get("ref_unc_max", float("nan")) for r in rs),
            }
            for k in ("nu_top_rel_err", "q_bottom_rel_err", "C_D_rel_err"):
                v = [r[k] for r in rs if r.get(k) is not None]
                if v:
                    row[k] = (min(v), max(v))
            rows.append(row)
    return rows


# --- G3: benchmark problems --------------------------------------------------------------------------------------------
def bench_spec(spec: dict, b: dict, seed: int) -> dict:
    s = SP.deep_merge(spec, spec["gate"].get("common") or {})
    s = SP.deep_merge(s, b.get("overrides") or {})
    s["seed"] = seed
    return s


def cavity_setup(b: dict, ref_cfg: dict) -> tuple[dict, dict, dict]:
    """(parameters, conditions, problem config) of a cavity benchmark."""
    if b["kind"] == "ghia_cavity":
        par = {"Re": float(b["Re"]), "Pr": float(b["Pr"]), "Ri": 0.0, "Bi": 1.0}
        cond = {
            "velocity": {"top": {"profile": "uniform", "amplitude": 1.0}},
            "thermal": {w: {"type": "dirichlet", "value": 1.0 if w == "top" else 0.0} for w in P2.WALLS},
        }
    else:  # dvd: free-fall scaling U = sqrt(g beta dT L): Re = sqrt(Ra/Pr), Pe = sqrt(Ra Pr), Ri = 1
        Ra, Pr = float(b["Ra"]), float(b["Pr"])
        par = {"Re": float(np.sqrt(Ra / Pr)), "Pr": Pr, "Ri": 1.0, "Bi": 1.0}
        cond = {
            "velocity": {},
            "thermal": {
                "left": {"type": "dirichlet", "value": 1.0},
                "right": {"type": "dirichlet", "value": 0.0},
                "bottom": {"type": "neumann", "value": 0.0},
                "top": {"type": "neumann", "value": 0.0},
            },
        }
    return par, cond, {"parameters": par, "conditions": cond, "reference": ref_cfg}


def graetz_setup(spec: dict, b: dict) -> tuple[dict, dict]:
    pcfg = SP.deep_merge(
        spec["problems2d"]["D_graetz"],
        {"thermal": {w: {"type": "dirichlet", "value": float(b["theta_wall"])} for w in ("bottom", "top")}},
    )
    return {"Re": float(b["Re"]), "Pr": float(b["Pr"]), "Bi": 1.0, "Ri": 0.0}, pcfg


def ref_nu_hot(ref: dict, theta_wall: float = 1.0) -> float:
    """Mean hot-wall (x = 0) Nusselt number of a cell-centred field, 2nd-order one-sided gradient (bc_order 2, D30)."""
    th, h = np.asarray(ref["theta"]), float(ref["x"][1] - ref["x"][0])
    g = (-8.0 * theta_wall + 9.0 * th[:, 0] - th[:, 1]) / (3.0 * h)
    return float(-g.mean())


def pinn_nu_hot(prob, net, n: int = 128) -> float:
    p0 = next(net.parameters())
    y = (np.arange(n) + 0.5) / n
    X = torch.as_tensor(np.stack([np.zeros(n), y], 1), dtype=p0.dtype, device=p0.device)
    return float(-prob.fields(net, X, order=1)["theta_x"].detach().mean())


def ref_nu_graetz(ref: dict, theta_wall: float, x_lo: float, x_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Local Nu_Dh(x) = 2 |dtheta/dy|_wall / |theta_wall - theta_bulk| (Dh = 2H, H = 1; Phase 1 convention) at the
    reference columns inside [x_lo, x_hi]; bottom-wall gradient 2nd-order one-sided."""
    x, y = np.asarray(ref["x"]), np.asarray(ref["y"])
    u, th = np.asarray(ref["u"]), np.asarray(ref["theta"])
    h = float(y[1] - y[0])
    sel = (x >= x_lo) & (x <= x_hi)
    g = (-8.0 * theta_wall + 9.0 * th[0] - th[1]) / (3.0 * h)
    tb = (u * th).sum(0) / u.sum(0)
    return x[sel], (2.0 * np.abs(g) / np.abs(theta_wall - tb))[sel]


def pinn_nu_graetz(prob, net, xs: np.ndarray, theta_wall: float, n: int = 64) -> np.ndarray:
    p0 = next(net.parameters())
    y = (np.arange(n) + 0.5) / n * prob.Ly
    out = []
    for xv in xs:
        F = prob.fields(net, prob.inputs(np.stack([np.full(n, xv), y], 1), None, p0), order=0)
        u, th = F["u"].detach().cpu().numpy(), F["theta"].detach().cpu().numpy()
        tb = float((u * th).sum() / u.sum())
        g = float(prob.fields(net, prob.inputs(np.array([[xv, 0.0]]), None, p0), order=1)["theta_y"].detach()[0])
        out.append(2.0 * abs(g) / abs(theta_wall - tb))
    return np.asarray(out)


def centrelines(prob, net, n: int = 129) -> dict[str, np.ndarray]:
    """u(x = 0.5, y) and v(x, y = 0.5) of a cavity network."""
    p0 = next(net.parameters())
    s = np.linspace(0.0, 1.0, n)
    mk = lambda a: torch.as_tensor(a, dtype=p0.dtype, device=p0.device)  # noqa: E731
    u = prob.fields(net, mk(np.stack([np.full(n, 0.5), s], 1)), order=0)["u"].detach().cpu().numpy()
    v = prob.fields(net, mk(np.stack([s, np.full(n, 0.5)], 1)), order=0)["v"].detach().cpu().numpy()
    return {"s": s, "u": u, "v": v}


def ghia_deviation(s: np.ndarray, prof: np.ndarray, Re: int, which: str) -> dict:
    """Max and RMS |profile - Ghia| at the included interior Ghia points (wall points are boundary values)."""
    c, val, mask = bm.ghia_points(Re, which)
    inner = mask & (c > 0) & (c < 1)
    d = np.interp(c[inner], s, prof) - val[inner]
    return {"max": float(np.abs(d).max()), "rms": float(np.sqrt(np.mean(d**2))), "n": int(inner.sum())}


def ref_centrelines(ref: dict) -> dict[str, np.ndarray]:
    """Centrelines of a cell-centred reference (even n: average of the two middle columns / rows)."""
    u, v, x = np.asarray(ref["u"]), np.asarray(ref["v"]), np.asarray(ref["x"])
    m = len(x) // 2
    return {"s": x, "u": 0.5 * (u[:, m - 1] + u[:, m]), "v": 0.5 * (v[m - 1, :] + v[m, :])}


def _net(s: dict, prob) -> torch.nn.Module:
    dev = resolve_device(s["device"])
    torch.manual_seed(s["seed"])
    return build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(
        device=dev, dtype=torch.float64 if s["dtype"] == "float64" else torch.float32
    )


def run_benchmark(spec: dict, name: str, seed: int, ref: dict, out_dir: Path, tag: str) -> dict:
    b = spec["gate"]["benchmarks"][name]
    s = bench_spec(spec, b, seed)
    form = SP.deep_merge(s["formulation"], b.get("formulation") or {})
    if b["kind"] in ("ghia_cavity", "dvd"):
        _, _, pcfg = cavity_setup(b, spec["gate"]["reference"])
        prob = P2.build_cavity(pcfg, form, ref)
    else:
        vals, pcfg = graetz_setup(spec, b)
        prob = CH.build_channel("D_graetz", pcfg, form, [], vals, 1.0)
    net = _net(s, prob)
    tr = Trainer(net, prob, s, out_dir, tag=tag)
    prob.fit_normalisation(net, tr.batch(0))
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    out = {"benchmark": name, "kind": b["kind"], "seed": seed, "status": res["status"], "train_wall_s": wall}
    if b["kind"] == "ghia_cavity":
        out.update(P2.evaluate2d(prob, net, full=False))
        cl = centrelines(prob, net)
        out["centrelines"] = {k: v.tolist() for k, v in cl.items()}
        for w in ("u", "v"):
            out[f"ghia_{w}"] = ghia_deviation(cl["s"], cl[w], int(b["Re"]), w)
    elif b["kind"] == "dvd":
        out.update(P2.evaluate2d(prob, net, full=False))
        out["Nu_hot"] = pinn_nu_hot(prob, net)
    else:
        x_lo, x_hi = b["window"]
        xs, nu_ref = ref_nu_graetz(ref, float(b["theta_wall"]), x_lo, x_hi)
        nu = pinn_nu_graetz(prob, net, xs, float(b["theta_wall"]))
        out.update(CH.evaluate_channel(prob, net, ref, None))
        out.update(
            {
                "x": xs.tolist(),
                "Nu_x": nu.tolist(),
                "Nu_mean": float(nu.mean()),
                "Nu_x_rel_dev_max": float(np.max(np.abs(nu - nu_ref) / nu_ref)),
            }
        )
    return out


def bench_reference(spec: dict, name: str) -> dict:
    """Reference solver solution (fine grid + coarse-fine differences) and its benchmark quantities."""
    b = spec["gate"]["benchmarks"][name]
    t0 = time.perf_counter()
    if b["kind"] in ("ghia_cavity", "dvd"):
        par, cond, _ = cavity_setup(b, spec["gate"]["reference"])
        rc = spec["gate"]["reference"]
        ref = cavity_reference(par, cond, int(rc["grid"]), int(rc["refine"]), float(rc["tol"]))
        q = {}
        if b["kind"] == "ghia_cavity":
            cl = ref_centrelines(ref)
            q = {f"ghia_{w}": ghia_deviation(cl["s"], cl[w], int(b["Re"]), w) for w in ("u", "v")}
            ref["centrelines"] = {k: v.tolist() for k, v in cl.items()}
        else:
            q = {"Nu_hot": ref_nu_hot(ref)}
    else:
        vals, pcfg = graetz_setup(spec, b)
        rc = pcfg["reference"]
        ref = CH.channel_reference(
            vals, pcfg, int(rc["cells_per_unit"]), int(rc["refine"]), float(rc["tol"]), rc["scheme"]
        )
        xs, nu = ref_nu_graetz(ref, float(b["theta_wall"]), *b["window"])
        q = {"x": xs.tolist(), "Nu_x": nu.tolist(), "Nu_mean": float(nu.mean())}
    ref["quantities"] = q
    ref["wall_s"] = time.perf_counter() - t0
    return ref


def literature(b: dict) -> dict:
    if b["kind"] == "ghia_cavity":
        return {"source": "Ghia et al. 1982 (VERIFIED-SECONDARY, LITERATURE.md)", "Re": int(b["Re"])}
    if b["kind"] == "dvd":
        Ra = float(b["Ra"])
        return {
            "source": "de Vahl Davis 1983 via Wasberg 2003 (VERIFIED-SECONDARY); Hortmann 1990 same table",
            "Nu": bm.DE_VAHL_DAVIS_NU.get(Ra),
            "Nu_hortmann": bm.HORTMANN_NU.get(Ra),
        }
    Pe = float(b["Re"]) * float(b["Pr"])
    return {
        "source": "Graetz CWT fully developed Nu_Dh: Chebyshev eigenvalue problem (DERIVED, Step 1.7); 7.54 without axial conduction (Shah & London, UNVERIFIED)",
        "Nu_axial": gz.nu_cwt_spectral(64, Pe)["Nu"],
        "Nu_no_axial": gz.nu_cwt_spectral(64)["Nu"],
        "Pe": Pe,
    }


# --- report ------------------------------------------------------------------------------------------------------------
def _rng(t) -> str:
    return "—" if t is None else f"{t[0]:.2g}–{t[1]:.2g}"


def gate_report(
    rows: list[dict], ablation: dict | None, bench: dict | None, spec: dict, path: Path, tag: str
) -> list[str]:
    tgt = spec["evaluation"]["target_rel_l2"]
    L = [
        f"# Phase 4 gate — `{tag}` (auto-generated by experiments/p4_gate.py)",
        "",
        "## G1 — 2D PINN results vs the Phase 1 solver on the interpolation-test parameters",
        "",
        "Compiled from the saved study runs (records with split = interp; seeds pooled). Error = worst-field rel L2 vs the fine "
        "reference; ref unc = largest reference coarse-fine difference of those points; Nu/q/C_D = relative errors (min–max).",
        "",
        "| study (run, commit) | model | records (seeds) | error mean (min–max) | ref unc | Nu_top err | q err | C_D err | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            L.append(f"| {r['label']} ({r['run']}) | — | — | not found | — | — | — | — | missing |")
            continue
        st = (
            "target met"
            if r["err_max"] < tgt
            else ("not captured" if r["err_mean"] >= NOT_CAPTURED else "below target")
        )
        L.append(
            f"| {r['label']} ({r['run']}, {r['git']}) | {r['group']} | {r['n']} ({r['seeds']}) | {r['err_mean']:.3g} ({r['err_min']:.3g}–{r['err_max']:.3g}) | "
            f"{_fmt(r['ref_unc'], '.1e')} | {_rng(r.get('nu_top_rel_err'))} | {_rng(r.get('q_bottom_rel_err'))} | {_rng(r.get('C_D_rel_err'))} | {st} |"
        )
    L += ["", f"Target rel L2 < {tgt:g} (status 'target met' = every record below it).", ""]
    L += ["## G2 — ablation (Step 4.7)", ""]
    if ablation:
        L += [
            f"Full table: docs/tables/phase4_7_ablation.md (run {ablation['run']}, {ablation['git']}). Key rows (interp error, mean ± std over seeds; training s):",
            "",
        ]
        L += ["| configuration | error | extrap error | training s |", "|---|---|---|---|"]
        for n, g in ablation["agg"].items():
            L.append(
                f"| {n} | {g['err_mean']:.3g} ± {g['err_std']:.2g} | {g['ext_mean']:.3g} ± {g['ext_std']:.2g} | {g['time_mean']:.0f} |"
            )
    else:
        L.append("Ablation run not found.")
    L += ["", "## G3 — benchmark comparison (physics-only PINNs, no data)", ""]
    if not bench:
        L.append("Benchmarks not run (--compile-only).")
    else:
        L.append(
            f"Network {spec['network']['depth']}x{spec['network']['width']}, Adam {spec['training']['adam']['epochs']} + L-BFGS "
            f"{spec['training']['lbfgs']['iterations']}, common {spec['gate'].get('common') or 'none'}; per benchmark: formulation and overrides below; "
            f"seeds {spec['gate']['seeds']}; reference solver {spec['gate']['reference']['grid']}²/{spec['gate']['reference']['grid'] * spec['gate']['reference']['refine']}² "
            "(cavities) or problems2d.D_graetz.reference (channel)."
        )
        L.append("")
        for name, B in bench.items():
            b, lit, rq, rs = B["cfg"], B["literature"], B["ref"], B["runs"]
            L += [
                f"### {name} ({b['kind']}): formulation {b.get('formulation')}, overrides {b.get('overrides') or 'none'}",
                "",
                f"Literature: {lit['source']}.",
                "",
            ]
            if b["kind"] == "ghia_cavity":
                L += [
                    "| | u(y) at x = 0.5: max / RMS dev from Ghia | v(x) at y = 0.5: max / RMS dev | field worst rel L2 vs solver | training s |",
                    "|---|---|---|---|---|",
                ]
                L.append(
                    f"| solver ({rq['n_fine']}²) | {rq['q']['ghia_u']['max']:.3g} / {rq['q']['ghia_u']['rms']:.3g} | {rq['q']['ghia_v']['max']:.3g} / {rq['q']['ghia_v']['rms']:.3g} | — | — |"
                )
                for r in rs:
                    L.append(
                        f"| PINN seed {r['seed']} | {r['ghia_u']['max']:.3g} / {r['ghia_u']['rms']:.3g} | {r['ghia_v']['max']:.3g} / {r['ghia_v']['rms']:.3g} | {r['rel_l2_max']:.3g} | {r['train_wall_s']:.0f} |"
                    )
                mu = np.mean([r["ghia_u"]["rms"] for r in rs])
                mv = np.mean([r["ghia_v"]["rms"] for r in rs])
                L.append("")
                L.append(
                    f"Mean PINN RMS deviation from Ghia: u {mu:.3g}, v {mv:.3g} ({rs[0]['ghia_u']['n']} / {rs[0]['ghia_v']['n']} interior points; "
                    f"{mu / max(rq['q']['ghia_u']['rms'], 1e-300):.1f}x / {mv / max(rq['q']['ghia_v']['rms'], 1e-300):.1f}x the solver's)."
                )
            elif b["kind"] == "dvd":
                L += [
                    "| | mean hot-wall Nu | rel dev from de Vahl Davis | field worst rel L2 vs solver | training s |",
                    "|---|---|---|---|---|",
                ]
                L.append(f"| literature | {lit['Nu']} (Hortmann {lit['Nu_hortmann']}) | — | — | — |")
                L.append(
                    f"| solver ({rq['n_fine']}²) | {rq['q']['Nu_hot']:.4g} | {abs(rq['q']['Nu_hot'] - lit['Nu']) / lit['Nu']:.2e} | — | — |"
                )
                for r in rs:
                    L.append(
                        f"| PINN seed {r['seed']} | {r['Nu_hot']:.4g} | {abs(r['Nu_hot'] - lit['Nu']) / lit['Nu']:.2e} | {r['rel_l2_max']:.3g} | {r['train_wall_s']:.0f} |"
                    )
            else:
                L += [
                    f"Pe = {lit['Pe']:g}; fully developed Nu_Dh (eigenvalue): {lit['Nu_axial']:.4f} with axial conduction, {lit['Nu_no_axial']:.4f} without; "
                    f"local Nu averaged over x in {b['window']}.",
                    "",
                    "| | mean Nu_Dh over the window | rel dev from eigenvalue Nu | max local dev from solver | field worst rel L2 vs solver | training s |",
                    "|---|---|---|---|---|---|",
                ]
                L.append(
                    f"| solver | {rq['q']['Nu_mean']:.4g} | {abs(rq['q']['Nu_mean'] - lit['Nu_axial']) / lit['Nu_axial']:.2e} | — | — | — |"
                )
                for r in rs:
                    L.append(
                        f"| PINN seed {r['seed']} | {r['Nu_mean']:.4g} | {abs(r['Nu_mean'] - lit['Nu_axial']) / lit['Nu_axial']:.2e} | {r['Nu_x_rel_dev_max']:.2e} | {r['rel_l2_max']:.3g} | {r['train_wall_s']:.0f} |"
                    )
            L.append("")
        L += [
            "C_D: no literature value for the square cylinder of D_obstacle (D69); the Schäfer–Turek circle is not a supported PINN "
            "geometry: C_D is compared with the verified solver only (G1). St: not applicable (all Phase 4 problems are steady).",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def figures(bench: dict, tag: str, ex) -> list[str]:
    paths = []
    for name, B in bench.items():
        b, rq, rs = B["cfg"], B["ref"], B["runs"]
        if b["kind"] == "ghia_cavity":
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            for ax, w in zip(axs, ("u", "v"), strict=True):
                c, val, mask = bm.ghia_points(int(b["Re"]), w)
                ax.plot(c[mask], val[mask], "ks", ms=4, mfc="none", label="Ghia et al. 1982")
                cl = rq["centrelines"]
                ax.plot(cl["s"], cl[w], "k-", lw=0.8, label=f"solver {rq['n_fine']}²")
                for i, r in enumerate(rs):
                    ax.plot(
                        r["centrelines"]["s"],
                        r["centrelines"][w],
                        "-",
                        color=SEED_COLORS[i % 5],
                        lw=1,
                        label=f"PINN seed {r['seed']}",
                    )
                ax.set(xlabel="y (x = 0.5)" if w == "u" else "x (y = 0.5)", ylabel=w, title=f"{name}: {w} centreline")
                ax.legend(fontsize=6)
            paths += save_fig(fig, f"p4_gate_{name}_{tag}", ex.dir)
        elif b["kind"] == "graetz_cwt":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rq["q"]["x"], rq["q"]["Nu_x"], "k-", lw=1, label="solver")
            for i, r in enumerate(rs):
                ax.plot(r["x"], r["Nu_x"], "-", color=SEED_COLORS[i % 5], lw=1, label=f"PINN seed {r['seed']}")
            ax.axhline(B["literature"]["Nu_axial"], color="0.4", ls="--", label="eigenvalue (axial conduction)")
            ax.axhline(B["literature"]["Nu_no_axial"], color="0.6", ls=":", label="eigenvalue (no axial conduction)")
            ax.set(xlabel="x / H", ylabel="local Nu_Dh", title=f"{name}: Pe = {B['literature']['Pe']:g}")
            ax.legend(fontsize=6)
            paths += save_fig(fig, f"p4_gate_{name}_{tag}", ex.dir)
        else:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            lit = B["literature"]
            ax.axhline(lit["Nu"], color="k", ls="--", label="de Vahl Davis")
            ax.plot([0], [rq["q"]["Nu_hot"]], "ks", label=f"solver {rq['n_fine']}²")
            for i, r in enumerate(rs):
                ax.plot([1 + i], [r["Nu_hot"]], "o", color=SEED_COLORS[i % 5], label=f"PINN seed {r['seed']}")
            ax.set(xticks=[], ylabel="mean hot-wall Nu", title=f"{name}: Ra = {b['Ra']:g}")
            ax.legend(fontsize=6)
            paths += save_fig(fig, f"p4_gate_{name}_{tag}", ex.dir)
    return [str(p) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--compile-only", action="store_true", help="G1 + G2 from the saved runs, no benchmark training")
    a = ap.parse_args()
    tag = a.tag or a.profile
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    g = spec["gate"]
    rows = compile_rows(g["compile"])
    ab = None
    adir = ROOT / "results" / g["ablation_run"]
    if (adir / "metrics.json").exists():
        am = json.loads((adir / "metrics.json").read_text(encoding="utf-8"))
        agg = ablation_aggregate(am["records"])
        ab = {
            "run": g["ablation_run"],
            "git": am["git"][:7],
            "agg": {n: agg[n] for n in g["ablation_rows"] if n in agg},
        }
    apply_style()
    with Experiment(f"p4_gate_compile_{tag}" if a.compile_only else f"p4_gate_{tag}", spec) as ex:
        bench = None
        saved_gate = ROOT / "results" / f"p4_gate_{tag}" / "metrics.json"
        if a.compile_only and saved_gate.exists():
            # Phase 7: the benchmark figures from the saved G3 records (curves and references are in the metrics),
            # without training the benchmark networks again
            from aerosynapse.thermal.utils import replay as RP

            saved_bench = json.loads(saved_gate.read_text(encoding="utf-8")).get("benchmarks")
            if saved_bench:
                for f in figures(saved_bench, tag, RP.target(f"p4_gate_{tag}")):
                    print(f)
        if not a.compile_only:
            names = list(g["benchmarks"])
            refs = run_tasks(
                [(MOD, "bench_reference", dict(spec=spec, name=n)) for n in names],
                spec["execution"],
                lambda i, r, s: print(
                    f"[reference {names[i]}] {r['wall_s']:.0f}s {r['quantities'] if 'Nu_hot' in r['quantities'] else ''}",
                    flush=True,
                ),
            )
            ref_of = dict(zip(names, refs, strict=True))
            jobs, labels = [], []
            for n in names:
                for seed in g["seeds"]:
                    jobs.append(
                        (
                            MOD,
                            "run_benchmark",
                            dict(
                                spec=spec,
                                name=n,
                                seed=seed,
                                ref=ref_of[n],
                                out_dir=ex.dir / "runs" / f"{n}_s{seed}",
                                tag=tag,
                            ),
                        )
                    )
                    labels.append((n, seed))

            def show(i: int, r: dict, sec: float) -> None:
                extra = {k: r[k] for k in ("Nu_hot", "Nu_mean") if k in r}
                if "ghia_u" in r:
                    extra = {"ghia_u_rms": round(r["ghia_u"]["rms"], 4), "ghia_v_rms": round(r["ghia_v"]["rms"], 4)}
                print(
                    f"[{r['benchmark']} s{r['seed']}] worst {r['rel_l2_max']:.3e} {extra} train {r['train_wall_s']:.0f}s {r['status']}",
                    flush=True,
                )

            outs = run_tasks(jobs, spec["execution"], show)
            bench = {}
            for n in names:
                rf = ref_of[n]
                bench[n] = {
                    "cfg": g["benchmarks"][n],
                    "literature": literature(g["benchmarks"][n]),
                    "ref": {
                        "n_fine": rf.get("n_fine"),
                        "q": rf["quantities"],
                        "coarse_fine_diff": rf["coarse_fine_diff"],
                        "centrelines": rf.get("centrelines"),
                        "wall_s": rf["wall_s"],
                    },
                    "runs": [o for (m, _), o in zip(labels, outs, strict=True) if m == n],
                }
            for (n, seed), o in zip(labels, outs, strict=True):
                ex.log(
                    benchmark=n,
                    seed=seed,
                    rel_l2_max=o["rel_l2_max"],
                    train_wall_s=o["train_wall_s"],
                    status=o["status"],
                    Nu=o.get("Nu_hot", o.get("Nu_mean")),
                    ghia_u_rms=o.get("ghia_u", {}).get("rms"),
                    ghia_v_rms=o.get("ghia_v", {}).get("rms"),
                )
            ex.metrics["figures"] = figures(bench, tag, ex)
        ex.metrics.update(
            {"compiled": rows, "ablation": ab, "benchmarks": bench, "git": git_hash(), "torch": torch.__version__}
        )
        custom = a.config or a.sets or tag != spec["profile"] or a.compile_only
        table = ex.dir / "gate.md" if custom else ROOT / "docs" / "tables" / "phase4_gate.md"
        gate_report(rows, ab, bench, spec, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
