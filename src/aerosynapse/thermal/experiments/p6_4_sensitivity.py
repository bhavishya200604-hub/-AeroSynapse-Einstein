"""Step 6.4: global sensitivity — Sobol indices of mean Nu w.r.t. Re, Pr, Bi from the PINN surrogate, cross-checked
against the verified FV solver (Monte Carlo and polynomial chaos).

Per case (sensitivity_study.cases): a saved PARAMETRIC PINN (members = seeds or variants) and the solver on the same
log-uniform box (default: the surrogate's training box).
  solver      Saltelli design (scrambled Sobol, base N), N (d + 2) solves in worker processes; first-order (Saltelli
              2010) and total (Jansen) indices with bootstrap intervals; convergence over prefixes of the design;
              PCE (total degree p) on the same solves with leave-one-out error; grid check at the corners and centre
  surrogate   the same design extended to N_pinn (its first N rows are the solver's): indices of every member and of
              the ensemble mean; paired QoI errors on the solver's points; a RELOAD CHECK reproduces the QoI that the
              training run recorded at its test points (proves the networks are rebuilt exactly)

    python experiments/p6_4_sensitivity.py [--profile smoke] [--config my.yaml] [--set key=value] [--tag name]
                                           [--cases D_graetz,C_cavity] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aerosynapse.thermal.pinn import sensitivity as SA
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import sensitivity_figs as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_4_sensitivity"
NAMES = ["Re", "Pr", "Bi"]


# --------------------------------------------------------------------------- case adapters
class Graetz:
    """Step 4.7 ablation run: one configuration, members = its seeds. QoI mean Nu = -mean dtheta/dy, bottom wall."""

    qoi = "mean Nu (bottom wall)"

    def __init__(self, cfg: dict):
        from aerosynapse.thermal.experiments import p4_6_hybrid as HY
        from aerosynapse.thermal.experiments import p4_7_ablation as A7
        from aerosynapse.thermal.experiments.p4_5b_channel import case_design
        from aerosynapse.thermal.pinn import channel2d as CH

        self.run = ROOT / cfg["surrogate"]["run"]
        self.conf = cfg["surrogate"]["configuration"]
        cfgfile = str(self.run / "config.yaml")
        self.base = SP.load_spec([cfgfile], "smoke")
        ab = self.base["ablation_study"]
        self.case = ab["case"]
        on = next(c for c in A7.configurations(ab) if c["name"] == self.conf)["on"]
        s = A7.config_spec([cfgfile], "smoke", {}, ab, on)
        self.design = case_design(self.base, self.case)
        self.pcfg = SP.deep_merge(self.base["problems2d"][self.case], (ab.get("problems") or {}).get(self.case) or {})
        self.box = self.design["train_box"]
        self.fixed = dict(self.design["fixed"])
        self.Lx = float(self.pcfg["geometry"]["Lx"])
        self.members, self.specs = [], []  # specs: each member's training spec (Step 6.6b rebuilds init / batch)
        for seed in ab["seeds"]:
            s0 = HY.study_spec(s, seed)
            form = SP.deep_merge(s0["formulation"], s0["hybrid_study"]["formulation"])
            prob = CH.build_channel(
                self.case, s0["problems2d"][self.case], form, self.design["dims"], self.fixed, self.design["u_lim"]
            )
            net = HY._net(s0, prob)
            net.load_state_dict(
                torch.load(self.run / "runs" / f"{self.conf}_s{seed}" / "checkpoint.pt", weights_only=False)["model"]
            )
            net.eval()
            self.members.append((f"seed {seed}", prob, net))
            self.specs.append(s0)
        self.log_dirs = [self.run / "runs" / f"{self.conf}_s{seed}" for seed in ab["seeds"]]

    def initial(self, i: int) -> torch.nn.Module:
        """The member's network at initialisation (same seed and construction as in training)."""
        from aerosynapse.thermal.experiments import p4_6_hybrid as HY

        return HY._net(self.specs[i], self.members[i][1])

    def pinn(self, prob, net, P: np.ndarray, n_wall: int, chunk: int = 200) -> np.ndarray:
        x = (np.arange(n_wall) + 0.5) / n_wall * self.Lx
        return _wall_mean(prob, net, P, np.stack([x, np.zeros_like(x)], 1), "theta_y", -1.0, chunk)

    def recorded(self) -> list[tuple[dict, float]]:
        met = json.loads((self.run / "metrics.json").read_text(encoding="utf-8"))
        return [
            ({k: r[k] for k in NAMES}, -float(r["q_bottom"]), int(r["seed"]))
            for r in met["records"]
            if r.get("config") == self.conf
        ]

    @staticmethod
    def solve_points(P: list[list[float]], pcfg: dict, fixed: dict, cells_per_unit: int) -> list[float]:
        from aerosynapse.thermal.pinn import channel2d as CH

        return [
            -float(
                CH.solve_channel({**fixed, **dict(zip(NAMES, p, strict=True))}, pcfg, cells_per_unit, 1e-10, "quick")[
                    "q_bottom"
                ]
            )
            for p in P
        ]

    def solver_job(self, P, grid_cfg: dict) -> tuple:
        return (
            "Graetz.solve_points",
            dict(P=P, pcfg=self.pcfg, fixed=self.fixed, cells_per_unit=int(grid_cfg["cells_per_unit"])),
        )


class Cavity:
    """Step 4.5a parametric run: members = its parametric variants (seed 0). QoI Nu_top (as recorded in 4.5a)."""

    qoi = "Nu_top"

    def __init__(self, cfg: dict):
        from aerosynapse.thermal.experiments import p4_5_parametric as P45
        from aerosynapse.thermal.pinn import param2d as PP
        from aerosynapse.thermal.pinn import problems2d as P2
        from aerosynapse.thermal.pinn.architectures import build_network

        self.run = ROOT / cfg["surrogate"]["run"]
        self.base = SP.load_spec([str(self.run / "config.yaml")], "smoke")
        self.design = P45.study_design(self.base)
        self.box = self.design["train_box"]
        self.fixed = dict(self.design["fixed"])
        self.cond = P2.cavity_conditions(self.base["problems2d"]["C_cavity"])
        self.scheme = self.base["param_study"]["reference"]["scheme"]
        variants = {v["name"]: v for v in self.base["param_study"]["variants"]}
        seeds = [int(x) for x in cfg["surrogate"].get("seeds", [0])]  # default: seed 0 (Steps 6.4 / 6.6b)
        self.members, self.specs, self.log_dirs = [], [], []
        for name in cfg["surrogate"]["variants"]:
            for seed in seeds:
                s = P45.variant_spec(self.base, variants[name])
                s["seed"] = seed
                form = SP.deep_merge(s["formulation"], s["param_study"]["formulation"])
                prob = PP.build_param_cavity(
                    s["problems2d"]["C_cavity"], form, self.design["dims"], self.fixed, self.design["u_lim"]
                )
                net = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(dtype=torch.float64)
                d = self.run / "runs" / f"{name}_s{seed}"
                net.load_state_dict(torch.load(d / "checkpoint.pt", weights_only=False)["model"])
                net.eval()
                self.members.append((name if seeds == [0] else f"{name} seed {seed}", prob, net))
                self.specs.append(s)
                self.log_dirs.append(d)

    def initial(self, i: int) -> torch.nn.Module:
        from aerosynapse.thermal.pinn.architectures import build_network

        prob = self.members[i][1]
        torch.manual_seed(int(self.specs[i]["seed"]))
        return build_network(self.specs[i]["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(
            dtype=torch.float64
        )

    def pinn(self, prob, net, P: np.ndarray, n_wall: int, chunk: int = 200) -> np.ndarray:
        x = (np.arange(n_wall) + 0.5) / n_wall
        return _wall_mean(prob, net, P, np.stack([x, np.ones_like(x)], 1), "theta_y", 1.0, chunk)

    def recorded(self) -> list[tuple[dict, float]]:
        met = json.loads((self.run / "metrics.json").read_text(encoding="utf-8"))
        names = [m[0] for m in self.members]
        return [
            ({k: r[k] for k in NAMES}, float(r["nu_top"]), r["variant"])
            for r in met["records"]
            if r["variant"] in names
        ]

    @staticmethod
    def solve_points(P: list[list[float]], cond: dict, fixed: dict, grid: int, scheme: str) -> list[float]:
        from aerosynapse.thermal.pinn.ref2d import solve_steady_decoupled

        return [
            float(
                solve_steady_decoupled({**fixed, **dict(zip(NAMES, p, strict=True))}, cond, grid, 1e-10, scheme)[
                    "nu_top"
                ]
            )
            for p in P
        ]

    def solver_job(self, P, grid_cfg: dict) -> tuple:
        return (
            "Cavity.solve_points",
            dict(P=P, cond=self.cond, fixed=self.fixed, grid=int(grid_cfg["grid"]), scheme=self.scheme),
        )


ADAPTERS = {"D_graetz": Graetz, "C_cavity": Cavity}


def _wall_mean(prob, net, P: np.ndarray, wall_xy: np.ndarray, field: str, sign: float, chunk: int) -> np.ndarray:
    """sign * mean over the wall points of one first-derivative field, for every parameter row of P (Re, Pr, Bi)."""
    p0 = next(net.parameters())
    out = np.empty(len(P))
    nw = len(wall_xy)
    names = [d.name for d in prob.sweep]
    for i in range(0, len(P), chunk):
        blk = P[i : i + chunk]
        enc = np.stack([prob.encode(dict(zip(NAMES, row, strict=True))) for row in blk])
        X = np.concatenate([np.tile(wall_xy, (len(blk), 1)), np.repeat(enc, nw, axis=0)], 1)
        assert names == NAMES[: len(names)], names
        d = prob.fields(net, torch.as_tensor(X, dtype=p0.dtype, device=p0.device), order=1)
        out[i : i + len(blk)] = sign * d[field].detach().cpu().numpy().reshape(len(blk), nw).mean(axis=1)
    return out


def solve_chunk(case: str, method: str, kw: dict) -> list[float]:
    """Worker: one chunk of solver points (own process)."""
    cls = ADAPTERS[case]
    return getattr(cls, method.split(".")[1])(**kw)


# --------------------------------------------------------------------------- the study of one case
def run_case(name: str, cfg: dict, st: dict, ex) -> dict:
    t0 = time.perf_counter()
    ad = ADAPTERS[name](cfg)
    box = (
        ad.box
        if cfg.get("ranges", "train_box") == "train_box"
        else {k: list(map(float, v)) for k, v in cfg["ranges"].items()}
    )
    ranges = [(float(box[k][0]), float(box[k][1]), "log") for k in NAMES]
    outside = {k: [box[k][0] < ad.box[k][0] * (1 - 1e-9), box[k][1] > ad.box[k][1] * (1 + 1e-9)] for k in NAMES}
    d = 3
    # reload check: QoI at the training run's own test points, as recorded there
    rec = ad.recorded()
    by_member = {m[0]: m for m in ad.members}
    diffs = []
    for vals, q_rec, who in rec:
        key = f"seed {who}" if isinstance(who, int) else who
        _, prob, net = by_member[key]
        q = ad.pinn(prob, net, np.array([[vals[k] for k in NAMES]]), int(cfg["pinn"]["n_wall"]))[0]
        diffs.append(abs(q - q_rec) / max(abs(q_rec), 1e-300))
    reload = {"n": len(diffs), "max_rel_diff": float(max(diffs)) if diffs else None}
    print(
        f"{name}: reload check on {len(diffs)} recorded points, max rel diff {reload['max_rel_diff']:.2e}", flush=True
    )
    # solver: Saltelli design, chunks in worker processes
    n_s, n_p = int(cfg["solver"]["n"]), int(cfg["pinn"]["n"])
    A, B, AB = SA.saltelli(max(n_s, n_p), d, int(st["seed"]))
    U_s = SA.stack(A[:n_s], B[:n_s], [m[:n_s] for m in AB])
    P_s = SA.to_phys(U_s, ranges)
    ch = int(st["chunk"])
    chunks = [P_s[i : i + ch].tolist() for i in range(0, len(P_s), ch)]
    jobs = []
    for c in chunks:
        meth, kw = ad.solver_job(c, cfg["solver"])
        jobs.append((MOD, "solve_chunk", dict(case=name, method=meth, kw=kw)))
    corners = np.array([[a, b, c] for a in (0.0, 1.0) for b in (0.0, 1.0) for c in (0.0, 1.0)] + [[0.5, 0.5, 0.5]])
    P_c = SA.to_phys(corners, ranges).tolist()
    for g in (cfg["solver"], cfg["check_grid"]):
        meth, kw = ad.solver_job(P_c, g)
        jobs.append((MOD, "solve_chunk", dict(case=name, method=meth, kw=kw)))
    ts = time.perf_counter()
    outs = run_tasks(
        jobs,
        {"isolate": True, **st.get("execution", {})},
        lambda i, r, sec: (
            print(f"{name} solver chunk {i + 1}/{len(jobs)}: {sec:.0f} s", flush=True)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs)
            else None
        ),
    )
    solver_wall = time.perf_counter() - ts
    f_s = np.concatenate([np.asarray(o) for o in outs[: len(chunks)]])
    q_sa, q_fine = np.asarray(outs[-2]), np.asarray(outs[-1])
    grid_check = {
        "points": P_c,
        "qoi": q_sa.tolist(),
        "qoi_check_grid": q_fine.tolist(),
        "max_rel_diff": float(np.max(np.abs(q_sa - q_fine) / np.maximum(np.abs(q_fine), 1e-300))),
    }
    solver = SA.indices(f_s, n_s, d) | SA.bootstrap(f_s, n_s, d, int(st["bootstrap"]), int(st["seed"]))
    conv = []
    blocks = f_s.reshape(d + 2, n_s)
    for m in sorted({int(v) for v in st["convergence"] if int(v) <= n_s} | {n_s}):
        fm = blocks[:, :m].ravel()
        conv.append(
            {"N": m} | SA.indices(fm, m, d) | SA.bootstrap(fm, m, d, min(int(st["bootstrap"]), 500), int(st["seed"]))
        )
    pce = SA.pce_fit(U_s, f_s, int(cfg["pce_degree"]))
    # surrogate: the extended design; members and ensemble mean; paired comparison on the solver's points
    U_p = SA.stack(A[:n_p], B[:n_p], [m[:n_p] for m in AB])
    P_p = SA.to_phys(U_p, ranges)
    tp = time.perf_counter()
    F_p = {m[0]: ad.pinn(m[1], m[2], P_p, int(cfg["pinn"]["n_wall"])) for m in ad.members}
    pinn_wall = time.perf_counter() - tp
    ens = np.mean(list(F_p.values()), axis=0)
    pinn = {"ensemble": SA.indices(ens, n_p, d) | SA.bootstrap(ens, n_p, d, int(st["bootstrap"]), int(st["seed"]))}
    for k, f in F_p.items():
        pinn[k] = SA.indices(f, n_p, d)
    # paired: the same N_s rows (blockwise prefix of the extended design)
    pb = ens.reshape(d + 2, n_p)[:, :n_s].ravel()
    pinn["ensemble_same_design"] = SA.indices(pb, n_s, d) | SA.bootstrap(
        pb, n_s, d, int(st["bootstrap"]), int(st["seed"])
    )
    rel = np.abs(pb - f_s) / np.maximum(np.abs(f_s), 1e-300)
    paired = {
        "rel_err_median": float(np.median(rel)),
        "rel_err_p95": float(np.percentile(rel, 95)),
        "rel_err_max": float(rel.max()),
        "bias_mean_rel": float(np.mean((pb - f_s) / np.maximum(np.abs(f_s), 1e-300))),
        "sample": {"solver": f_s[: min(len(f_s), 2000)].tolist(), "pinn": pb[: min(len(pb), 2000)].tolist()},
    }
    for k, f in F_p.items():
        fb = f.reshape(d + 2, n_p)[:, :n_s].ravel()
        paired[f"rel_err_median_{k}"] = float(np.median(np.abs(fb - f_s) / np.maximum(np.abs(f_s), 1e-300)))
    np.savez_compressed(
        ex.dir / f"{name}_outputs.npz",
        U_s=U_s,
        f_solver=f_s,
        U_pinn=U_p,
        **{f"pinn_{i}": f for i, f in enumerate(F_p.values())},
    )
    return {
        "qoi": ad.qoi,
        "box": {k: list(map(float, box[k])) for k in NAMES},
        "train_box": {k: list(map(float, ad.box[k])) for k in NAMES},
        "outside_train_box": outside,
        "members": [m[0] for m in ad.members],
        "reload_check": reload,
        "grid_check": grid_check,
        "solver": solver | {"n": n_s, "n_runs": int(n_s * (d + 2)), "wall_s": solver_wall, "settings": cfg["solver"]},
        "convergence": conv,
        "pce": {k: v for k, v in pce.items() if k not in ("coef", "alphas")},
        "pinn": pinn | {"n": n_p, "n_runs": int(n_p * (d + 2)), "wall_s": pinn_wall},
        "paired": paired,
        "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--cases", default=None, help="comma-separated subset of sensitivity_study.cases")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    cli = SP.parse_set(a.sets)
    spec = SP.load_spec(a.config or None, a.profile, cli)
    st = spec["sensitivity_study"]
    custom = bool(a.config or a.sets or a.tag != "default" or a.profile != "smoke" or a.cases)
    odir = ROOT / "results" / f"p6_4_sensitivity_{a.tag}"
    table = odir / "sensitivity.md" if custom else ROOT / "docs" / "tables" / "phase6_4_sensitivity.md"
    if a.figures_only:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        met["figures"] = figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        print(f"wrote {table}")
        return
    cases = [c.strip() for c in a.cases.split(",")] if a.cases else list(st["cases"])
    with Experiment(f"p6_4_sensitivity_{a.tag}", spec) as ex:
        ex.metrics["cases"] = {}
        for c in cases:
            ex.metrics["cases"][c] = run_case(c, st["cases"][c], st, ex)
            print(f"{c}: done in {ex.metrics['cases'][c]['wall_s']:.0f} s", flush=True)
        ex.metrics["git"] = git_hash()
        ex.metrics["figures"] = figures(ex.metrics, a.tag, ex.dir, docs=not custom)
        report(ex.metrics, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def _ci(v, ci) -> str:
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Step 6.4 global sensitivity — `{tag}` (auto-generated by experiments/p6_4_sensitivity.py)",
        "",
        "Sobol indices of the QoI w.r.t. Re, Pr, Bi (independent, log-uniform on the box). Solver: verified FV, Saltelli "
        "design, first order S (Saltelli 2010) and total ST (Jansen) with bootstrap 95 % intervals, and a PCE on the same "
        "solves. Surrogate: the saved parametric PINN (members and ensemble mean) on the same design extended. "
        "Verification only (solver vs surrogate; no measured data).",
        "",
    ]
    for name, c in met["cases"].items():
        b = c["box"]
        L += [
            f"## {name}: {c['qoi']} on Re [{b['Re'][0]:.3g}, {b['Re'][1]:.3g}], Pr [{b['Pr'][0]:.3g}, {b['Pr'][1]:.3g}], "
            f"Bi [{b['Bi'][0]:.3g}, {b['Bi'][1]:.3g}]",
            "",
            f"Surrogate members: {', '.join(c['members'])}. Reload check: the rebuilt networks reproduce the QoI recorded by "
            f"their training run at {c['reload_check']['n']} points to max rel diff {c['reload_check']['max_rel_diff']:.1e}. "
            f"Solver: {c['solver']['n_runs']} solves ({c['solver']['wall_s']:.0f} s), settings {c['solver']['settings']}; "
            f"grid check (corners + centre) max rel diff {c['grid_check']['max_rel_diff']:.2g}. "
            f"PCE degree {c['pce']['degree']} ({c['pce']['n_terms']} terms), LOO rel error {c['pce']['loo_rel_error']:.2g}. "
            f"Surrogate: {c['pinn']['n_runs']} evaluations per member ({c['pinn']['wall_s']:.0f} s for all members).",
            "",
            "| index | input | solver MC [95 %] | solver PCE | PINN ensemble [95 %] | PINN ensemble, solver design [95 %] | PINN members |",
            "|---|---|---|---|---|---|---|",
        ]
        mem = [k for k in c["pinn"] if k not in ("ensemble", "ensemble_same_design", "n", "n_runs", "wall_s")]
        for idx, cik in (("S", "S_ci95"), ("ST", "ST_ci95")):
            for i, k in enumerate(NAMES):
                L.append(
                    f"| {idx} | {k} | {_ci(c['solver'][idx][i], c['solver'][cik][i])} | {c['pce'][idx][i]:.3f} | "
                    f"{_ci(c['pinn']['ensemble'][idx][i], c['pinn']['ensemble'][cik][i])} | "
                    f"{_ci(c['pinn']['ensemble_same_design'][idx][i], c['pinn']['ensemble_same_design'][cik][i])} | "
                    + ", ".join(f"{c['pinn'][m][idx][i]:.3f}" for m in mem)
                    + " |"
                )
        p = c["paired"]
        L += [
            "",
            f"QoI: solver mean {c['solver']['mean']:.4g}, variance {c['solver']['var']:.4g}; PINN ensemble mean "
            f"{c['pinn']['ensemble']['mean']:.4g}, variance {c['pinn']['ensemble']['var']:.4g}. Paired PINN vs solver on "
            f"{c['solver']['n_runs']} points: relative error median {p['rel_err_median']:.3g}, 95th percentile "
            f"{p['rel_err_p95']:.3g}, max {p['rel_err_max']:.3g}, mean signed {p['bias_mean_rel']:+.3g}; per member median "
            + ", ".join(
                f"{k.replace('rel_err_median_', '')} {v:.3g}" for k, v in p.items() if k.startswith("rel_err_median_")
            )
            + ".",
            "",
            "Solver index convergence (ST, N = "
            + " / ".join(str(r["N"]) for r in c["convergence"])
            + "): "
            + "; ".join(
                f"{k} " + " / ".join(f"{r['ST'][i]:.3f}" for r in c["convergence"]) for i, k in enumerate(NAMES)
            )
            + ".",
            "",
        ]
        if any(any(v) for v in c["outside_train_box"].values()):
            L += [
                "**Box extends beyond the surrogate's training box (extrapolation, see Step 6.3):** "
                + str(c["outside_train_box"]),
                "",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    paths = []
    paths += save_fig(F.indices(met["cases"]), f"p6_4_indices_{tag}", odir, docs=docs)
    paths += save_fig(F.convergence(met["cases"]), f"p6_4_convergence_{tag}", odir, docs=docs)
    paths += save_fig(F.paired(met["cases"]), f"p6_4_paired_{tag}", odir, docs=docs)
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
