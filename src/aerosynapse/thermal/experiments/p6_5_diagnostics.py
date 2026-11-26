"""Step 6.5: why a PINN trains or fails — spectral bias, NTK eigen-spectrum, Hessian and loss landscape.

(A) For every case x variant (diagnostics_study): train one network (profile budget) and keep weight snapshots; per
    snapshot: truth error, its DCT spectrum (which frequencies are learnt when), the truth-free loss; at ntk_snapshots
    of them: the residual NTK (spectrum, block traces per group, drift from the initial kernel, share of the PDE residual
    in the top kernel directions) and the extreme Hessian eigenvalues; at the final weights: the full Hessian spectrum,
    a 2D loss-landscape slice (trajectory-PCA or filter-normalised random directions) with the projected trajectory, and
    the loss along the straight line from the initial to the final weights.
(B) Width study: the kernel drift ||K_end - K_0|| / ||K_0|| vs width (lazy / NTK regime).
(C) Sigma scan (Fourier features): kernel-target alignment of the OUTPUT NTK at initialisation with a coarse FD solution
    (truth-free, before any training) vs the error after training — can the NTK predict a good sigma?
(D) Toy spectral bias: plain least-squares fit of sum_k sin(2 pi k x); the residual amplitude of every frequency vs epoch.

    python experiments/p6_5_diagnostics.py [--profile smoke] [--config my.yaml] [--set key=value] [--tag name]
                                           [--parts all|cases,width,sigma,toy] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aerosynapse.thermal.pinn import diagnostics as D
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import diagnostics_figs as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_5_diagnostics"


def _spec(files, profile, cli, over: dict, seed: int) -> dict:
    return SP.load_spec(files or None, profile, SP.deep_merge(cli, SP.deep_merge(over, {"seed": int(seed)})))


def _net(spec, prob):
    from aerosynapse.thermal.pinn.architectures import build_network

    torch.manual_seed(int(spec["seed"]))
    return build_network(spec["network"], prob.lo, prob.hi).to(dtype=torch.float64)


def _truth_grid(prob, value: float, nx: int):
    x = np.linspace(0.0, 1.0, nx)
    t = np.array([prob.T]) if prob.has_time else None
    y = prob.reference(x, t, value)
    return x, t, (y[-1] if prob.has_time else y)


def _predict(net, prob, x, t, value):
    from aerosynapse.thermal.pinn.metrics import predict_grid

    u = predict_grid(net, prob, x, t, value)[0]
    return u[-1] if prob.has_time else u


def train_with_snapshots(spec, prob, out_dir: Path, every: int):
    from aerosynapse.thermal.pinn.trainer import Trainer

    spec = SP.deep_merge(spec, {"training": {"eval_every": int(every)}})
    net = _net(spec, prob)
    snaps = [D.flat(net)]

    def eval_fn(m):
        snaps.append(D.flat(m))
        return {"snap": len(snaps) - 1}

    tr = Trainer(net, prob, spec, out_dir, eval_fn=eval_fn, tag="diag")
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    steps = [("init", 0)] + [(r["phase"], int(r["step"])) for r in tr.history if "snap" in r]
    return net, snaps, steps[: len(snaps)], res, wall


# --------------------------------------------------------------------------- (A) one case x variant
def case_task(files, profile, cli, ds: dict, case: str, variant: str, out_dir: str) -> dict:
    c = ds["cases"][case]
    spec = _spec(files, profile, cli, ds["variants"][variant] or {}, ds["seed"])
    prob = P.build_problem(c["problem"], spec["problems"][c["problem"]], float(c["value"]))
    value = float(c["value"])
    net, snaps, steps, res, wall = train_with_snapshots(spec, prob, Path(out_dir), ds["snapshot_every"])
    final = D.flat(net)
    pp = ds["points"]
    pts = D.point_set(prob, int(pp["n_pde"]), int(pp["n_ic"]), int(pp["n_t"]))
    x, t, y = _truth_grid(prob, value, int(ds["spectrum"]["nx"]))
    ny = max(np.linalg.norm(y), 1e-300)
    rec = {"rel_l2": [], "loss": [], "spectrum": []}
    for th in snaps:
        D.set_flat(net, th)
        e = _predict(net, prob, x, t, value) - y
        rec["rel_l2"].append(float(np.linalg.norm(e) / ny))
        rec["loss"].append(float(D.loss(net, prob, pts).detach()))
        rec["spectrum"].append(D.dct_amplitudes(e)[:256].tolist())
    rec["spectrum_target"] = D.dct_amplitudes(y)[:256].tolist()
    # NTK and Hessian extremes at evenly spread snapshots
    idx = sorted({int(round(v)) for v in np.linspace(0, len(snaps) - 1, int(ds["ntk_snapshots"]))})
    ntk = []
    K0 = None
    for i in idx:
        D.set_flat(net, snaps[i])
        J, lab = D.jacobian(net, prob, pts)
        st = D.ntk_stats(J, lab)
        K = st.pop("_K")
        K0 = K if K0 is None else K0
        r = torch.cat([v.detach().reshape(-1) for v in D.residuals(net, prob, pts).values()]).numpy()
        w, V = np.linalg.eigh(K)
        coef = V[:, ::-1].T @ r
        share = np.cumsum(coef**2) / max(float(r @ r), 1e-300)
        e = {
            "snapshot": i,
            "phase": steps[i][0],
            "step": steps[i][1],
            "drift": D.drift(K0, K),
            "residual_share_top": [float(share[k - 1]) for k in (1, 5, 10, 20)] + [float(share[-1])],
        }
        e |= {k: v for k, v in st.items() if k != "pde_eigenvectors" or i in (idx[0], idx[-1])}
        if ds["hessian"].get("top_every_snapshot", True):
            e["hessian"] = D.hessian_extremes(net, prob, pts)
        ntk.append(e)
    D.set_flat(net, final)
    out = {
        "case": case,
        "variant": variant,
        "problem": c["problem"],
        "value": value,
        "steps": steps,
        "status": res["status"],
        "wall_s": wall,
        "n_params": int(final.numel()),
        **rec,
        "ntk": ntk,
        "points": pp,
    }
    hs = ds["hessian"]
    if hs.get("full", True) and final.numel() <= int(hs["max_params"]):
        H = D.hessian_full(net, prob, pts)
        lam = np.linalg.eigvalsh(H)
        out["hessian_spectrum"] = lam.tolist()
    # landscape + trajectory
    ls = ds["landscape"]
    n = int(ls["n"])
    if ls["directions"] == "pca":
        d1, d2, coords = D.pca_directions(snaps, final)
        # span = multiple of the trajectory's extent along each direction
        s1 = float(ls["span"]) * 1.25 * max(np.abs(coords[:, 0]).max(), 1e-8)
        s2 = float(ls["span"]) * 1.25 * max(np.abs(coords[:, 1]).max(), 1e-8)
        out["trajectory"] = coords.tolist()
    else:  # filter-normalised random directions: span in units of the weight norms
        rng = np.random.default_rng(int(ds["seed"]))
        d1, d2 = D.filter_normalised(net, rng), D.filter_normalised(net, rng)
        s1 = s2 = float(ls["span"])
    a, b = np.linspace(-s1, s1, n), np.linspace(-s2, s2, n)
    out["landscape"] = {
        "a": a.tolist(),
        "b": b.tolist(),
        "loss": D.landscape(net, prob, pts, d1, d2, a, b).tolist(),
        "directions": ls["directions"],
    }
    s_, L_ = D.interpolate(net, prob, pts, snaps[0], final)
    out["interpolation"] = {"s": s_.tolist(), "loss": L_.tolist()}
    return out


# --------------------------------------------------------------------------- (B) width study
def width_task(files, profile, cli, ds: dict, width: int, out_dir: str) -> dict:
    w = ds["width_study"]
    c = ds["cases"][w["case"]]
    over = {
        "network": {"width": int(width)},
        "training": {"adam": {"epochs": int(w["adam_epochs"])}, "lbfgs": {"iterations": 0}},
    }
    spec = _spec(files, profile, cli, over, ds["seed"])
    prob = P.build_problem(c["problem"], spec["problems"][c["problem"]], float(c["value"]))
    net, snaps, steps, res, wall = train_with_snapshots(spec, prob, Path(out_dir), max(int(w["adam_epochs"]) // 5, 1))
    pp = w.get("points") or {"n_pde": 32, "n_ic": 16, "n_t": 4}  # J is rows x params: keep the rows few for wide nets
    pts = D.point_set(prob, int(pp["n_pde"]), int(pp["n_ic"]), int(pp["n_t"]))
    Ks, eig = [], []
    for th in snaps:
        D.set_flat(net, th)
        J, lab = D.jacobian(net, prob, pts)
        K = J @ J.T
        Ks.append(K)
        eig.append(float(np.linalg.eigvalsh(K)[-1]))
    return {
        "width": int(width),
        "n_params": int(D.flat(net).numel()),
        "steps": steps,
        "drift": [D.drift(Ks[0], K) for K in Ks],
        "lambda_max": eig,
        "param_change": float(torch.linalg.norm(snaps[-1] - snaps[0]) / torch.linalg.norm(snaps[0])),
        "wall_s": wall,
    }


# --------------------------------------------------------------------------- (C) sigma scan
def sigma_task(files, profile, cli, ds: dict, sigma: float, out_dir: str) -> dict:
    sc = ds["sigma_scan"]
    c = ds["cases"][sc["case"]]
    over = {"network": {"arch": "fourier", "fourier": {"sigma": float(sigma)}}}
    spec = _spec(files, profile, cli, over, ds["seed"])
    prob = P.build_problem(c["problem"], spec["problems"][c["problem"]], float(c["value"]))
    value = float(c["value"])
    net = _net(spec, prob)
    xg = np.linspace(0.0, 1.0, 129)
    # truth-free target: a coarse FD solution (64 cells, central), interpolated (steady problems only)
    from aerosynapse.thermal.numerics import fd1d

    cc = prob.cond
    y_fd = None
    if prob.name == "A_steady":
        f = fd1d.ConvDiff1D(
            64,
            scheme="central",
            a=cc["c"],
            D=1.0 / value,
            left=fd1d.dirichlet(cc["theta_left"]),
            right=fd1d.dirichlet(cc["theta_right"]),
        )
        y_fd = np.interp(xg, f.x, f.nodal(f.steady(), 0.0))
    Ju = D.output_jacobian(net, prob, xg)
    Ku = Ju @ Ju.T
    u0 = _predict(net, prob, xg, None, value)
    out = {"sigma": float(sigma), "alignment_fd": D.alignment(Ku, y_fd - u0) if y_fd is not None else None}
    lam = np.clip(np.linalg.eigvalsh(Ku)[::-1], 0, None)
    out["output_ntk_eigenvalues"] = lam[:64].tolist()
    if sc.get("train", True):
        from aerosynapse.thermal.experiments.p3_pinn1d import train_one

        r = train_one(spec, prob, [value], Path(out_dir), f"sigma{sigma:g}")["records"][0][0]
        out |= {"rel_l2": r["rel_l2"], "val_loss": r["val_loss"], "wall_s": r["wall_s"]}
    return out


# --------------------------------------------------------------------------- (E) gradient conflict (post hoc)
def conflict_task(files, profile, cli, ds: dict, case: str, variant: str, ckpt: str) -> dict:
    """Per-group gradients of the unweighted loss at the INITIAL and the FINAL (saved) weights: g_pde = grad mean r_pde^2,
    g_cond = grad of the boundary (+ initial) terms; norms, cosine, and |g_pde + g_cond| / (|g_pde| + |g_cond|)
    (near 0: the groups cancel — a stationary point that is not a minimum of either group)."""
    c = ds["cases"][case]
    spec = _spec(files, profile, cli, ds["variants"][variant] or {}, ds["seed"])
    prob = P.build_problem(c["problem"], spec["problems"][c["problem"]], float(c["value"]))
    net = _net(spec, prob)
    pp = ds["points"]
    pts = D.point_set(prob, int(pp["n_pde"]), int(pp["n_ic"]), int(pp["n_t"]))

    def stats():
        r = D.residuals(net, prob, pts)
        ps = D.params(net)
        lp = torch.mean(r["pde"] ** 2)
        lc = sum(torch.mean(v**2) for k, v in r.items() if k != "pde")

        def g(L, keep):  # parameters a group does not touch get a zero gradient
            gr = torch.autograd.grad(L, ps, retain_graph=keep, allow_unused=True)
            return torch.cat(
                [(z if z is not None else torch.zeros_like(q)).reshape(-1) for z, q in zip(gr, ps, strict=True)]
            )

        gp, gc = g(lp, True), g(lc, False)
        n1, n2, nt = float(gp.norm()), float(gc.norm()), float((gp + gc).norm())
        return {
            "loss_pde": float(lp.detach()),
            "loss_cond": float(lc.detach()),
            "grad_pde": n1,
            "grad_cond": n2,
            "cosine": float(gp @ gc / max(n1 * n2, 1e-300)),
            "cancellation": nt / max(n1 + n2, 1e-300),
        }

    init = stats()
    net.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
    return {"case": case, "variant": variant, "init": init, "final": stats()}


# --------------------------------------------------------------------------- (D) toy spectral bias
def toy_task(files, profile, cli, ds: dict, variant: str) -> dict:
    from aerosynapse.thermal.pinn.architectures import build_network

    ty = ds["toy"]
    spec = _spec(files, profile, cli, ty["variants"][variant] or {}, ds["seed"])
    torch.manual_seed(int(ds["seed"]))
    net = build_network(spec["network"], [0.0], [1.0]).to(dtype=torch.float64)
    n = int(ty["n_points"])
    x = np.arange(n) / n
    ks = [int(k) for k in ty["frequencies"]]
    y = sum(np.sin(2 * np.pi * k * x) for k in ks)
    X = torch.as_tensor(x[:, None])
    Y = torch.as_tensor(y)
    opt = torch.optim.Adam(net.parameters(), lr=float(ty["lr"]))
    rec_ep, amp, full = [], [], []
    for ep in range(int(ty["epochs"]) + 1):
        if ep % int(ty["record_every"]) == 0:
            with torch.no_grad():
                r = (Y - net(X).squeeze(-1)).numpy()
            F_ = np.abs(np.fft.rfft(r)) * 2 / n
            rec_ep.append(ep)
            amp.append([float(F_[k]) for k in ks])
            full.append(F_[:64].tolist())
        opt.zero_grad()
        L = torch.mean((net(X).squeeze(-1) - Y) ** 2)
        L.backward()
        opt.step()
    return {"variant": variant, "frequencies": ks, "epochs": rec_ep, "amplitude": amp, "spectrum": full}


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--parts", default="all")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument(
        "--conflict", action="store_true", help="gradient conflict from the saved final weights (no training)"
    )
    ap.add_argument(
        "--reuse", action="store_true", help="reuse task results cached in <results>/tasks/ by an earlier run"
    )
    a = ap.parse_args()
    cli = SP.parse_set(a.sets)
    spec = SP.load_spec(a.config or None, a.profile, cli)
    ds = spec["diagnostics_study"]
    custom = bool(a.config or a.sets or a.tag != "default" or a.profile != "smoke" or a.parts != "all")
    odir = ROOT / "results" / f"p6_5_diagnostics_{a.tag}"
    table = odir / "diagnostics.md" if custom else ROOT / "docs" / "tables" / "phase6_5_diagnostics.md"
    if a.figures_only or a.conflict:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        if a.conflict:
            common = dict(files=a.config, profile=a.profile, cli=cli, ds=ds)
            todo = [(r["case"], r["variant"]) for r in met.get("cases", [])]
            jobs = [
                (
                    MOD,
                    "conflict_task",
                    common | dict(case=c, variant=v, ckpt=str(odir / "runs" / f"{c}_{v}" / "checkpoint.pt")),
                )
                for c, v in todo
            ]
            met["conflict"] = run_tasks(jobs, {"isolate": True, **ds.get("execution", {})})
            met["conflict_git"] = git_hash()
            (odir / "metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
        met["figures"] = figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        print(f"wrote {table}")
        return
    parts = {"cases", "width", "sigma", "toy"} if a.parts == "all" else {p.strip() for p in a.parts.split(",")}
    with Experiment(f"p6_5_diagnostics_{a.tag}", spec) as ex:
        common = dict(files=a.config, profile=a.profile, cli=cli, ds=ds)
        jobs, labels = [], []
        if "cases" in parts:
            for c in ds["cases"]:
                for v in ds["variants"]:
                    jobs.append(
                        (MOD, "case_task", common | dict(case=c, variant=v, out_dir=str(ex.dir / "runs" / f"{c}_{v}")))
                    )
                    labels.append(("case", c, v))
        if "width" in parts and ds.get("width_study"):
            for w in ds["width_study"]["widths"]:
                jobs.append(
                    (MOD, "width_task", common | dict(width=int(w), out_dir=str(ex.dir / "runs" / f"width_{w}")))
                )
                labels.append(("width", w, None))
        if "sigma" in parts and ds.get("sigma_scan"):
            for s in ds["sigma_scan"]["sigmas"]:
                jobs.append(
                    (MOD, "sigma_task", common | dict(sigma=float(s), out_dir=str(ex.dir / "runs" / f"sigma_{s:g}")))
                )
                labels.append(("sigma", s, None))
        if "toy" in parts and ds.get("toy"):
            for v in ds["toy"]["variants"]:
                jobs.append((MOD, "toy_task", common | dict(variant=v)))
                labels.append(("toy", v, None))
        print(f"{len(jobs)} tasks", flush=True)
        cache = ex.dir / "tasks"
        cache.mkdir(parents=True, exist_ok=True)
        key = ["_".join(str(x) for x in lab if x is not None) for lab in labels]
        cached = {
            i: json.loads((cache / f"{k}.json").read_text(encoding="utf-8"))
            for i, k in enumerate(key)
            if a.reuse and (cache / f"{k}.json").exists()
        }
        todo = [i for i in range(len(jobs)) if i not in cached]

        def done(j, r, sec):  # every finished task is cached at once: a crash later does not lose it
            i = todo[j]
            (cache / f"{key[i]}.json").write_text(json.dumps(r), encoding="utf-8")
            print(f"[{j + 1}/{len(todo)}] {labels[i]}: {sec:.0f} s", flush=True)

        new = run_tasks([jobs[i] for i in todo], {"isolate": True, **ds.get("execution", {})}, done) if todo else []
        res = dict(cached) | dict(zip(todo, new, strict=True))
        outs = [res[i] for i in range(len(jobs))]
        if cached:
            print(f"reused {len(cached)} cached task results", flush=True)
        ex.metrics["cases"] = [o for (k, *_), o in zip(labels, outs, strict=True) if k == "case"]
        ex.metrics["width"] = [o for (k, *_), o in zip(labels, outs, strict=True) if k == "width"]
        ex.metrics["sigma"] = [o for (k, *_), o in zip(labels, outs, strict=True) if k == "sigma"]
        ex.metrics["toy"] = [o for (k, *_), o in zip(labels, outs, strict=True) if k == "toy"]
        ex.metrics["git"] = git_hash()
        ex.metrics["figures"] = figures(ex.metrics, a.tag, ex.dir, docs=not custom)
        report(ex.metrics, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def _g(v, f=".3g") -> str:
    return "—" if v is None else format(v, f)


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Step 6.5 training diagnostics — `{tag}` (auto-generated by experiments/p6_5_diagnostics.py)",
        "",
        "Truth-free quantities on a fixed point set (loss, NTK, Hessian); the error and its spectrum vs the analytic "
        "solution. NTK = residual kernel J Jᵀ; drift = ‖K_t − K_0‖/‖K_0‖; trace per point = tr(K_g)/n_g. Verification only.",
        "",
    ]
    if met.get("cases"):
        L += [
            "| case | variant | params | final rel L2 | final loss | NTK eff. rank (init → final) | trace PDE / BC per point (init) | NTK drift (final) | Hessian λmax (final) | Hessian λmin / λmax (full) | residual share in top-10 NTK directions (final) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in met["cases"]:
            n0, n1 = r["ntk"][0], r["ntk"][-1]
            tr = n0["trace_per_point"]
            bc = np.mean([v for k, v in tr.items() if k.startswith("bc")])
            hs = r.get("hessian_spectrum")
            ratio = f"{hs[0] / hs[-1]:.2g}" if hs else "—"
            L.append(
                f"| {r['case']} | {r['variant']} | {r['n_params']} | {r['rel_l2'][-1]:.3g} | {r['loss'][-1]:.3g} | "
                f"{n0['effective_rank']:.1f} → {n1['effective_rank']:.1f} | {tr['pde']:.3g} / {bc:.3g} | {n1['drift']:.3g} | "
                f"{_g((n1.get('hessian') or {}).get('max'))} | {ratio} | {n1['residual_share_top'][2]:.2f} |"
            )
        L.append("")
    if met.get("conflict"):
        L += [
            "Gradient conflict (unweighted PDE vs boundary / initial terms; cancellation = |g_pde + g_cond| / (|g_pde| + |g_cond|)):",
            "",
            "| case | variant | loss PDE / cond (final) | |g_pde| / |g_cond| (init → final) | cosine (init → final) | cancellation (init → final) |",
            "|---|---|---|---|---|---|",
        ]
        for r in met["conflict"]:
            i, f = r["init"], r["final"]
            L.append(
                f"| {r['case']} | {r['variant']} | {f['loss_pde']:.3g} / {f['loss_cond']:.3g} | {i['grad_pde']:.3g} / {i['grad_cond']:.3g} → "
                f"{f['grad_pde']:.3g} / {f['grad_cond']:.3g} | {i['cosine']:+.2f} → {f['cosine']:+.2f} | {i['cancellation']:.3f} → {f['cancellation']:.3f} |"
            )
        L.append("")
    if met.get("width"):
        L += [
            "Width study (lazy regime): "
            + "; ".join(
                f"width {w['width']} ({w['n_params']} params): final NTK drift {w['drift'][-1]:.3g}, relative weight change {w['param_change']:.3g}"
                for w in met["width"]
            )
            + ".",
            "",
        ]
    if met.get("sigma"):
        L += [
            "| Fourier σ | output-NTK alignment with a coarse FD solution (init, truth-free) | rel L2 after training | validation loss |",
            "|---|---|---|---|",
        ]
        for s in met["sigma"]:
            L.append(f"| {s['sigma']:g} | {_g(s['alignment_fd'])} | {_g(s.get('rel_l2'))} | {_g(s.get('val_loss'))} |")
        L.append("")
    if met.get("toy"):
        L += ["Toy spectral bias (residual amplitude of each frequency; 1 = not learnt): epochs to reach < 0.1:", ""]
        for t in met["toy"]:
            amp = np.asarray(t["amplitude"])
            first = []
            for j, k in enumerate(t["frequencies"]):
                hit = np.flatnonzero(amp[:, j] < 0.1)
                first.append(f"k={k}: " + (str(t["epochs"][hit[0]]) if hit.size else f"never (final {amp[-1, j]:.2f})"))
            L.append(f"- {t['variant']}: " + ", ".join(first))
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    paths = []
    if met.get("cases"):
        paths += save_fig(F.spectral_surfaces(met["cases"]), f"p6_5_spectral3d_{tag}", odir, docs=docs)
        paths += save_fig(F.error_spectra(met["cases"]), f"p6_5_spectra_{tag}", odir, docs=docs)
        paths += save_fig(F.ntk_spectra(met["cases"]), f"p6_5_ntk_spectra_{tag}", odir, docs=docs)
        paths += save_fig(F.ntk_waterfall(met["cases"]), f"p6_5_ntk_waterfall3d_{tag}", odir, docs=docs)
        paths += save_fig(F.ntk_eigenfunctions(met["cases"]), f"p6_5_ntk_eigenfunctions_{tag}", odir, docs=docs)
        paths += save_fig(F.dynamics(met["cases"]), f"p6_5_dynamics_{tag}", odir, docs=docs)
        paths += save_fig(F.hessian(met["cases"]), f"p6_5_hessian_{tag}", odir, docs=docs)
        paths += save_fig(F.landscapes3d(met["cases"]), f"p6_5_landscape3d_{tag}", odir, docs=docs)
        paths += save_fig(F.landscapes2d(met["cases"]), f"p6_5_landscape2d_{tag}", odir, docs=docs)
    if met.get("width") or met.get("sigma"):
        paths += save_fig(
            F.width_sigma(met.get("width") or [], met.get("sigma") or []), f"p6_5_width_sigma_{tag}", odir, docs=docs
        )
    if met.get("toy"):
        paths += save_fig(F.toy(met["toy"]), f"p6_5_toy3d_{tag}", odir, docs=docs)
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
