"""Step 4.2 diagnostics (saved metrics for two observations of the formulation study).

1. Cavity reference convergence: Newton FV on n, 2n, 4n cells; relative coarse-fine differences per field and their
   ratios (a ratio of 4 = second order in the asymptotic range).
2. Divergence of the hard-constrained primitive cavity run: where is |div u| large, and how large is the x-derivative
   of the transfinite boundary interpolant G_u there? (hypothesis: the steep lid profile enters the interior through G_u,
   and continuity is only a loss term in the primitive formulation).

    python experiments/p4_2_diagnostics.py --tag smoke            # uses results/p4_2_formulations_smoke
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from aerosynapse.thermal.data.generators import restrict_cell_average
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.ref2d import solve_steady
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.style import SEQ_CMAP, apply_style, save_fig


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def reference_convergence(spec: dict, grids: list[int]) -> dict:
    pc = spec["problems2d"]["C_cavity"]
    par = {k: float(v) for k, v in pc["parameters"].items()}
    cond = P2.cavity_conditions(pc)
    sols = [solve_steady(par, cond, n, float(pc["reference"]["tol"])) for n in grids]
    out: dict = {"grids": grids, "diff": {}, "ratio": {}, "nu_top": [s["nu_top"] for s in sols]}
    for k in ("u", "v", "theta", "p"):
        d = []
        for c, f in zip(sols[:-1], sols[1:], strict=True):
            r = restrict_cell_average(f[k], 2)
            d.append(rel(c[k], r - r.mean() if k == "p" else r))
        out["diff"][k] = d
        out["ratio"][k] = [d[i] / d[i + 1] for i in range(len(d) - 1)]
    nu = out["nu_top"]
    out["diff"]["nu_top"] = [abs(nu[i] - nu[i + 1]) / abs(nu[i + 1]) for i in range(len(nu) - 1)]
    return out


def divergence_map(spec: dict, rdir, variant: str, n: int) -> dict:
    case = "C_cavity"
    entry = next(e for e in spec["formulation_study"]["variants"] if e["name"] == variant)
    _, form = SP.formulation_variant(spec, entry)
    ref = dict(np.load(rdir / "reference_C_cavity.npz"))
    ref["nu_top"] = float("nan")
    prob = P2.build_problem2d(case, spec, form, ref)
    net = build_network(spec["network"], list(prob.lo), list(prob.hi), prob.out_dim).double()
    ck = torch.load(rdir / "runs" / f"{case}_{variant}_s0" / "checkpoint.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    pts = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1))
    d = P2._batched(prob, net, pts, 1, ("u_x", "v_y"))
    div = np.abs(d["u_x"] + d["v_y"]).reshape(X.shape)
    corner = (Y > 0.9) & ((X < 0.1) | (X > 0.9))
    res = {
        "variant": variant,
        "max_div": float(div.max()),
        "argmax_xy": [float(X.flat[div.argmax()]), float(Y.flat[div.argmax()])],
        "max_div_top_corners": float(div[corner].max()),
        "max_div_elsewhere": float(div[~corner].max()),
        "rms_div_top_corners": float(np.sqrt((div[corner] ** 2).mean())),
        "rms_div_elsewhere": float(np.sqrt((div[~corner] ** 2).mean())),
        "frac_points_div_gt_1": float((div > 1).mean()),
    }
    if prob.hard and prob.kind == "primitive":
        P = pts.clone().requires_grad_(True)
        g = torch.autograd.grad(prob.Gu(P[:, 0], P[:, 1]).sum(), P)[0][:, 0].detach().numpy().reshape(X.shape)
        res["max_abs_dGu_dx"] = float(np.abs(g).max())
        res["max_abs_dGu_dx_top_corners"] = float(np.abs(g[corner]).max())
        res["max_abs_dGu_dx_elsewhere"] = float(np.abs(g[~corner]).max())
    return res | {"_div": div, "_x": xs}


def figures(maps: list[dict], tag: str, ex: Experiment) -> list:
    """log10 |div u| of the two cavity formulations, side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m in zip(axs, maps, strict=True):
        im = ax.imshow(np.log10(m["_div"] + 1e-16), origin="lower", extent=(0, 1, 0, 1), cmap=SEQ_CMAP)
        ax.set(title=f"log10 |div u|: {m['variant']}", xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    return save_fig(fig, f"p4_2_divergence_cavity_{tag}", ex.dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--grids", default="32,64,128")
    ap.add_argument("--n", type=int, default=128, help="evaluation grid for the divergence maps")
    a = ap.parse_args()
    rdir = ROOT / "results" / f"p4_2_formulations_{a.tag}"
    spec = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    apply_style()
    with Experiment(f"p4_2_diagnostics_{a.tag}", {"source": str(rdir), "grids": a.grids, "n": a.n}) as ex:
        conv = reference_convergence(spec, [int(g) for g in a.grids.split(",")])
        print("reference convergence:", conv, flush=True)
        maps = [divergence_map(spec, rdir, v, a.n) for v in ("primitive", "prim_hard")]
        for m in maps:
            print({k: v for k, v in m.items() if not k.startswith("_")}, flush=True)
        paths = figures(maps, a.tag, ex)
        ex.metrics.update(
            {
                "reference_convergence": conv,
                "divergence": [{k: v for k, v in m.items() if not k.startswith("_")} for m in maps],
                "figures": [str(p) for p in paths],
                "git": git_hash(),
            }
        )


if __name__ == "__main__":
    main()
