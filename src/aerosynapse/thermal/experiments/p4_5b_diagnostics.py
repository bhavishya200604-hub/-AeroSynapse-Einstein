"""Step 4.5b diagnostics: mass conservation of the trained channel PINNs. Reloads the final networks saved by a
p4_5b_channel run (runs/*/checkpoint.pt, phase 'done') with the run's own config and design, and evaluates the PINN
flow rate Q = int u dy at the inlet and outlet at every design point (aerosynapse.thermal.pinn.channel2d.flow_rates). The exact flow
conserves mass, so Q_out / Q_in = 1; a ratio well below 1 means the network has lost mass along the channel.

    python experiments/p4_5b_diagnostics.py --tags smoke psi_check

--residuals: where the PDE residuals of the trained D_obstacle networks sit. On a uniform grid of fluid points
(--n points per unit length) at the first design point, the raw (unnormalised) momentum and energy residuals are
binned by the distance to the nearest obstacle corner; per bin: number of points, mean square of each residual and its
share of the total sum of squares (results/p4_5b_residuals).

    python experiments/p4_5b_diagnostics.py --residuals --tags smoke hard_psi

Outputs (both modes): metrics.json, log.csv (one row per record / bin), report.md (readable tables) and figures
(results/<exp>/ and docs/figures/): p4_5b_flow_rates.png (Q_out/Q_in per test point, one panel per case) and
p4_5b_residuals_vs_corner.png (mean-square momentum residual against corner distance, one line per run).
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from aerosynapse.thermal.experiments.p4_5b_channel import variant_spec
from aerosynapse.thermal.data.design import Dim
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.style import apply_style, save_fig

COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#000000"]


def load_net(s: dict, prob, ck) -> torch.nn.Module:
    net = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(torch.float64)
    st = torch.load(ck, map_location="cpu", weights_only=False)
    assert st["phase"] == "done", f"{ck}: training not finished ({st['phase']})"
    net.load_state_dict(st["model"])
    return net.eval()


BINS = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, np.inf)


def residual_bins(tag: str, n: int) -> list[dict]:
    rdir = ROOT / "results" / f"p4_5b_channel_{tag}"
    met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
    spec = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    cs, case = spec["channel_study"], "D_obstacle"
    d = met["designs"][case]
    dims = [Dim(nm, lo, hi, sc) for nm, lo, hi, sc in d["dims"]]
    pt, seed = d["points"][0], cs["seeds"][0]
    out = []
    for v in cs["variants"]:
        s = variant_spec(spec, v)
        form = SP.deep_merge(s["formulation"], cs["formulation"])
        if v["kind"] == "parametric":
            prob = CH.build_channel(case, s["problems2d"][case], form, dims, d["fixed"], d["u_lim"])
            run, vals = f"{case}_{v['name']}_s{seed}", pt["values"]
        else:
            prob = CH.build_channel(case, s["problems2d"][case], form, [], pt["values"], 1.0)
            run, vals = f"{case}_{v['name']}_p{pt['index']}_s{seed}", None
        net = load_net(s, prob, rdir / "runs" / run / "checkpoint.pt")
        x = (np.arange(int(n * prob.Lx)) + 0.5) / n
        y = (np.arange(int(n * prob.Ly)) + 0.5) / n
        X, Y = (a.ravel() for a in np.meshgrid(x, y))
        keep = ~prob.inside_obstacle(X, Y)
        X, Y = X[keep], Y[keep]
        o = prob.obs
        corners = [(o["x0"], o["y0"]), (o["x0"], o["y1"]), (o["x1"], o["y0"]), (o["x1"], o["y1"])]
        dist = np.min([np.hypot(X - cx, Y - cy) for cx, cy in corners], 0)
        p0 = next(net.parameters())
        sq = {k: np.empty(len(X)) for k in ("mom_x", "mom_y", "energy")}
        for i in range(0, len(X), 2000):
            sl = slice(i, i + 2000)
            r = prob.residuals(prob.fields(net, prob.inputs(np.stack([X[sl], Y[sl]], 1), vals, p0), order=2))
            for k in sq:
                sq[k][sl] = r[k].detach().numpy() ** 2
        rec = {"tag": tag, "variant": v["name"], "point": pt["index"], "n_points": int(len(X)), "bins": []}
        for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
            m = (dist >= lo) & (dist < hi)
            b = {"d_lo": lo, "d_hi": float(hi), "n": int(m.sum())}
            for k, a in sq.items():
                b[f"ms_{k}"] = float(a[m].mean()) if m.any() else float("nan")
                b[f"share_{k}"] = float(a[m].sum() / a.sum())
            rec["bins"].append(b)
            print(
                f"{tag:9s} {v['name']:10s} corner dist [{lo:.2f}, {hi:.2f}) n={b['n']:6d}  "
                + "  ".join(f"{k}: ms {b['ms_' + k]:.3g} share {b['share_' + k]:.3f}" for k in sq)
            )
        rec["ms_total"] = {k: float(a.mean()) for k, a in sq.items()}
        out.append(rec)
    return out


def run_tag(tag: str) -> list[dict]:
    rdir = ROOT / "results" / f"p4_5b_channel_{tag}"
    met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
    spec = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    cs = spec["channel_study"]
    out = []
    for case, d in met["designs"].items():
        dims = [Dim(n, lo, hi, sc) for n, lo, hi, sc in d["dims"]]
        for v in cs["variants"]:
            s = variant_spec(spec, v)
            form = SP.deep_merge(s["formulation"], cs["formulation"])
            for seed in cs["seeds"]:
                for pt in d["points"]:
                    if v["kind"] == "parametric":
                        prob = CH.build_channel(case, s["problems2d"][case], form, dims, d["fixed"], d["u_lim"])
                        run, vals = f"{case}_{v['name']}_s{seed}", pt["values"]
                    else:
                        prob = CH.build_channel(case, s["problems2d"][case], form, [], pt["values"], 1.0)
                        run, vals = f"{case}_{v['name']}_p{pt['index']}_s{seed}", None
                    net = load_net(s, prob, rdir / "runs" / run / "checkpoint.pt")
                    rec = {"tag": tag, "case": case, "variant": v["name"], "seed": seed, "point": pt["index"]}
                    rec.update({"split": pt["split"], **CH.flow_rates(prob, net, vals)})
                    out.append(rec)
                    print(
                        f"{tag:10s} {case:11s} {v['name']:10s} p{pt['index']:<3d} {pt['split']:6s} "
                        f"Q_in {rec['flux_inlet']:.4f}  Q_out {rec['flux_outlet']:.4f}  ratio {rec['flux_ratio']:.4f}"
                    )
    return out


def tag_color(tag: str, recs: list[dict]) -> str:
    """One colour per run tag (variants are told apart by marker and line style)."""
    tags = list(dict.fromkeys(r["tag"] for r in recs))
    return COLORS[tags.index(tag) % len(COLORS)]


def flow_report(recs: list[dict], ex: Experiment) -> list[str]:
    """report.md table + figure of Q_out / Q_in per test point."""
    for r in recs:
        ex.log(
            **{k: r[k] for k in ("tag", "case", "variant", "point", "split", "flux_inlet", "flux_outlet", "flux_ratio")}
        )
    runs = list(dict.fromkeys((r["tag"], r["variant"]) for r in recs))
    cases = list(dict.fromkeys(r["case"] for r in recs))
    L = [
        "# Step 4.5b diagnostics: PINN flow rate Q_out / Q_in (exact 1)",
        "",
        "Auto-generated by experiments/p4_5b_diagnostics.py from the saved networks; Q = int u dy (64-point midpoint rule).",
    ]
    for c in cases:
        pts = sorted({r["point"] for r in recs if r["case"] == c})
        L += ["", f"## {c}", "", "| run (tag / variant) | " + " | ".join(f"p{p}" for p in pts) + " |"]
        L.append("|---|" + "---|" * len(pts))
        for tg, v in runs:
            row = {r["point"]: r["flux_ratio"] for r in recs if r["case"] == c and r["tag"] == tg and r["variant"] == v}
            if row:
                L.append(f"| {tg} / {v} | " + " | ".join(f"{row[p]:.4f}" if p in row else "" for p in pts) + " |")
    (ex.dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    apply_style()
    fig, axs = plt.subplots(1, len(cases), figsize=(5.5 * len(cases), 3.8), squeeze=False)
    for ax, c in zip(axs[0], cases, strict=True):
        for i, (tg, v) in enumerate(runs):
            rs = [r for r in recs if r["case"] == c and r["tag"] == tg and r["variant"] == v]
            if rs:
                ax.plot(
                    [r["point"] for r in rs],
                    [r["flux_ratio"] for r in rs],
                    "o-" if v == "parametric" else "s--",
                    color=tag_color(tg, recs),
                    label=f"{tg} / {v}",
                )
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set(xlabel="test point", ylabel="$Q_{out}/Q_{in}$ (exact 1)", title=c)
        ax.set_xticks(sorted({r["point"] for r in recs if r["case"] == c}))
        ax.legend(fontsize=6)
    return [str(x) for x in save_fig(fig, "p4_5b_flow_rates", ex.dir)]


def residual_report(recs: list[dict], ex: Experiment) -> list[str]:
    """report.md table + figure of the mean-square momentum residual against corner distance."""
    for r in recs:
        for b in r["bins"]:
            ex.log(tag=r["tag"], variant=r["variant"], point=r["point"], **b)
    L = [
        "# Step 4.5b diagnostics: D_obstacle PDE residuals by distance to the nearest obstacle corner",
        "",
        "Auto-generated by experiments/p4_5b_diagnostics.py --residuals: raw (unnormalised) residuals on a uniform grid of",
        "fluid points at the first test point; ms = mean square in the bin, share = fraction of the total sum of squares.",
    ]
    for key in ("mom_x", "mom_y", "energy"):
        b0 = recs[0]["bins"]
        hdr = [f"[{b['d_lo']:g}, {b['d_hi']:g})" for b in b0]
        L += [
            "",
            f"## {key}: ms (share)",
            "",
            "| run (tag / variant) | " + " | ".join(hdr) + " |",
            "|---|" + "---|" * len(hdr),
        ]
        for r in recs:
            L.append(
                f"| {r['tag']} / {r['variant']} | "
                + " | ".join(f"{b['ms_' + key]:.3g} ({b['share_' + key]:.2f})" for b in r["bins"])
                + " |"
            )
    (ex.dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key in zip(axs, ("mom_x", "energy"), strict=True):
        for i, r in enumerate(recs):
            mids = [0.5 * (b["d_lo"] + min(b["d_hi"], 1.0)) for b in r["bins"]]
            ax.loglog(
                mids,
                [b["ms_" + key] for b in r["bins"]],
                "o-" if r["variant"] == "parametric" else "s--",
                color=tag_color(r["tag"], recs),
                label=f"{r['tag']} / {r['variant']}",
            )
        ax.set(
            xlabel="distance to nearest obstacle corner (bin centre)",
            ylabel=f"mean square {key} residual",
            title=f"D_obstacle: {key}",
        )
        ax.legend(fontsize=6)
    return [str(x) for x in save_fig(fig, "p4_5b_residuals_vs_corner", ex.dir)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=["smoke", "psi_check"])
    ap.add_argument("--residuals", action="store_true", help="D_obstacle residuals binned by distance to the corners")
    ap.add_argument("--n", type=int, default=64, help="--residuals: grid points per unit length")
    a = ap.parse_args()
    if a.residuals:
        recs = [r for t in a.tags for r in residual_bins(t, a.n)]
        with Experiment("p4_5b_residuals", {"tags": a.tags, "n": a.n}) as ex:
            ex.metrics.update({"residual_bins": recs, "git": git_hash()})
            ex.metrics["figures"] = residual_report(recs, ex)
            print(f"wrote {ex.dir / 'metrics.json'}")
        return
    recs = [r for t in a.tags for r in run_tag(t)]
    with Experiment("p4_5b_diagnostics", {"tags": a.tags}) as ex:
        ex.metrics.update({"flow_rates": recs, "git": git_hash()})
        ex.metrics["figures"] = flow_report(recs, ex)
        print(f"wrote {ex.dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
