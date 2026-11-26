"""Step 6B.2b gap run G1 (paper/PREREG_G1_G3.md): statistics of the Step 4.7 ablation at 5 seeds (no training).

Per (configuration, seed) the mean over the interpolation test points of the worst-field relative L2 (primary; R4.7a),
pooled from the Step 4.7 run (seeds 0-2) and the G1 run (seeds 3-4; same config, bit-reproducible training, R6.7b).
Each leave-one-out configuration is compared with `full`, each add-one-in configuration with `base`, paired by seed:
paired t-test on the log ratio (primary), Holm over the 12 comparisons, bootstrap 95 % CI (10 000, seed 0), exact
Wilcoxon p reported. ESTABLISHED if Holm p < 0.05 AND the geometric-mean ratio is beyond 2x. The extrapolation error and
the heat-flux error are secondary metrics with the same test (own Holm family each).

    python -m aerosynapse.thermal.experiments.p6b_g1_stats [--config my.yaml] [--set key=value] [--tag name]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from aerosynapse.thermal.data.spec import coerce_numbers, deep_merge, parse_set
from aerosynapse.thermal.utils import stats as S
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.style import save_fig

DEFAULT = ROOT / "configs" / "pinn" / "gap_g1.yaml"


def load_cfg(files: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for f in files:
        cfg = deep_merge(cfg, coerce_numbers(yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}))
    return deep_merge(cfg, parse_set(sets))


def table(runs: list[str], split: str, field: str) -> tuple[dict, dict]:
    """{config: {seed: mean over the split's points}} and {config: design}."""
    vals: dict[str, dict[int, list[float]]] = {}
    design: dict[str, str] = {}
    for run in runs:
        met = json.loads((ROOT / run / "metrics.json").read_text(encoding="utf-8"))
        for c in met["configurations"]:
            design[c["name"]] = c["design"]
        for r in met["records"]:
            if r.get("split") == split and r.get(field) is not None:
                vals.setdefault(r["config"], {}).setdefault(int(r["seed"]), []).append(float(r[field]))
    return {c: {s: float(np.mean(v)) for s, v in d.items()} for c, d in vals.items()}, design


def compare(tab: dict, design: dict, cfg: dict) -> list[dict]:
    an = cfg["analysis"]
    refs = cfg["references"]
    seeds = [int(s) for s in cfg["seeds"]]
    out = []
    for c, d in design.items():
        ref = refs[d]
        if c == ref:
            continue
        common = [s for s in seeds if s in tab.get(c, {}) and s in tab.get(ref, {})]
        t = S.paired_log_ratio(
            np.array([tab[c][s] for s in common]),
            np.array([tab[ref][s] for s in common]),
            int(an["n_boot"]),
            int(an["boot_seed"]),
        )
        out.append({"config": c, "design": d, "reference": ref, "seeds": common, **t})
    for e, pa in zip(out, S.holm([e["p_t"] for e in out]), strict=True):
        e["p_holm"] = pa
        big = e["gm_ratio"] > float(an["ratio_min_effect"]) or e["gm_ratio"] < 1 / float(an["ratio_min_effect"])
        e["established"] = bool(pa < float(an["alpha"]) and big)
        e["verdict"] = "ESTABLISHED" if e["established"] else "NOT ESTABLISHED"
    return out


def figure(res: dict, tag: str, odir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    from aerosynapse.thermal.viz.style import apply_style

    apply_style()
    prim = res["primary"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    y = np.arange(len(prim))
    for i, e in enumerate(prim):
        col = "#D55E00" if e["established"] else "#0072B2"
        lo, hi = e["gm_ratio_ci95"]
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot(e["gm_ratio"], i, "o", color=col)
        ax.plot(np.exp(e["log_ratios"]), np.full(len(e["log_ratios"]), i), "|", color="0.4", ms=7)
    ax.axvline(1, color="0.5", lw=0.8)
    for v in (0.5, 2.0):
        ax.axvline(v, color="0.7", lw=0.6, ls=":")
    ax.set_yticks(y, [f"{e['config']}  vs {e['reference']}" for e in prim], fontsize=8)
    ax.set_xscale("log")
    ticks = [0.1, 0.2, 0.5, 1, 2, 5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.plot([], [], "o-", color="#D55E00", label="ESTABLISHED (Holm p < 0.05 and beyond 2x)")
    ax.plot([], [], "o-", color="#0072B2", label="not established")
    ax.plot([], [], "|", color="0.4", label="single seeds")
    fig.legend(fontsize=8, loc="outside lower center", ncols=3)
    ax.set_xlabel("error ratio (configuration / reference): geometric mean, 95 % bootstrap CI, seeds")
    ax.invert_yaxis()
    paths = save_fig(fig, f"p6b_g1_ablation_{tag}", odir, docs=False)
    plt.close(fig)
    return [str(p) for p in paths]


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Gap run G1 — Step 4.7 ablation at 5 seeds — `{tag}` (auto-generated by experiments/p6b_g1_stats.py)",
        "",
        "Pre-registered in paper/PREREG_G1_G3.md. Ratio = configuration / reference error (geometric mean over seeds); "
        "< 1: the configuration is MORE accurate than its reference. Verification only (errors vs the FV solver).",
        "",
    ]
    for k, res in met["results"].items():
        L += [
            f"## {k}",
            "",
            "| configuration | reference | seeds | GM ratio [95 % CI] | paired t p | Holm p | Wilcoxon p | verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for e in res:
            L.append(
                f"| {e['config']} | {e['reference']} | {len(e['seeds'])} | {e['gm_ratio']:.3g} "
                f"[{e['gm_ratio_ci95'][0]:.3g}, {e['gm_ratio_ci95'][1]:.3g}] | {e['p_t']:.3g} | {e['p_holm']:.3g} | "
                f"{e['p_wilcoxon']:.3g} | {e['verdict']} |"
            )
        L.append("")
    L += ["## Per-seed primary metric (mean worst-field rel L2, interpolation points)", ""]
    tab = met["tables"]["primary"]
    seeds = sorted({int(s) for d in tab.values() for s in d})
    L += ["| configuration | " + " | ".join(f"s{s}" for s in seeds) + " |", "|---" * (len(seeds) + 1) + "|"]
    for c, d in tab.items():
        L.append(
            f"| {c} | "
            + " | ".join(f"{d[str(s)] if str(s) in d else d.get(s, float('nan')):.3g}" for s in seeds)
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.sets)
    with Experiment(f"p6b_g1_stats_{a.tag}", cfg | {"seed": 0}) as ex:
        m = cfg["metrics"]
        specs = {"primary": m["primary"], **m["secondary"]}
        tables, results = {}, {}
        for k, sp in specs.items():
            tab, design = table(cfg["runs"], sp["split"], sp["field"])
            tables[k] = tab
            results[k] = compare(tab, design, cfg)
        ex.metrics["tables"] = tables
        ex.metrics["results"] = results
        ex.metrics["git"] = git_hash()
        ex.metrics["figures"] = figure(results, a.tag, ex.dir)
        # JSON round trip turns the seed keys into strings: report from the saved form
        report(json.loads(json.dumps(ex.metrics)), ex.dir / "g1.md", a.tag)
    print(f"wrote {ex.dir / 'g1.md'}")


if __name__ == "__main__":
    main()
