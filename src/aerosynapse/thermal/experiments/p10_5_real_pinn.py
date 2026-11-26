"""Phase 10.5: physics-informed neural networks TRAINED ON REAL measured soil temperatures (D149).

    python -m aerosynapse.thermal.experiments.p10_5_real_pinn [--config FILE] [--set key=value] [--quick] [--tag TAG]

For every station and leave-one-sensor-out split of configs/real/pinn.yaml: the PINN of src/inverse/realpinn.py is
trained with every seed, and the verified classical Crank-Nicolson estimator is fitted on the same sensors; both
predict the held-out sensor, scored after the same spin-up. Reported: alpha (PINN per seed, classical), held-out RMSE,
the paired difference PINN - classical with a bootstrap 95 % CI over seeds (n = number of seeds; no significance is
claimed from five seeds), training wall time.

Writes results/p10_5_real_pinn_<tag>/metrics.json, docs/tables/phase10_5_real_pinn.md and figures (default tag:
docs/figures/real/p10_5_*; other tags: the results folder).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from aerosynapse.thermal.data.spec import coerce_numbers, deep_merge, parse_set
from aerosynapse.thermal.inverse import realdata as R
from aerosynapse.thermal.inverse import realpinn as P
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.style import DATA, PINN, SOLVER, apply_style, save_fig

DEFAULT = ROOT / "configs" / "real" / "pinn.yaml"


def load_cfg(files: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for f in files:
        cfg = deep_merge(cfg, coerce_numbers(yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}))
    return deep_merge(cfg, parse_set(sets))


def station_window(cfg: dict, station: str) -> dict:
    d, m = cfg["data"], cfg["model"]
    s = R.load_uscrn(ROOT / d["folder"] / str(d["year"]) / f"CRNH0203-{d['year']}-{station}.txt")
    cols = [i for i, z in enumerate(s.z) if m["top_m"] - 1e-9 <= z <= m["bottom_m"] + 1e-9]
    win = R.window(s, R.iso_to_seconds(s, d["start"]), float(d["days"]), int(d["max_gap_h"]), cols=cols)
    if np.isnan(win["T"]).any():
        raise ValueError(f"{station}: gaps longer than {d['max_gap_h']} h in the window")
    return win


def boot_ci(x: np.ndarray, n: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run(cfg: dict) -> tuple[dict, dict]:
    m, soil = cfg["model"], yaml.safe_load((ROOT / "configs/real/soil.yaml").read_text(encoding="utf-8"))
    fcfg = soil["fit"]
    spin = float(m["spinup_h"]) * 3600
    rows, series = [], {}
    for st in cfg["data"]["stations"]:
        win = station_window(cfg, st)
        for sp in cfg["splits"]:
            hz, inner = float(sp["holdout"]), [float(v) for v in sp["train_inner"]]
            k = int(np.argmin(np.abs(win["z"] - hz)))
            obs = win["T"][:, k]
            t0 = time.perf_counter()
            cl = P.classical_loo(win, inner, hz, m, fcfg)
            cl_s = time.perf_counter() - t0
            cl_rmse = P.rmse_after(win["t"], obs, cl["prediction"], spin)
            runs = []
            for seed in cfg["seeds"]:
                r = P.train(
                    win, [m["top_m"], *inner, m["bottom_m"]], m["top_m"], m["bottom_m"], seed=int(seed), **cfg["pinn"]
                )
                tr = [int(np.argmin(np.abs(win["z"] - z))) for z in [m["top_m"], *inner, m["bottom_m"]]]
                runs.append(
                    {
                        "seed": int(seed),
                        "alpha": r["alpha"],
                        "rmse_holdout": P.rmse_after(win["t"], obs, r["prediction"][:, k], spin),
                        "rmse_train_sensors": [
                            P.rmse_after(win["t"], win["T"][:, j], r["prediction"][:, j], spin) for j in tr
                        ],
                        "final_loss": r["history"][-1],
                        "wall_s": r["wall_s"],
                        "n_parameters": r["n_parameters"],
                    }
                )
                print(
                    f"{st} holdout {hz:g} m seed {seed}: alpha {r['alpha']:.3e} (classical {cl['alpha']:.3e}), "
                    f"RMSE {runs[-1]['rmse_holdout']:.3f} K (classical {cl_rmse:.3f}) {r['wall_s']:.0f} s",
                    flush=True,
                )
                if int(seed) == int(cfg["seeds"][0]):
                    series[(st, hz)] = {
                        "t": win["t"],
                        "obs": obs,
                        "pinn": r["prediction"][:, k],
                        "classical": cl["prediction"],
                        "history": r["history"],
                    }
            a = np.array([x["alpha"] for x in runs])
            e = np.array([x["rmse_holdout"] for x in runs])
            d = e - cl_rmse
            rows.append(
                {
                    "station": st,
                    "holdout_m": hz,
                    "train_inner_m": inner,
                    "classical": {
                        "alpha": cl["alpha"],
                        "converged": cl["converged"],
                        "rmse_holdout": cl_rmse,
                        "wall_s": cl_s,
                    },
                    "pinn_runs": runs,
                    "pinn_alpha": {
                        "mean": float(a.mean()),
                        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                        "min": float(a.min()),
                        "max": float(a.max()),
                    },
                    "pinn_rmse_holdout": {
                        "mean": float(e.mean()),
                        "sd": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
                        "min": float(e.min()),
                        "max": float(e.max()),
                    },
                    "alpha_ratio_pinn_over_classical": float(a.mean() / cl["alpha"]),
                    "rmse_diff_pinn_minus_classical": {
                        "mean": float(d.mean()),
                        "ci95": boot_ci(d, int(cfg["bootstrap"])) if len(d) > 1 else [float(d[0])] * 2,
                        "seeds_pinn_better": int((d < 0).sum()),
                        "n": len(d),
                    },
                    "pinn_wall_s_median": float(np.median([x["wall_s"] for x in runs])),
                }
            )
    return {"rows": rows}, series


def figures(series: dict, rows: list[dict], tag: str, odir: Path) -> list[str]:
    out = ROOT / "docs" / "figures" / "real" if tag == "default" else odir
    apply_style()
    names = []
    st0 = rows[0]["station"]
    # 1 held-out sensor of the default station: measured, PINN (first seed), classical
    keys = [k for k in series if k[0] == st0]
    fig, axs = plt.subplots(len(keys), 1, figsize=(7.2, 2.3 * len(keys)), sharex=True, squeeze=False)
    for ax, key in zip(axs[:, 0], keys, strict=True):
        s = series[key]
        days = (s["t"] - s["t"][0]) / R.DAY
        ax.plot(days, s["obs"], color=DATA, lw=1.4, label="measured (never shown to either method)")
        ax.plot(days, s["pinn"], color=PINN, lw=1.0, label="PINN trained on the other sensors")
        ax.plot(days, s["classical"], color=SOLVER, lw=1.0, ls="--", label="classical CN estimator, same sensors")
        ax.axvspan(0, 1, color="0.85", lw=0)
        ax.set(ylabel=f"T at {key[1] * 100:.0f} cm [°C]")
    axs[0, 0].set_title(f"Held-out sensor, NOAA USCRN {st0.replace('_', ' ')} (grey: spin-up, not scored)")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    axs[-1, 0].set(xlabel="days from the window start")
    names.append(f"p10_5_holdout_{tag}")
    save_fig(fig, names[-1], out, docs=False)
    # 2 alpha and held-out RMSE per station and split: seeds (dots) vs classical (bar)
    names.append(fig_alpha_rmse(rows, tag, out))
    # 3 training history of the default case (first seed)
    h = series[keys[0]]["history"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    st = [x["step"] for x in h]
    axs[0].semilogy(st, [x["data"] for x in h], color=DATA, label="data misfit")
    axs[0].semilogy(st, [x["pde"] for x in h], color=PINN, label="heat-equation residual")
    axs[0].set(xlabel="step", ylabel="loss (scaled)", title="training (warm-up, Adam, L-BFGS)")
    axs[0].legend(fontsize=7)
    axs[1].plot(st, [x["alpha"] * 1e7 for x in h], color=PINN)
    axs[1].axhline(rows[0]["classical"]["alpha"] * 1e7, color=SOLVER, ls="--", label="classical estimate")
    axs[1].set(xlabel="step", ylabel="alpha [1e-7 m²/s]", title="diffusivity during training")
    axs[1].legend(fontsize=7)
    names.append(f"p10_5_training_{tag}")
    save_fig(fig, names[-1], out, docs=False)
    return names


def fig_alpha_rmse(rows: list[dict], tag: str, out: Path) -> str:
    """Alpha and held-out RMSE per station and held-out depth: PINN seeds (dots) vs the classical estimator (bars);
    drawn from the recorded rows only (python -m aerosynapse.thermal.experiments.p10_5_real_pinn --replot redraws it from metrics.json)."""
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # station code (state) and depth: short labels that do not overlap
    labels = [f"{r['station'].split('_')[0]}\n{r['holdout_m'] * 100:.0f} cm" for r in rows]
    xs = np.arange(len(rows))
    for i, r in enumerate(rows):
        axs[0].bar(i, r["classical"]["alpha"] * 1e7, color=SOLVER, alpha=0.35, width=0.6)
        axs[0].plot([i] * len(r["pinn_runs"]), [x["alpha"] * 1e7 for x in r["pinn_runs"]], "o", ms=3, color=PINN)
        axs[1].bar(i, r["classical"]["rmse_holdout"], color=SOLVER, alpha=0.35, width=0.6)
        axs[1].plot([i] * len(r["pinn_runs"]), [x["rmse_holdout"] for x in r["pinn_runs"]], "o", ms=3, color=PINN)
    axs[0].set(ylabel="alpha [1e-7 m²/s]", title="diffusivity: PINN seeds (dots), classical (bar)")
    axs[1].set(ylabel="held-out RMSE [K]", title="held-out sensor error")
    for ax in axs:
        ax.set_xticks(xs, labels, fontsize=7)
    name = f"p10_5_alpha_rmse_{tag}"
    save_fig(fig, name, out, docs=False)
    return name


def table(met: dict, cfg: dict, path: Path) -> None:
    L = [
        "# Phase 10.5 — PINN trained on real soil temperatures vs the classical estimator",
        "",
        f"Generated by `python -m aerosynapse.thermal.experiments.p10_5_real_pinn` (git {met['git']}). Window {cfg['data']['start']}, "
        f"{cfg['data']['days']} days; seeds {cfg['seeds']}; leave-one-sensor-out; both scored after "
        f"{cfg['model']['spinup_h']} h spin-up. VALIDATION against measured data, n = {len(cfg['seeds'])} seeds per "
        "case: the CI is a bootstrap over seeds, no significance is claimed.",
        "",
        "| station | held out | alpha classical | alpha PINN mean (min-max) | RMSE classical [K] | RMSE PINN mean ± sd [K] "
        "| PINN - classical [K] (95 % CI) | seeds PINN better | PINN wall [s] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in met["rows"]:
        a, e, d = r["pinn_alpha"], r["pinn_rmse_holdout"], r["rmse_diff_pinn_minus_classical"]
        L.append(
            f"| {r['station'].replace('_', ' ')} | {r['holdout_m'] * 100:.0f} cm | {r['classical']['alpha']:.3g} | "
            f"{a['mean']:.3g} ({a['min']:.3g}-{a['max']:.3g}) | {r['classical']['rmse_holdout']:.3f} | "
            f"{e['mean']:.3f} ± {e['sd']:.3f} | {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | "
            f"{d['seeds_pinn_better']} / {d['n']} | {r['pinn_wall_s_median']:.0f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--quick", action="store_true", help="one station, one split, two seeds, short training")
    ap.add_argument("--replot", action="store_true", help="redraw the alpha / RMSE figure from the saved metrics.json")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.sets)
    if a.replot:
        import json

        met = json.loads((ROOT / "results" / f"p10_5_real_pinn_{a.tag}" / "metrics.json").read_text(encoding="utf-8"))
        out = (
            ROOT / "docs" / "figures" / "real" if a.tag == "default" else ROOT / "results" / f"p10_5_real_pinn_{a.tag}"
        )
        print("redrawn:", fig_alpha_rmse(met["rows"], a.tag, out))
        return
    if a.quick:
        cfg = deep_merge(
            cfg,
            {
                "data": {"stations": cfg["data"]["stations"][:1]},
                "splits": cfg["splits"][:1],
                "seeds": [0, 1],
                "pinn": {"adam_steps": 800, "lbfgs_steps": 50, "warmup": 200},
            },
        )
        a.tag = a.tag if a.tag != "default" else "quick"
    with Experiment(f"p10_5_real_pinn_{a.tag}", cfg) as ex:
        met, series = run(cfg)
        ex.metrics.update(met | {"git": git_hash()})
        ex.metrics["figures"] = figures(series, met["rows"], a.tag, ex.dir)
        if a.tag == "default":
            table(ex.metrics, cfg, ROOT / "docs" / "tables" / "phase10_5_real_pinn.md")
    for r in met["rows"]:
        d = r["rmse_diff_pinn_minus_classical"]
        print(
            f"{r['station']} {r['holdout_m'] * 100:.0f} cm: alpha PINN {r['pinn_alpha']['mean']:.3e} vs classical "
            f"{r['classical']['alpha']:.3e}; RMSE PINN {r['pinn_rmse_holdout']['mean']:.3f} vs "
            f"{r['classical']['rmse_holdout']:.3f} K; diff {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
        )


if __name__ == "__main__":
    main()
