"""Phase 10.1: thermal diffusivity of real soils from NOAA USCRN temperature profiles (VALIDATION against measured
data, the first in this project), with identify-before-you-fit, exact complex-step sensitivities, a convergence
pipeline and autocorrelation-robust uncertainty (src/inverse/realdata.py).

    python -m aerosynapse.thermal.experiments.p10_1_real_soil [--config my.yaml] [--set fit.start=2023-08-01] [--tag name] [--quick]

Steps (all numerics, no network is trained):
  0 provenance   the station files' SHA-256 against results/p10_fetch_default (the bytes that were downloaded)
  1 verification forward model vs the exact damped wave (observed order in space and time), the fast propagator vs
                 the step-by-step solver, complex-step vs central-difference derivatives
  2 stations     sensor coverage; stations with fewer than three usable depths get the two-depth harmonic estimate only
  3 compare      on the default window: the models of `compare` fitted on the training window and ranked on the
                 following hold-out window against a no-physics baseline (linear interpolation in depth)
  4 seasonal     weekly fits through the year (models of seasonal.models), alpha against soil moisture
                 (exploratory, not pre-registered: Spearman rho with a bootstrap interval, no confirmatory claim)
  5 robustness   convergence from poor initial guesses: plain Levenberg-Marquardt vs the strategy pipeline
Writes results/p10_1_real_soil_<tag>/metrics.json, docs/tables/phase10_1_real_data.md and figures docs/figures/real/p10_1_*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import spearmanr

from aerosynapse.thermal.data.spec import coerce_numbers, deep_merge, parse_set
from aerosynapse.thermal.inverse import realdata as R
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.style import DATA, PALETTE, PINN, REF, SOLVER, apply_style, save_fig

DEFAULT = ROOT / "configs" / "real" / "soil.yaml"


def load_cfg(files: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for f in files:
        cfg = deep_merge(cfg, coerce_numbers(yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}))
    return deep_merge(cfg, parse_set(sets))


def station_file(cfg: dict, st: str) -> Path:
    d = cfg["data"]
    return ROOT / d["folder"] / str(d["year"]) / f"CRNH0203-{d['year']}-{st}.txt"


def load_series(cfg: dict, st: str) -> R.Series:
    """A station of the NOAA product, or YOUR table (data.source = csv; the 'station' is then data.csv.name)."""
    if cfg["data"].get("source", "uscrn") == "csv":
        c = cfg["data"]["csv"]
        return R.load_table(
            ROOT / c["path"],
            c["time_col"],
            {k: float(v) for k, v in c["positions"].items()},
            c.get("time_unit", "auto"),
        )
    return R.load_uscrn(station_file(cfg, st))


def start_seconds(s: R.Series, cfg: dict) -> float:
    """fit.start as an ISO date (NOAA, date-time tables) or as hours from the first row (any table)."""
    st = cfg["fit"]["start"]
    if isinstance(st, int | float):
        return float(s.t[0]) + float(st) * 3600
    return R.iso_to_seconds(s, str(st))


def stations(cfg: dict) -> list[str]:
    return (
        [cfg["data"]["csv"].get("name", "user_table")]
        if cfg["data"].get("source") == "csv"
        else cfg["data"]["stations"]
    )


def provenance(cfg: dict) -> dict:
    rec = ROOT / "results" / "p10_fetch_default" / "metrics.json"
    known = (
        {f["path"]: f["sha256"] for f in json.loads(rec.read_text(encoding="utf-8"))["files"]} if rec.exists() else {}
    )
    out = {}
    for st in stations(cfg):
        p = ROOT / cfg["data"]["csv"]["path"] if cfg["data"].get("source") == "csv" else station_file(cfg, st)
        h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        rel = p.relative_to(ROOT).as_posix()
        out[st] = {"path": rel, "sha256": h, "matches_download_record": known.get(rel) == h if h else False}
    return out


# ---------------------------------------------------------------------------------------------------- verification
def verification(cfg: dict, quick: bool) -> dict:
    v = cfg["verification"]
    a, A, per = float(v["alpha"]), float(v["amplitude"]), float(v["period_h"]) * 3600
    dzs = v["dz_list"][:3] if quick else v["dz_list"]
    dts = v["dt_list"][:3] if quick else v["dt_list"]
    e_dz = [R.verify_wave(a, A, per, 0.05, 0.45, float(dz), 60.0) for dz in dzs]
    e_dt = [R.verify_wave(a, A, per, 0.05, 0.45, float(dzs[-1]), float(dt)) for dt in dts]

    def order(h, e):
        return float(np.polyfit(np.log(h[-2:]), np.log(e[-2:]), 1)[0])

    return {
        "dz": [float(x) for x in dzs],
        "err_dz": e_dz,
        "order_space": order(np.array(dzs, float), np.array(e_dz)),
        "dt": [float(x) for x in dts],
        "err_dt": e_dt,
        "order_time": order(np.array(dts, float), np.array(e_dt)),
    }


def derivative_checks(win: dict, mcfg: dict, fcfg: dict) -> dict:
    prob = R.conduction_problem(win, mcfg, fcfg)
    m = prob.model
    a = np.array([3e-7])
    fast_vs_step = float(np.max(np.abs(m.run(a) - m.run_stepwise(a))))
    x = np.log(a)
    Jc = prob.jacobian(x)[:, 0]
    h = 1e-6
    Jf = (prob.residual(x + h) - prob.residual(x - h)) / (2 * h)
    return {
        "fast_vs_stepwise_max_abs": fast_vs_step,
        "complex_step_vs_central_difference_rel": float(np.linalg.norm(Jc - Jf) / np.linalg.norm(Jc)),
    }


# ---------------------------------------------------------------------------------------------------- per station
def coverage(s: R.Series) -> dict:
    miss = np.isnan(s.T).mean(0)
    usable = [float(z) for z, mfr in zip(s.z, miss, strict=True) if mfr < 0.5]
    return {"missing_fraction": miss.tolist(), "usable_depths_m": usable}


def model_cfg(cfg: dict, name: str) -> dict:
    return {**cfg["fit"], **(cfg["compare"].get(name) or {})}


def summarise(r: dict) -> dict:
    """The JSON-friendly part of a fit record (no arrays)."""
    return {
        "rows": r["rows"],
        "converged": r["fit"]["converged"],
        "stationarity": r["fit"]["stationarity"],
        "at_bound": r["fit"]["at_bound"],
        "strategy_used": r["strategy_used"],
        "attempts": [{k: v for k, v in a.items() if k != "x"} for a in r["attempts"]],
        "identifiability": r["identifiability"],
        "rmse_train": r["rmse_per_sensor"],
        "residual_sd": r["covariance"]["s"],
        "hac_over_gn_sd": [
            h / g if g > 0 else None for h, g in zip(r["covariance"]["sd_hac"], r["covariance"]["sd_gn"], strict=True)
        ],
        "score_lag1_autocorrelation": r["covariance"]["lag1_autocorrelation"],
        "harmonic": r["harmonic"]["pairs"],
    }


def compare_models(s: R.Series, cfg: dict, default_station: bool = True) -> dict:
    t0 = start_seconds(s, cfg)
    days = float(cfg["fit"]["window_days"])
    gap = int(cfg["data"]["max_gap_h"])
    tr, te = R.window(s, t0, days, gap), R.window(s, t0 + days * R.DAY, days, gap)
    out = {
        "train_start": tr["stamp"][0] if tr["stamp"] else None,
        "valid_train": tr["valid_fraction"],
        "valid_test": te["valid_fraction"],
        "models": {},
        "series": {},
    }
    if tr["valid_fraction"] < 1 or te["valid_fraction"] < 1:
        out["skipped"] = "gaps longer than data.max_gap_h in the training or hold-out window"
        return out
    names = list(cfg["compare"])
    if not default_station:
        names = [n for n in names if n in cfg.get("compare_other_stations", names)]
    for name in names:
        fc = model_cfg(cfg, name)
        tic = time.time()
        r = R.fit_conduction(tr, cfg["model"], fc)
        h = R.holdout(tr, te, cfg["model"], fc, r["x"])
        rec = summarise(r) | {
            "rmse_holdout": h["rmse_model"],
            "rmse_baseline_holdout": h["rmse_linear_baseline"],
            "skill_holdout": h["skill"],
            "wall_s": round(time.time() - tic, 1),
        }
        if not rec["converged"] and rec["at_bound"]:
            # diagnosis, not a result: where is the optimum if the plausibility bounds are relaxed? (depth bound x3,
            # alpha bounds x10 on each side). A model that converges only outside its plausible range is rejected.
            lo, hi = (float(v) for v in fc.get("alpha_bounds", [1e-8, 1e-5]))
            relaxed = {
                **fc,
                "alpha_bounds": [lo / 10, hi * 10],
                "depth_bound_m": 3 * float(fc.get("depth_bound_m", 0.03)),
            }
            rr = R.fit_conduction(tr, cfg["model"], relaxed)
            rec["relaxed_bounds_diagnostic"] = {
                "bounds": {"alpha": relaxed["alpha_bounds"], "depth_m": relaxed["depth_bound_m"]},
                "converged": rr["fit"]["converged"],
                "at_bound": rr["fit"]["at_bound"],
                "rows": rr["rows"],
                "rmse_train": rr["rmse_per_sensor"],
            }
        out["models"][name] = rec
        out["series"][name] = {"train_pred": r["prediction"], "test_pred": h["prediction"]}
        print(
            f"   {name:24s} conv {rec['converged']!s:5} holdout rmse {np.round(h['rmse_model'], 3)} "
            f"({rec['wall_s']} s)",
            flush=True,
        )
    out["series"]["train"] = {"t": tr["t"], "T": tr["T"], "z": tr["z"]}
    out["series"]["test"] = {"t": te["t"], "T": te["T"], "baseline": h["baseline"]}
    ok = {k: v for k, v in out["models"].items() if v["converged"]}
    if ok:
        out["best_holdout"] = min(ok, key=lambda k: float(np.mean(ok[k]["rmse_holdout"])))
    return out


def seasonal(s: R.Series, cfg: dict, quick: bool, default_station: bool = True) -> dict:
    sc = dict(cfg["seasonal"])
    if not default_station:
        sc["models"] = sc.get("models_other_stations", sc["models"])
    gap = int(cfg["data"]["max_gap_h"])
    start = float(s.t[0]) if cfg["data"].get("source") == "csv" else R.iso_to_seconds(s, f"{cfg['data']['year']}-01-01")
    n = 6 if quick else int(365 // float(sc["step_days"]))
    out: dict = {m: [] for m in sc["models"]}
    for i in range(n):
        t0 = start + i * float(sc["step_days"]) * R.DAY
        w = R.window(s, t0, float(sc["window_days"]), gap)
        if w["valid_fraction"] < float(sc["min_valid_fraction"]) or not len(w["t"]) or np.isnan(w["T"]).any():
            continue
        moist = w.get("moisture")
        mean_m = float(np.nanmean(moist[:, :3])) if moist is not None and np.isfinite(moist[:, :3]).any() else None
        frozen = bool(np.nanmin(w["T"][:, :4]) < float(sc["freeze_c"]))
        for name in sc["models"]:
            fc = model_cfg(cfg, name)
            tic = time.time()
            try:
                r = R.fit_conduction(w, cfg["model"], fc)
            except ValueError as err:
                out[name].append({"start": w["stamp"][0], "error": str(err)})
                continue
            print(
                f"   week {i + 1:2d} {name:13s} conv {r['fit']['converged']!s:5} {r['strategy_used'][:22]:22s} "
                f"frozen {frozen!s:5} ({time.time() - tic:.1f} s)",
                flush=True,
            )
            out[name].append(
                {
                    "start": w["stamp"][0],
                    "t_days": float((t0 - start) / R.DAY),
                    "alpha": [row["estimate"] for row in r["rows"] if row["parameter"].startswith("alpha")],
                    "ci_hac": [row["ci95_hac"] for row in r["rows"] if row["parameter"].startswith("alpha")],
                    "converged": r["fit"]["converged"],
                    "strategy_used": r["strategy_used"],
                    "rmse": r["rmse_per_sensor"],
                    "moisture": mean_m,
                    "mean_T": float(np.nanmean(w["T"][:, :4])),
                    "frozen": frozen,
                }
            )
    return out


def moisture_correlation(rows: list[dict], n_boot: int, seed: int) -> dict | None:
    pts = [
        (r["alpha"][0], r["moisture"])
        for r in rows
        if "alpha" in r and r["converged"] and r["moisture"] is not None and not r["frozen"]
    ]
    if len(pts) < 8:
        return None
    a, m = np.log(np.array(pts)[:, 0]), np.array(pts)[:, 1]
    rho = float(spearmanr(a, m).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(a), len(a))
        if np.ptp(a[k]) > 0 and np.ptp(m[k]) > 0:
            boots.append(spearmanr(a[k], m[k]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_weeks": len(pts), "spearman_rho": rho, "ci95": [float(lo), float(hi)]}


def robustness(s: R.Series, cfg: dict) -> dict:
    t0 = start_seconds(s, cfg)
    w = R.window(s, t0, float(cfg["fit"]["window_days"]), int(cfg["data"]["max_gap_h"]))
    out = {}
    for name in ("homogeneous", "three_layers"):
        fc0 = model_cfg(cfg, name)
        ref = R.fit_conduction(w, cfg["model"], fc0)
        x_ref = np.array(ref["x"])
        rows = []
        for init in cfg["robustness"]["inits"]:
            fc = {**fc0, "alpha_init": float(init)}
            for plain in (True, False):
                r = R.fit_conduction(w, cfg["model"], fc, plain=plain)
                dev = float(np.max(np.abs(np.array(r["x"]) - x_ref)))
                rows.append(
                    {
                        "init": float(init),
                        "mode": "plain LM" if plain else "pipeline",
                        "converged": r["fit"]["converged"],
                        "max_log_deviation_from_reference": dev,
                        "reached_reference": bool(r["fit"]["converged"] and dev < 1e-3),
                        "strategy_used": r["strategy_used"],
                        "nfev": sum(a["nfev"] for a in r["attempts"]),
                    }
                )
        out[name] = {"reference_x": x_ref.tolist(), "runs": rows}
        print(
            f"   robustness {name}: plain {sum(r['reached_reference'] for r in rows if r['mode'] == 'plain LM')}"
            f"/{len(rows) // 2}, pipeline {sum(r['reached_reference'] for r in rows if r['mode'] == 'pipeline')}"
            f"/{len(rows) // 2}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------------------------------- figures
def figures(met: dict, series: dict, cfg: dict, tag: str, odir: Path) -> list[str]:
    # the default run writes into docs/figures/real (a folder of its own: the Phase 7 catalogue covers docs/figures);
    # other tags into their results folder
    out = ROOT / "docs" / "figures" / "real" if tag == "default" else odir
    apply_style()
    names = []
    st0 = stations(cfg)[0]
    # 1 data overview of the default station
    s = series["data"]
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    days = s.t / R.DAY
    for j, z in enumerate(s.z):
        axs[0].plot(days, s.T[:, j], lw=0.6, color=PALETTE[j], label=f"{z * 100:.0f} cm")
        if s.moisture is not None:
            axs[1].plot(days, s.moisture[:, j], lw=0.8, color=PALETTE[j])
    axs[0].set(ylabel="soil temperature [°C]", title=f"NOAA USCRN {st0.replace('_', ' ')}, {cfg['data']['year']}")
    axs[0].legend(ncol=5, fontsize=7, loc="upper left")
    axs[1].set(xlabel="day of year", ylabel="soil moisture [m³ m⁻³]")
    names.append(f"p10_1_data_{tag}")
    save_fig(fig, names[-1], out, docs=False)
    # 2 default-window fit and hold-out
    cm = met["stations"][st0]["compare"]
    if "skipped" not in cm:
        tr, te = series["compare"]["train"], series["compare"]["test"]
        fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
        t_tr = tr["t"] / R.DAY - tr["t"][0] / R.DAY
        t_te = te["t"] / R.DAY - tr["t"][0] / R.DAY
        for i, z in enumerate(cfg["model"]["inner_m"]):
            ax = axs[i]
            k = int(np.argmin(np.abs(tr["z"] - z)))
            ax.plot(t_tr, tr["T"][:, k], color=DATA, lw=1.4, label="measured")
            ax.plot(t_te, te["T"][:, k], color=DATA, lw=1.4)
            for name, col, ls in (("homogeneous", SOLVER, "--"), ("three_layers", PINN, "-")):
                if name in series["compare"]:
                    sp = series["compare"][name]
                    ax.plot(t_tr, sp["train_pred"][:, i], color=col, lw=0.9, ls=ls, label=name.replace("_", " "))
                    ax.plot(t_te, sp["test_pred"][:, i], color=col, lw=0.9, ls=ls)
            ax.plot(t_te, te["baseline"][:, i], color=REF, lw=0.7, ls=":", label="linear in depth (no physics)")
            ax.axvspan(t_te[0], t_te[-1], color="0.92", zorder=0)
            ax.set(ylabel=f"T at {z * 100:.0f} cm [°C]")
        axs[0].legend(ncol=2, fontsize=7)
        axs[0].set_title("training window (white) and hold-out window (grey)")
        axs[1].set(xlabel="days from the start of the training window")
        names.append(f"p10_1_fit_{tag}")
        save_fig(fig, names[-1], out, docs=False)
    # 3 model comparison on hold-out, all stations with a comparison
    rows = [(st, m, v) for st, d in met["stations"].items() for m, v in d.get("compare", {}).get("models", {}).items()]
    if rows:
        sts = list(dict.fromkeys(r[0] for r in rows))
        mods = list(cfg["compare"])
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        w = 0.8 / (len(mods) + 1)
        for j, mname in enumerate(mods):
            vals = [
                np.mean(met["stations"][st]["compare"]["models"][mname]["rmse_holdout"])
                if mname in met["stations"][st]["compare"].get("models", {})
                else np.nan
                for st in sts
            ]
            conv = [met["stations"][st]["compare"]["models"].get(mname, {}).get("converged", False) for st in sts]
            bars = ax.bar(np.arange(len(sts)) + j * w, vals, w, color=PALETTE[j], label=mname.replace("_", " "))
            for b, c in zip(bars, conv, strict=True):
                if not c:
                    b.set_hatch("//")
        base = [np.mean(met["stations"][st]["compare"]["models"][mods[0]]["rmse_baseline_holdout"]) for st in sts]
        ax.bar(np.arange(len(sts)) + len(mods) * w, base, w, color="0.75", label="linear in depth")
        ax.set_xticks(np.arange(len(sts)) + 0.4 - w / 2, [s_.split("_")[1] for s_ in sts])
        ax.set(ylabel="hold-out RMSE [K]", title="model comparison on the hold-out window (hatched: not converged)")
        ax.legend(fontsize=7, ncol=3)
        names.append(f"p10_1_compare_{tag}")
        save_fig(fig, names[-1], out, docs=False)
    # 4 seasonal alpha and moisture
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for j, (st, d) in enumerate(met["stations"].items()):
        rows_ = [r for r in d.get("seasonal", {}).get("homogeneous", []) if "alpha" in r and r["converged"]]
        if not rows_:
            continue
        t = [r["t_days"] for r in rows_]
        a = [r["alpha"][0] for r in rows_]
        axs[0].semilogy(t, a, "o-", ms=3, lw=0.8, color=PALETTE[j], label=st.split("_")[1])
        mm = [(r["moisture"], r["alpha"][0]) for r in rows_ if r["moisture"] is not None and not r["frozen"]]
        if mm:
            axs[1].semilogy(*zip(*mm, strict=True), "o", ms=3, color=PALETTE[j])
    axs[0].set(xlabel="day of year", ylabel="weekly α, homogeneous model [m² s⁻¹]")
    axs[0].legend(fontsize=7)
    axs[1].set(
        xlabel="mean soil moisture 5–20 cm [m³ m⁻³]", ylabel="weekly α [m² s⁻¹]", title="unfrozen weeks (exploratory)"
    )
    names.append(f"p10_1_seasonal_{tag}")
    save_fig(fig, names[-1], out, docs=False)
    # 5 verification
    v = met["verification"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axs[0].loglog(v["dz"], v["err_dz"], "o-", color=SOLVER)
    axs[0].set(xlabel="Δz [m]", ylabel="max error [K]", title=f"space: order {v['order_space']:.2f}")
    axs[1].loglog(v["dt"], v["err_dt"], "o-", color=SOLVER)
    axs[1].set(xlabel="Δt [s]", ylabel="max error [K]", title=f"time: order {v['order_time']:.2f}")
    names.append(f"p10_1_verification_{tag}")
    save_fig(fig, names[-1], out, docs=False)
    return names


# ---------------------------------------------------------------------------------------------------- report
def report(met: dict, cfg: dict, path: Path) -> None:
    L = [
        "# Phase 10.1 — thermal diffusivity of real soils (auto-generated by experiments/p10_1_real_soil.py)",
        "",
        f"Data: NOAA USCRN hourly product, {cfg['data']['year']} (public domain); model: 1D conduction between the "
        f"{cfg['model']['top_m'] * 100:.0f} cm and {cfg['model']['bottom_m'] * 100:.0f} cm sensors, predicting "
        f"{', '.join(f'{z * 100:.0f}' for z in cfg['model']['inner_m'])} cm. This is VALIDATION against measured data "
        "(no PINN involved); seasonal correlations are exploratory, not pre-registered.",
        "",
        "## Verification of the forward model",
        "",
        f"- exact damped wave: observed order {met['verification']['order_space']:.2f} in space, "
        f"{met['verification']['order_time']:.2f} in time",
        f"- fast propagator vs step-by-step solver: max difference "
        f"{met['derivatives']['fast_vs_stepwise_max_abs']:.1e} K",
        f"- complex-step vs central-difference Jacobian: relative difference "
        f"{met['derivatives']['complex_step_vs_central_difference_rel']:.1e}",
        "",
        "## Stations",
        "",
        "| station | usable depths [cm] | data file matches download record |",
        "|---|---|---|",
    ]
    for st, d in met["stations"].items():
        L.append(
            f"| {st} | {', '.join(f'{z * 100:.0f}' for z in d['coverage']['usable_depths_m'])} | "
            f"{met['provenance'][st]['matches_download_record']} |"
        )
    L += [
        "",
        f"## Model comparison (training window from {cfg['fit']['start']}, {cfg['fit']['window_days']} days; "
        "hold-out = the following window)",
        "",
        "| station | model | converged | strategy | α estimates [m² s⁻¹] (95 % HAC interval) | hold-out RMSE [K] | "
        "skill vs linear-in-depth |",
        "|---|---|---|---|---|---|---|",
    ]
    for st, d in met["stations"].items():
        cmp_ = d.get("compare", {})
        if "skipped" in cmp_ or not cmp_.get("models"):
            L.append(f"| {st} | — | — | — | {cmp_.get('skipped', d.get('note', 'not fitted'))} | — | — |")
            continue
        for mname, v in cmp_["models"].items():
            al = "; ".join(
                f"{r['estimate']:.3g} [{r['ci95_hac'][0]:.2g}, {r['ci95_hac'][1]:.2g}]"
                for r in v["rows"]
                if r["parameter"].startswith("alpha")
            )
            other = "; ".join(
                f"{r['parameter']} {r['estimate']:.3g}" for r in v["rows"] if not r["parameter"].startswith("alpha")
            )
            L.append(
                f"| {st} | {mname} | {v['converged']} | {v['strategy_used']} | {al}{'; ' + other if other else ''} "
                f"| {', '.join(f'{x:.3f}' for x in v['rmse_holdout'])} | "
                f"{', '.join(f'{x:.2f}' for x in v['skill_holdout'])} |"
            )
    L += [
        "",
        "## Seasonal sweep (weekly windows; homogeneous model)",
        "",
        "| station | weeks fitted | converged | frozen weeks | median α [m² s⁻¹] | Spearman ρ(log α, moisture) [95 % CI] |",
        "|---|---|---|---|---|---|",
    ]
    for st, d in met["stations"].items():
        rows = [r for r in d.get("seasonal", {}).get("homogeneous", []) if "alpha" in r]
        if not rows:
            continue
        mc = d.get("moisture_correlation")
        corr = (
            "n/a"
            if not mc
            else f"{mc['spearman_rho']:.2f} [{mc['ci95'][0]:.2f}, {mc['ci95'][1]:.2f}] (n = {mc['n_weeks']})"
        )
        L.append(
            f"| {st} | {len(rows)} | {sum(r['converged'] for r in rows)} | {sum(r['frozen'] for r in rows)} | "
            f"{np.median([r['alpha'][0] for r in rows]):.3g} | {corr} |"
        )
    rb = met.get("robustness", {})
    if rb:
        L += [
            "",
            "## Convergence from poor initial guesses",
            "",
            "| model | mode | reached the reference solution |",
            "|---|---|---|",
        ]
        for mname, d in rb.items():
            for mode in ("plain LM", "pipeline"):
                rr = [r for r in d["runs"] if r["mode"] == mode]
                L.append(f"| {mname} | {mode} | {sum(r['reached_reference'] for r in rr)} / {len(rr)} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--quick", action="store_true", help="few seasonal windows and verification grids (smoke)")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.sets)
    with Experiment(f"p10_1_real_soil_{a.tag}", cfg) as ex:
        tic = time.time()
        met: dict = {"provenance": provenance(cfg)}
        print("verification ...", flush=True)
        met["verification"] = verification(cfg, a.quick)
        series: dict = {}
        met["stations"] = {}
        for i, st in enumerate(stations(cfg)):
            print(f"station {st}", flush=True)
            s = load_series(cfg, st)
            d: dict = {"coverage": coverage(s), "meta": {k: v for k, v in s.meta.items() if k != "air_temperature"}}
            usable = d["coverage"]["usable_depths_m"]
            need = [cfg["model"]["top_m"], *cfg["model"]["inner_m"], cfg["model"]["bottom_m"]]
            if not all(any(abs(u - z) < 1e-9 for u in usable) for z in need):
                d["note"] = (
                    f"sensors at {', '.join(f'{z * 100:.0f}' for z in need)} cm required; usable: "
                    f"{', '.join(f'{z * 100:.0f}' for z in usable)} cm -> two-depth harmonic estimate only"
                )
                t0 = start_seconds(s, cfg)
                w = R.window(s, t0, float(cfg["fit"]["window_days"]), int(cfg["data"]["max_gap_h"]))
                cols = [int(np.argmin(np.abs(s.z - z))) for z in usable]
                d["harmonic"] = R.harmonic(w["t"], w["T"][:, cols], np.array(usable))["pairs"]
                met["stations"][st] = d
                print(f"   {d['note']}", flush=True)
                continue
            if i == 0:
                series["data"] = s
                t0 = start_seconds(s, cfg)
                w = R.window(s, t0, float(cfg["fit"]["window_days"]), int(cfg["data"]["max_gap_h"]))
                met["derivatives"] = derivative_checks(w, cfg["model"], model_cfg(cfg, "homogeneous"))
            cm = compare_models(s, cfg, default_station=i == 0)
            if i == 0 and "skipped" not in cm:
                series["compare"] = cm.pop("series")
            else:
                cm.pop("series", None)
            d["compare"] = cm
            d["seasonal"] = seasonal(s, cfg, a.quick, default_station=i == 0)
            d["moisture_correlation"] = moisture_correlation(
                d["seasonal"].get("homogeneous", []), int(cfg["seasonal"]["bootstrap"]), int(cfg["seed"])
            )
            met["stations"][st] = d
            ex.metrics.update(met)  # partial results survive an interruption
        print("robustness ...", flush=True)
        met["robustness"] = robustness(load_series(cfg, stations(cfg)[0]), cfg)
        met["figures"] = figures(met, series, cfg, a.tag, ex.dir)
        met["wall_s"] = round(time.time() - tic, 1)
        met["git"] = git_hash()
        ex.metrics.update(met)
        report(
            met, cfg, ROOT / "docs" / "tables" / f"phase10_1_real_data{'' if a.tag == 'default' else '_' + a.tag}.md"
        )
    print(f"done in {met['wall_s']} s")


if __name__ == "__main__":
    main()
