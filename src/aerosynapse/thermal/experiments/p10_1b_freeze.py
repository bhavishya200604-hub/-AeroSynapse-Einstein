"""Phase 10.1b: the weeks that do not converge — does latent heat explain them?

    python -m aerosynapse.thermal.experiments.p10_1b_freeze [--stations NE_Lincoln_11_SW] [--tag default]

In the seasonal sweep of Step 10.1 (results/p10_1_real_soil_default) some weekly fits of the conduction model do
not converge even after the strategy pipeline; most of them are frozen weeks, where soil water freezes and thaws. An
optimiser cannot repair a model that lacks a process, so this step changes the MODEL, not the optimiser: every
non-converged frozen week of the chosen stations is refitted with
  conduction    the Step 10.1 homogeneous model (same grid as below, for a like-for-like comparison)
  freezing      the apparent-heat-capacity model c(T) dT/dt = alpha T_zz, c = 1 + S phi(T; T_f, w)
                (src/inverse/realdata.FreezingConduction1D): unknown alpha, latent temperature S and freezing point
                T_f, with physically plausible bounds (configs/real/soil.yaml, section freeze)
using the same identify-before-you-fit + convergence pipeline. A fit that ends on a plausibility bound is reported as
not converged (the data want an implausible parameter), never forced.
Writes results/p10_1b_freeze_<tag>/metrics.json and docs/tables/phase10_1b_freeze.md.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml

from aerosynapse.thermal.inverse import realdata as R
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

CFG = ROOT / "configs" / "real" / "soil.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="results/p10_1_real_soil_default/metrics.json")
    ap.add_argument("--stations", nargs="*", default=None, help="default: the configured default station")
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    fz = cfg["freeze"]
    met = json.loads((ROOT / a.source).read_text(encoding="utf-8"))
    stations = a.stations or [cfg["data"]["stations"][0]]
    mcfg = {**cfg["model"], "dz_m": float(fz["dz_m"]), "dt_s": float(fz["dt_s"])}
    fcfg = {**cfg["fit"], **fz}
    with Experiment(
        f"p10_1b_freeze_{a.tag}", {"seed": 0, "source": a.source, "stations": stations, "freeze": fz}
    ) as ex:
        tic0 = time.time()
        rows = []
        for st in stations:
            s = R.load_uscrn(
                ROOT / cfg["data"]["folder"] / str(cfg["data"]["year"]) / f"CRNH0203-{cfg['data']['year']}-{st}.txt"
            )
            weeks = [
                w
                for w in met["stations"][st]["seasonal"]["homogeneous"]
                if "alpha" in w and w["frozen"] and not w["converged"]
            ]
            print(f"{st}: {len(weeks)} non-converged frozen weeks", flush=True)
            for w in weeks:
                t0 = R.iso_to_seconds(s, w["start"][:10]) + 1800  # stamps are mid-hour
                win = R.window(s, t0 - 1800, float(cfg["seasonal"]["window_days"]), int(cfg["data"]["max_gap_h"]))
                rec = {
                    "station": st,
                    "start": w["start"],
                    "min_T": float(np.nanmin(win["T"][:, :4])),
                    "max_T": float(np.nanmax(win["T"][:, :4])),
                    "step_10_1": {"converged": w["converged"], "rmse": w["rmse"], "alpha": w["alpha"][0]},
                }
                tic = time.time()
                rc = R.fit_conduction(win, mcfg, fcfg)
                rec["conduction"] = {
                    "converged": rc["fit"]["converged"],
                    "at_bound": rc["fit"]["at_bound"],
                    "stationarity": rc["fit"]["stationarity"],
                    "rmse": rc["rmse_per_sensor"],
                    "alpha": rc["rows"][0]["estimate"],
                    "strategy": rc["strategy_used"],
                }
                rf = R.fit_freezing(win, mcfg, fcfg)
                rec["freezing"] = {
                    "converged": rf["fit"]["converged"],
                    "at_bound": rf["fit"]["at_bound"],
                    "stationarity": rf["fit"]["stationarity"],
                    "rmse": rf["rmse_per_sensor"],
                    "rows": rf["rows"],
                    "strategy": rf["strategy_used"],
                    "rank": rf["identifiability"]["rank"],
                    "condition": rf["identifiability"]["condition"],
                }
                rec["wall_s"] = round(time.time() - tic, 1)
                rows.append(rec)
                ex.metrics["weeks"] = rows
                print(
                    f"   {w['start'][:10]}  T {rec['min_T']:5.1f}..{rec['max_T']:5.1f}  conduction conv "
                    f"{rec['conduction']['converged']!s:5} rmse {np.round(rec['conduction']['rmse'], 3)}  freezing conv "
                    f"{rec['freezing']['converged']!s:5} bound {rec['freezing']['at_bound']!s:5} rmse "
                    f"{np.round(rec['freezing']['rmse'], 3)}  ({rec['wall_s']} s)",
                    flush=True,
                )
        n = len(rows)
        summ = {
            "weeks": n,
            "conduction_converged": sum(r["conduction"]["converged"] for r in rows),
            "freezing_converged": sum(r["freezing"]["converged"] for r in rows),
            "freezing_at_bound": sum(r["freezing"]["at_bound"] for r in rows),
            "median_rmse_conduction": float(np.median([np.mean(r["conduction"]["rmse"]) for r in rows])) if n else None,
            "median_rmse_freezing": float(np.median([np.mean(r["freezing"]["rmse"]) for r in rows])) if n else None,
            "freezing_better_rmse": sum(
                np.mean(r["freezing"]["rmse"]) < np.mean(r["conduction"]["rmse"]) for r in rows
            ),
        }
        ex.metrics.update({"weeks": rows, "summary": summ, "wall_s": round(time.time() - tic0, 1), "git": git_hash()})
        L = [
            "# Phase 10.1b — non-converged frozen weeks: conduction vs latent-heat model "
            "(auto-generated by experiments/p10_1b_freeze.py)",
            "",
            f"Stations: {', '.join(stations)}. Freezing model: c(T) dT/dt = α T_zz, c = 1 + S φ(T; T_f, w = "
            f"{fz['freeze_width_k']} K); bounds S ∈ {fz['latent_bounds_k']} K, T_f ∈ {fz['freezing_point_bounds_c']} °C. "
            "A fit on a plausibility bound counts as not converged.",
            "",
            f"Summary: {summ['weeks']} weeks; conduction converged {summ['conduction_converged']}, freezing converged "
            f"{summ['freezing_converged']} (on a bound {summ['freezing_at_bound']}); median RMSE conduction "
            f"{summ['median_rmse_conduction'] if summ['median_rmse_conduction'] is None else round(summ['median_rmse_conduction'], 3)} K, "
            f"freezing {summ['median_rmse_freezing'] if summ['median_rmse_freezing'] is None else round(summ['median_rmse_freezing'], 3)} K; "
            f"freezing lower RMSE in {summ['freezing_better_rmse']} of {summ['weeks']} weeks.",
            "",
            "| week | T range [°C] | conduction: converged / RMSE [K] | freezing: converged / on bound / RMSE [K] | "
            "α [m² s⁻¹], S [K], T_f [°C] |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            fr = r["freezing"]
            est = ", ".join(f"{x['estimate']:.3g}" for x in fr["rows"])
            L.append(
                f"| {r['start'][:10]} | {r['min_T']:.1f} … {r['max_T']:.1f} | {r['conduction']['converged']} / "
                f"{', '.join(f'{v:.3f}' for v in r['conduction']['rmse'])} | {fr['converged']} / {fr['at_bound']} / "
                f"{', '.join(f'{v:.3f}' for v in fr['rmse'])} | {est} |"
            )
        (ROOT / "docs" / "tables" / f"phase10_1b_freeze{'' if a.tag == 'default' else '_' + a.tag}.md").write_text(
            "\n".join(L) + "\n", encoding="utf-8"
        )
    print(summ)


if __name__ == "__main__":
    main()
