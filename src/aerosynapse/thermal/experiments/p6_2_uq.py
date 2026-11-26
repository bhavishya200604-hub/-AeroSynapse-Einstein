"""Step 6.2: predictive uncertainty of parametric 1D PINNs — deep ensembles, MC-dropout, calibration, conformal.

Per problem (uq_study.problems) one PARAMETRIC network family over the parameter range (log-uniform input):
  deep ensemble   members seeds 0..M-1 at the profile's settings (Adam + L-BFGS); metrics for the first M of each size
  MC-dropout      one network per dropout rate (and seed), Adam only, S stochastic forward passes
Evaluation against the analytic truth on random TEST and CALIBRATION parameter values (independent log-uniform draws,
exchangeable) and on a log-spaced GRID including both range ends.
Metrics (src/pinn/uq.py): accuracy of the predictive mean, coverage of central Gaussian intervals, ECE, CRPS,
dispersion (rms error / rms sd), Spearman rank correlation of sd and |error|; split-conformal multipliers from the
calibration values, coverage on the test values. Members are saved for Step 6.3 (extrapolation).

    python experiments/p6_2_uq.py [--profile smoke] [--config my.yaml] [--set key=value] [--tag name]
                                  [--figures-only] [--reevaluate]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aerosynapse.thermal.experiments.p3_pinn1d import train_one
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import uq as UQ
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.metrics import predict_grid
from aerosynapse.thermal.pinn.networks import clear_dropout_masks, sample_dropout_masks
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import uq_figs as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_2_uq"


def values(rng: list[float], st: dict) -> tuple[list[float], dict]:
    """All evaluation values and their index sets: grid (n_values log-spaced, both ends), calibration and test
    (random log-uniform draws from the SAME distribution, so exchangeable; random_values.seed)."""
    lo, hi = np.log(float(rng[0])), np.log(float(rng[1]))
    grid = np.geomspace(float(rng[0]), float(rng[1]), int(st["n_values"]))
    rv = st["random_values"]
    g = np.random.default_rng(int(rv["seed"]))
    cal = np.exp(g.uniform(lo, hi, int(rv["calibration"])))
    test = np.exp(g.uniform(lo, hi, int(rv["test"])))
    allv = [float(v) for v in np.concatenate([grid, cal, test])]
    n_g, n_c = len(grid), len(cal)
    sets = {
        "grid": list(range(n_g)),
        "calibration": list(range(n_g, n_g + n_c)),
        "test": list(range(n_g + n_c, len(allv))),
    }
    return allv, sets


def problem_override(problem: str, rng: list[float]) -> dict:
    return {"problems": {problem: {"parametric": True, "parameter": {"values": [float(rng[0]), float(rng[1])]}}}}


def member_spec(files, profile: str, cli: dict, problem: str, rng: list[float], over: dict, seed: int) -> dict:
    o = SP.deep_merge(problem_override(problem, rng), SP.deep_merge(over, {"seed": int(seed)}))
    return SP.load_spec(files or None, profile, SP.deep_merge(cli, o))


def dropout_override(rate: float, adam_epochs: int) -> dict:
    return {
        "network": {"dropout": float(rate)},
        "training": {"adam": {"epochs": int(adam_epochs)}, "lbfgs": {"iterations": 0}},
    }


def train_task(files, profile, cli, problem, rng, over, seed, out_dir, p_eval) -> dict:
    """One member (own process); its checkpoint is reloaded for the evaluation."""
    spec = member_spec(files, profile, cli, problem, rng, over, seed)
    prob = P.build_problem(problem, spec["problems"][problem])
    out = train_one(spec, prob, list(p_eval), Path(out_dir), f"{problem}_s{seed}")
    recs = [r for r, _ in out["records"]]
    return {
        "wall_s": recs[0]["wall_s"],
        "status": recs[0]["status"],
        "n_params": recs[0]["n_params"],
        "rel_l2_per_value": [r["rel_l2"] for r in recs],
        "val_loss": recs[0]["val_loss"],
    }


def load_member(spec: dict, problem: str, ckpt: Path):
    prob = P.build_problem(problem, spec["problems"][problem])
    net = build_network(spec["network"], prob.lo, prob.hi).to(dtype=torch.float64)
    net.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
    return prob, net


def predictions(spec_of, problem: str, members: list[dict], vals: list[float], nx: int, nt: int, samples: int):
    """truth (n_values, *grid); ensemble members {name: (n_values, *grid)}; MC-dropout {name: (mean, sd)} streamed
    over the samples (one sampled network = one fixed mask set, evaluated at every value)."""
    x = np.linspace(0.0, 1.0, nx)
    ens, mcd, truth = {}, {}, None
    for m in members:
        spec = spec_of(m)
        prob, net = load_member(spec, problem, Path(m["dir"]) / "checkpoint.pt")
        t = (prob.T * np.arange(1, nt + 1) / nt) if prob.has_time else None
        if truth is None:
            truth = np.stack([prob.reference(x, t, v) for v in vals])
        if m["kind"] == "ensemble":
            ens[m["name"]] = np.stack([predict_grid(net, prob, x, t, v)[0] for v in vals])
            continue
        gen = torch.Generator().manual_seed(12345)
        s1 = s2 = 0.0
        for _ in range(samples):
            sample_dropout_masks(net, gen)
            u = np.stack([predict_grid(net, prob, x, t, v)[0] for v in vals])
            s1, s2 = s1 + u, s2 + u * u
        clear_dropout_masks(net)
        mean = s1 / samples
        mcd[m["name"]] = (mean, np.sqrt(np.maximum(s2 - samples * mean * mean, 0.0) / (samples - 1)))
    return truth, ens, mcd


def _field_scores(mu, sd, y) -> np.ndarray:
    """one score per parameter value: max over its grid points of |error| / sd (simultaneous band)."""
    r = np.abs(y - mu) / np.maximum(sd, UQ.SD_FLOOR)
    return r.reshape(r.shape[0], -1).max(axis=1)


def analyse(st: dict, allv: list[float], sets: dict, truth, ens: dict, mcd: dict) -> dict:
    """Every method on the random TEST values (exchangeable with the random CALIBRATION values) and on the fixed GRID
    values (both range ends; per-value detail). Conformal: pointwise (pooled points) and field-level (one score per
    value, max over the field) multipliers from the calibration values."""
    levels = st["levels"]
    it, ic, ig = (np.asarray(sets[k]) for k in ("test", "calibration", "grid"))
    names = list(ens)
    stack = np.stack([ens[n] for n in names])  # (M, n_values, *grid)
    methods = {}
    for M in st["ensemble"]["sizes"]:
        if M <= len(names):
            methods[f"ensemble_M{M}"] = UQ.moments(stack[:M])
    methods.update(mcd)
    out = {"methods": {}, "members": {}}
    for i, n in enumerate(names):
        out["members"][n] = float(np.linalg.norm(stack[i, it] - truth[it]) / np.linalg.norm(truth[it]))
    for name, (mu, sd) in methods.items():
        res = UQ.summary(mu[it], sd[it], truth[it], levels)
        res["grid"] = UQ.summary(mu[ig], sd[ig], truth[ig], levels)
        res["per_value"] = [
            {
                "value": allv[i],
                **{
                    q: v
                    for q, v in UQ.summary(mu[i], sd[i], truth[i], levels).items()
                    if q in ("rel_l2", "rmse", "rms_sd", "dispersion", "rank_corr")
                },
                "coverage_95": UQ.coverage(mu[i], sd[i], truth[i], [0.95])[0.95],
            }
            for i in ig
        ]
        r_t = np.abs(truth[it] - mu[it]) / np.maximum(sd[it], UQ.SD_FLOOR)
        r_g = np.abs(truth[ig] - mu[ig]) / np.maximum(sd[ig], UQ.SD_FLOOR)
        res["reliability"] = {"levels": UQ.ECE_LEVELS.tolist(), "raw": [], "conformal": []}
        for lv in UQ.ECE_LEVELS:
            res["reliability"]["raw"].append(float(np.mean(r_t <= UQ.z_of(float(lv)))))
            q = UQ.conformal_scale(mu[ic], sd[ic], truth[ic], 1 - float(lv))
            res["reliability"]["conformal"].append(float(np.mean(r_t <= q)))
        f_c, f_t, f_g = (_field_scores(mu[i], sd[i], truth[i]) for i in (ic, it, ig))
        conf = {}
        for a in st["conformal_alpha"]:
            a = float(a)
            q = UQ.conformal_scale(mu[ic], sd[ic], truth[ic], a)
            qf = UQ.conformal_scale(np.zeros_like(f_c), np.ones_like(f_c), f_c, a)
            conf[str(a)] = {
                "nominal": 1 - a,
                "q": q,
                "z_gauss": UQ.z_of(1 - a),
                "inflation": q / UQ.z_of(1 - a),
                "test_coverage": float(np.mean(r_t <= q)),
                "grid_coverage": float(np.mean(r_g <= q)),
                "field_q": qf,
                "field_test_coverage": float(np.mean(f_t <= qf)),
                "field_grid_coverage": float(np.mean(f_g <= qf)),
                "field_raw_test_coverage": float(np.mean(f_t <= UQ.z_of(1 - a))),
            }
        res["conformal"] = conf
        res["ece_conformal"] = float(np.mean(np.abs(np.array(res["reliability"]["conformal"]) - UQ.ECE_LEVELS)))
        out["methods"][name] = res
    # example for the band figure: middle grid value, time slice with the largest error of the largest ensemble
    k = int(ig[len(ig) // 2])
    best = f"ensemble_M{max(s for s in st['ensemble']['sizes'] if s <= len(names))}"
    mu, sd = methods[best]
    err = np.abs(mu[k] - truth[k])
    j = int(np.argmax(err.max(axis=-1))) if err.ndim == 2 else None
    pick = lambda a: a[k] if j is None else a[k, j]  # noqa: E731
    out["example"] = {
        "value": float(allv[k]),
        "slice": j,
        "truth": pick(truth).tolist(),
        "methods": {n: {"mu": pick(m_).tolist(), "sd": pick(s_).tolist()} for n, (m_, s_) in methods.items()},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument(
        "--reevaluate", action="store_true", help="recompute every metric from the saved members (no training)"
    )
    a = ap.parse_args()
    cli = SP.parse_set(a.sets)
    spec = SP.load_spec(a.config or None, a.profile, cli)
    st = spec["uq_study"]
    custom = bool(a.config or a.sets or a.tag != "default" or a.profile != "smoke")
    odir = ROOT / "results" / f"p6_2_uq_{a.tag}"
    table = odir / "uq.md" if custom else ROOT / "docs" / "tables" / "phase6_2_uq.md"

    def spec_of(m):
        return member_spec(
            a.config, a.profile, cli, m["problem"], st["problems"][m["problem"]]["range"], m["over"], m["seed"]
        )

    def evaluate_all(members: list[dict]) -> dict:
        res = {}
        nt = spec["evaluation"]["nt"]
        for prob, pc in st["problems"].items():
            allv, sets = values(pc["range"], st)
            ms = [m for m in members if m["problem"] == prob]
            truth, ens, mcd = predictions(spec_of, prob, ms, allv, st["eval_nx"], nt, st["mc_dropout"]["samples"])
            res[prob] = analyse(st, allv, sets, truth, ens, mcd) | {
                "grid_values": [allv[i] for i in sets["grid"]],
                "n_calibration": len(sets["calibration"]),
                "n_test": len(sets["test"]),
            }
        return res

    if a.figures_only or a.reevaluate:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        if a.reevaluate:
            met["analysis"] = evaluate_all(met["members"])
            met["reevaluated"] = {"git": git_hash(), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            (odir / "metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
        met["figures"] = figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        print(f"wrote {table}")
        return
    with Experiment(f"p6_2_uq_{a.tag}", spec) as ex:
        members, jobs = [], []
        for prob, pc in st["problems"].items():
            allv, sets = values(pc["range"], st)
            vt = [allv[i] for i in sets["grid"]]
            kinds = [("ensemble", f"ens_s{s}", {}, s) for s in range(int(st["ensemble"]["members"]))]
            for rate in st["mc_dropout"]["rates"]:
                for s in st["mc_dropout"]["seeds"]:
                    over = dropout_override(rate, st["mc_dropout"]["adam_epochs"])
                    kinds.append(("mc_dropout", f"mcd_{rate:g}_s{s}", over, s))
            for kind, name, over, s in kinds:
                d = ex.dir / "members" / f"{prob}_{name}"
                members.append(
                    {"problem": prob, "kind": kind, "name": name, "over": over, "seed": int(s), "dir": str(d)}
                )
                jobs.append(
                    (
                        MOD,
                        "train_task",
                        dict(
                            files=a.config,
                            profile=a.profile,
                            cli=cli,
                            problem=prob,
                            rng=pc["range"],
                            over=over,
                            seed=int(s),
                            out_dir=str(d),
                            p_eval=vt,
                        ),
                    )
                )
        print(f"{len(jobs)} trainings ({len(st['problems'])} problems)", flush=True)
        outs = run_tasks(
            jobs,
            {"isolate": True, **st.get("execution", {})},
            lambda i, r, sec: print(
                f"[{i + 1}/{len(jobs)}] {members[i]['problem']} {members[i]['name']}: "
                f"rel L2 (test values) {min(r['rel_l2_per_value']):.3g}-{max(r['rel_l2_per_value']):.3g}, {r['wall_s']:.0f} s",
                flush=True,
            ),
        )
        for m, r in zip(members, outs, strict=True):
            m["training"] = r
            ex.log(
                problem=m["problem"], member=m["name"], wall_s=r["wall_s"], status=r["status"], val_loss=r["val_loss"]
            )
        ex.metrics.update({"members": members, "git": git_hash()})
        ex.metrics["analysis"] = evaluate_all(members)
        ex.metrics["figures"] = figures(ex.metrics, a.tag, ex.dir, docs=not custom)
        report(ex.metrics, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def _pct(v) -> str:
    return f"{100 * v:.0f} %"


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Step 6.2 predictive uncertainty — `{tag}` (auto-generated by experiments/p6_2_uq.py)",
        "",
        "Parametric 1D networks (parameter = input, log-uniform over the range); truth = analytic solution. TEST and "
        "CALIBRATION = independent random log-uniform parameter values from the same distribution (exchangeable); GRID = "
        "log-spaced values including both range ends. Coverage = fraction of grid points inside mean ± z·sd (pointwise). "
        "Dispersion = rms error / rms sd (1 = consistent, > 1 overconfident). Rank corr. = Spearman(sd, |error|). Field "
        "conformal = one score per parameter value (max over the field): a simultaneous band. Verification only.",
        "",
    ]
    for prob, an in met["analysis"].items():
        L += [
            f"## {prob} ({SP.PROBLEMS.get(prob, 'parameter')} in [{an['grid_values'][0]:.3g}, {an['grid_values'][-1]:.3g}]; "
            f"{an['n_test']} test, {an['n_calibration']} calibration values, grid "
            + ", ".join(f"{v:.3g}" for v in an["grid_values"])
            + ")",
            "",
            "| method | rel L2 of mean | rms sd | dispersion | coverage 50 / 80 / 90 / 95 % | ECE | CRPS | rank corr. | "
            "ECE after conformal | grid: rel L2 / dispersion / 95 % coverage |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, r in an["methods"].items():
            c, g = r["coverage"], r["grid"]
            L.append(
                f"| {name} | {r['rel_l2']:.3g} | {r['rms_sd']:.2g} | {r['dispersion']:.3g} | "
                + " / ".join(_pct(c[k]) for k in sorted(c, key=float))
                + f" | {r['ece']:.3f} | {r['crps']:.2g} | {r['rank_corr']:.2f} | {r['ece_conformal']:.3f} | "
                f"{g['rel_l2']:.3g} / {g['dispersion']:.3g} / {_pct(g['coverage']['0.95'])} |"
            )
        mem = list(an["members"].values())
        if mem:
            L += [
                "",
                f"Single ensemble members, rel L2 on the test values: {min(mem):.3g}-{max(mem):.3g} (median {np.median(mem):.3g}).",
            ]
        L += [
            "",
            "| method | nominal | pointwise q (z) | coverage test / grid | field q | field coverage test / grid (raw z band on test) |",
            "|---|---|---|---|---|---|",
        ]
        for name, r in an["methods"].items():
            for c in r["conformal"].values():
                L.append(
                    f"| {name} | {_pct(c['nominal'])} | {c['q']:.3g} ({c['z_gauss']:.3g}) | {_pct(c['test_coverage'])} / "
                    f"{_pct(c['grid_coverage'])} | {c['field_q']:.3g} | {_pct(c['field_test_coverage'])} / "
                    f"{_pct(c['field_grid_coverage'])} ({_pct(c['field_raw_test_coverage'])}) |"
                )
        L += ["", "Per grid value (rmse, rms sd, 95 % coverage):", ""]
        for name in [k for k in an["methods"] if k.startswith("mcd_")] + [
            max((k for k in an["methods"] if k.startswith("ensemble_M")), key=lambda k: int(k[10:]))
        ]:
            L.append(
                f"- {name}: "
                + "; ".join(
                    f"{v['value']:.3g}: {v['rmse']:.2g}, {v['rms_sd']:.2g}, {_pct(v['coverage_95'])}"
                    for v in an["methods"][name]["per_value"]
                )
            )
        L.append("")
    tr = [m["training"] for m in met["members"] if "training" in m]
    if tr:
        L.append(f"Training: {len(tr)} members, {sum(x['wall_s'] for x in tr):.0f} s in total (single-thread each).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    an = met["analysis"]
    paths = []
    paths += save_fig(F.reliability(an), f"p6_2_reliability_{tag}", odir, docs=docs)
    paths += save_fig(F.spread_vs_error(an), f"p6_2_spread_{tag}", odir, docs=docs)
    paths += save_fig(F.bands(an), f"p6_2_bands_{tag}", odir, docs=docs)
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
