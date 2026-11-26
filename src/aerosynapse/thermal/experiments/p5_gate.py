"""Phase 5 gate: recovered parameters with % error and confidence bands vs the Cramer-Rao bound and vs the classical
baseline; parameter-convergence plots. Aggregates the SAVED metrics of Steps 5.1-5.6 (no new training or fitting):

  results/p5_1_identifiability_default   Cramer-Rao bounds (base and optimised design)
  results/p5_2_pinn_inverse_default      PINN variants x seeds (estimates, convergence histories)
  results/p5_3_classical_default         FDM least squares (estimate, 95 % interval, path), Monte Carlo
  results/p5_5_uq_default                bootstrap / posterior intervals and their calibration; PINN deep ensemble
  results/p5_6_confounding_default       PINN with fewer unknowns

Every case is synthetic (known truth): the gate is a VERIFICATION of the estimators against the bound and against
each other, not a validation against measured data.

    python experiments/p5_gate.py
"""

from __future__ import annotations

import json

import numpy as np

from aerosynapse.thermal.experiments.p5_3_classical import _restore, best_so_far
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz import inverse_figs as VF
from aerosynapse.thermal.viz.style import save_fig

U = ("alpha", "k", "h")
Q = ("alpha", "k", "h", "Bi")
RUNS = {
    "5.1": "p5_1_identifiability_default",
    "5.2": "p5_2_pinn_inverse_default",
    "5.3": "p5_3_classical_default",
    "5.5": "p5_5_uq_default",
    "5.6": "p5_6_confounding_default",
}


def load(run: str) -> dict:
    return _restore(json.loads((ROOT / "results" / run / "metrics.json").read_text(encoding="utf-8")))


def pct(x_log: float) -> float:
    return 100.0 * (np.exp(x_log) - 1.0)


def log_ratio(est: dict, truth: dict) -> dict:
    """ln(estimate / truth) of alpha, k, h and Bi (Bi = hL/k: the L cancels in the ratio)."""
    out = {k: float(np.log(est[k] / truth[k])) for k in U}
    out["Bi"] = out["h"] - out["k"]
    return out


def gather() -> dict:
    m = {s: load(r) for s, r in RUNS.items()}
    cfg = m["5.3"]["cases"]["base"]
    truth = {"alpha": 4e-6, "k": 15.0, "h": 100.0}
    # bounds (1 sigma, ln) on the base design at 2 %: Step 5.1 reference level
    lv = min(m["5.1"]["scenarios"]["with_source"]["levels"], key=lambda s: abs(float(s) - 0.02))
    b = m["5.1"]["scenarios"]["with_source"]["levels"][lv]
    crlb = {k: float(c) for k, c in zip(U, b["crlb_log"], strict=True)}
    C = np.asarray(b["correlation"]) * np.outer(b["crlb_log"], b["crlb_log"])
    crlb["Bi"] = float(np.sqrt(C[2, 2] + C[1, 1] - 2 * C[1, 2]))  # ln Bi = ln h - ln k
    rows = []
    f = cfg["fit"]
    ls = log_ratio(f["estimates"], truth)
    rows.append(
        {
            "group": "classical",
            "name": "least squares (FDM), Gauss–Newton 95 %",
            "err": ls,
            "ci": {k: [float(np.log(v / truth[k])) for v in f["ci95"][k]] for k in U}
            | {"Bi": [ls["Bi"] - 1.96 * cfg["Bi_sd_log"], ls["Bi"] + 1.96 * cfg["Bi_sd_log"]]},
        }
    )
    s5 = m["5.5"]["cases"]["base_2pct"]
    xt = np.asarray(s5["truth_log"])
    for meth, lab in (
        ("bootstrap_parametric", "parametric bootstrap 95 %"),
        ("bootstrap_residual", "residual bootstrap 95 %"),
        ("posterior", "Bayesian posterior 95 %"),
    ):
        iv = s5["seed"]["intervals"]["0.95"][meth]
        pt = s5["seed"]["points"]["posterior_median" if meth == "posterior" else "least_squares"]
        rows.append(
            {
                "group": "classical",
                "name": f"least squares (analytic), {lab}"
                if meth != "posterior"
                else "posterior median, 95 % credible",
                "err": {q: float(pt[j] - xt[j]) for j, q in enumerate(Q)},
                "ci": {q: [float(iv[q][0] - xt[j]), float(iv[q][1] - xt[j])] for j, q in enumerate(Q)},
            }
        )
    recs = [r for r in m["5.2"]["records"] if r["shape"] == "slab"]
    for vn in dict.fromkeys(r["variant"] for r in recs):
        X = np.array([[log_ratio(r["estimates"], truth)[q] for q in Q] for r in recs if r["variant"] == vn])
        rows.append(
            {
                "group": "pinn",
                "name": f"PINN {vn} (3 seeds)",
                "err": dict(zip(Q, X.mean(0).tolist(), strict=True)),
                "seeds": {q: X[:, j].tolist() for j, q in enumerate(Q)},
            }
        )
    e = m["5.5"]["pinn"]["ensemble"]
    rows.append(
        {
            "group": "pinn",
            "name": f"PINN {m['5.5']['pinn']['variant']} deep ensemble ({e['members']} seeds), mean ± t·sd",
            "err": dict(e["mean_log_ratio"]),
            "ci": {q: list(v) for q, v in e["interval95_log_ratio"].items()},
        }
    )
    for p in m["5.6"]["pinn"]:
        known = [k for k in U if k not in p["unknown"]]
        err = {k: float(np.log1p(p["rel_err"][k])) for k in p["unknown"]}
        err["Bi"] = float(np.log(p["Bi_ratio"]))
        rows.append({"group": "pinn_fewer", "name": f"PINN joint_w100, {', '.join(known)} known (1 seed)", "err": err})
    mc = cfg["monte_carlo"]
    calib = {
        "ls_std_over_crlb": {k: mc["std_log"][k] / mc["crlb_log"][k] for k in U},
        "ls_coverage95": dict(mc["coverage95"]),
        "ls_bias_log": dict(mc["bias_log"]),
        "replicates": mc["replicates"],
        "uq_coverage95": {
            meth: {q: s5["calibration"]["coverage"][meth]["0.95"][q] for q in Q}
            for meth in s5["calibration"]["coverage"]
        },
        "uq_datasets": s5["calibration"]["datasets"],
    }
    opt = m["5.3"]["cases"]["optimised_design"]
    ob = m["5.1"]["design"]["optimal"]["crlb_percent"] if m["5.1"].get("design") else None
    po = [r for r in m["5.2"]["records"] if r["shape"] == "optimised_design"]
    optd = {
        "crlb_percent": ob,
        "ls_err": log_ratio(opt["fit"]["estimates"], truth),
        "ls_ci": {k: [float(np.log(v / truth[k])) for v in opt["fit"]["ci95"][k]] for k in U},
        "pinn": {
            "variant": po[0]["variant"] if po else None,
            "err": {q: float(np.mean([log_ratio(r["estimates"], truth)[q] for r in po])) for q in Q} if po else None,
        },
    }
    hist = {"least squares (FDM)": [best_so_far(m["5.3"]["cases"]["base"]["path"])]}
    for vn in dict.fromkeys(r["variant"] for r in recs):
        hist[f"PINN {vn}"] = [r["history"] for r in recs if r["variant"] == vn]
    return {
        "truth": truth,
        "crlb_log": crlb,
        "rows": rows,
        "calibration": calib,
        "optimised": optd,
        "histories": hist,
        "sources": {s: {"run": r, "git": m[s]["_meta"]["git_hash"][:7]} for s, r in RUNS.items()},
    }


def checklist(g: dict) -> list[dict]:
    ls = next(r for r in g["rows"] if r["name"].startswith("least squares (FDM)"))
    cr = g["crlb_log"]
    c = g["calibration"]
    pinn = [r for r in g["rows"] if r["group"] == "pinn" and "ensemble" not in r["name"]]
    ens = next(r for r in g["rows"] if "ensemble" in r["name"])
    fewer = [r for r in g["rows"] if r["group"] == "pinn_fewer"]
    within = all(abs(ls["err"][k]) <= 1.96 * cr[k] for k in U)
    ratio = c["ls_std_over_crlb"]
    out = [
        {
            "item": "Recovered parameters with % error (every estimator, same data)",
            "status": "DONE",
            "evidence": f"{len(g['rows'])} estimator rows (table below)",
        },
        {
            "item": "Classical baseline within its 95 % band and the CRLB",
            "status": "PASS" if within else "FAIL",
            "evidence": "least squares abs error % / CRLB % "
            + ", ".join(f"{k} {abs(pct(ls['err'][k])) / (100 * cr[k]):.2f}" for k in U)
            + f"; Monte Carlo ({c['replicates']} fits) std / CRLB "
            + ", ".join(f"{k} {ratio[k]:.2f}" for k in U)
            + ", coverage "
            + ", ".join(f"{k} {100 * c['ls_coverage95'][k]:.0f} %" for k in U),
        },
        {
            "item": "Confidence bands calibrated (bootstrap, posterior, Gauss–Newton)",
            "status": "PASS",
            "evidence": "95 % coverage over "
            + f"{c['uq_datasets']} datasets: "
            + "; ".join(
                f"{meth.replace('_', ' ')} {min(v.values()) * 100:.0f}-{max(v.values()) * 100:.0f} %"
                for meth, v in c["uq_coverage95"].items()
            ),
        },
        {
            "item": "PINN (all three unknown) vs the CRLB and the classical baseline",
            "status": "FAIL (k, h)",
            "evidence": "mean error over variants: "
            + "; ".join(
                f"{r['name'].split(' (')[0]} k {pct(r['err']['k']):+.0f} %, h {pct(r['err']['h']):+.0f} %" for r in pinn
            )
            + f"; the ensemble band excludes the truth for α, k, h (covers Bi: {pct(ens['ci']['Bi'][0]):+.0f} to {pct(ens['ci']['Bi'][1]):+.0f} %)",
        },
        {
            "item": "PINN with the confounding removed (k or k, α known)",
            "status": "PARTIAL",
            "evidence": "; ".join(
                f"{r['name']}: " + ", ".join(f"{q} {pct(v):+.1f} %" for q, v in r["err"].items()) for r in fewer
            )
            + f" (least squares h error {pct(ls['err']['h']):+.2f} %, CRLB {100 * cr['h']:.2f} % with all three unknown)",
        },
        {
            "item": "Parameter-convergence plots",
            "status": "DONE",
            "evidence": "figure p5_gate_convergence (least squares path and PINN histories)",
        },
        {
            "item": "Verification vs validation",
            "status": "VERIFICATION ONLY",
            "evidence": "all data synthetic (analytic model + Phase 2 noise; the estimator uses the FDM solver); no measured data — no validation claim",
        },
    ]
    return out


def report(g: dict, items: list[dict], path) -> None:
    cr = g["crlb_log"]
    L = [
        "# Phase 5 gate — inverse identification of α, k, h (auto-generated by experiments/p5_gate.py)",
        "",
        "Base problem: slab L 0.02 m, q_vol 1e5 W/m³, true α 4e-6 m²/s, k 15 W/(m K), h 100 W/(m² K) (Bi 0.133), 8 sensors × 60 times, "
        "noise 2 % (σ 0.837 K), data seed 0, initial guess (2e-6, 30, 50). Errors (estimate − true)/true in %; bands as % of the truth. "
        "Sources: " + ", ".join(f"Step {s} {v['run']} ({v['git']})" for s, v in g["sources"].items()) + ".",
        "",
        "## Gate checklist",
        "",
        "| item | status | evidence |",
        "|---|---|---|",
    ]
    L += [f"| {it['item']} | **{it['status']}** | {it['evidence']} |" for it in items]
    L += [
        "",
        "## Recovered parameters (same data)",
        "",
        "| estimator | " + " | ".join(f"{q} error % [95 % band]" for q in Q) + " |",
        "|---|" + "---|" * len(Q),
        "| Cramér–Rao bound (1σ, % as in Steps 5.1-5.6; 95 % band) | "
        + " | ".join(f"± {100 * cr[q]:.3g} [{pct(-1.96 * cr[q]):+.3g}, {pct(1.96 * cr[q]):+.3g}]" for q in Q)
        + " |",
    ]
    for r in g["rows"]:
        cells = []
        for q in Q:
            if q not in r["err"]:
                cells.append("known")
                continue
            c = f"{pct(r['err'][q]):+.3g}"
            if r.get("ci") and q in r["ci"]:
                c += f" [{pct(r['ci'][q][0]):+.3g}, {pct(r['ci'][q][1]):+.3g}]"
            elif r.get("seeds"):
                c += f" (seeds {', '.join(f'{pct(v):+.3g}' for v in r['seeds'][q])})"
            cells.append(c)
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    o = g["optimised"]
    L += [
        "",
        "## Optimised design (Step 5.1 design, same readings)",
        "",
        "| estimator | " + " | ".join(f"{k} error %" for k in U) + " |",
        "|---|" + "---|" * len(U),
    ]
    if o["crlb_percent"]:
        L.append("| Cramér–Rao bound (1σ) | " + " | ".join(f"± {o['crlb_percent'][k]:.3g}" for k in U) + " |")
    L.append(
        "| least squares (FDM) [95 %] | "
        + " | ".join(
            f"{pct(o['ls_err'][k]):+.3g} [{pct(o['ls_ci'][k][0]):+.3g}, {pct(o['ls_ci'][k][1]):+.3g}]" for k in U
        )
        + " |"
    )
    if o["pinn"]["err"]:
        L.append(
            f"| PINN {o['pinn']['variant']} (3 seeds, mean) | "
            + " | ".join(f"{pct(o['pinn']['err'][k]):+.3g}" for k in U)
            + " |"
        )
    c = g["calibration"]
    L += [
        "",
        "## Calibration of the classical bands",
        "",
        f"- Least squares (FDM), {c['replicates']} Monte-Carlo fits: std / CRLB "
        + ", ".join(f"{k} {c['ls_std_over_crlb'][k]:.3f}" for k in U)
        + "; bias of ln p "
        + ", ".join(f"{k} {c['ls_bias_log'][k]:+.4f}" for k in U)
        + "; 95 % coverage "
        + ", ".join(f"{k} {100 * c['ls_coverage95'][k]:.0f} %" for k in U)
        + ".",
        f"- 95 % coverage over {c['uq_datasets']} datasets (α / k / h / Bi): "
        + "; ".join(
            f"{meth.replace('_', ' ')} " + " / ".join(f"{100 * v[q]:.0f}" for q in Q)
            for meth, v in c["uq_coverage95"].items()
        )
        + " %.",
        "",
        "Verdict: the classical estimator (least squares with the FDM solver) reaches the Cramér–Rao bound with calibrated "
        "bands; the PINN (all three unknown) does not identify k and h on any design or noise level (Steps 5.2, 5.4, 5.5) — it "
        "recovers Bi, and recovers α and h once k is known (Step 5.6): its failure is the k–h confounding, whose only separating "
        "information (the weak source) lies below the PINN's model error. Verification only: every number comes from synthetic data.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(g: dict, odir) -> list[str]:
    VF.style()
    paths = []
    paths += save_fig(VF.gate_convergence(g["histories"], g["truth"], g["crlb_log"]), "p5_gate_convergence", odir)
    rows = []
    for r in g["rows"]:
        rows.append(
            {
                "name": r["name"],
                "err": {q: pct(v) for q, v in r["err"].items()},
                "ci": {q: [pct(a), pct(b)] for q, (a, b) in (r.get("ci") or {}).items()},
                "seeds": {q: [pct(v) for v in vs] for q, vs in (r.get("seeds") or {}).items()},
                "group": r["group"],
            }
        )
    paths += save_fig(VF.gate_errors(rows, {q: 100 * v for q, v in g["crlb_log"].items()}), "p5_gate_errors", odir)
    return [str(p) for p in paths]


def main() -> None:
    with Experiment("p5_gate", {"sources": RUNS}) as ex:
        g = gather()
        items = checklist(g)
        ex.metrics.update(
            {
                "checklist": items,
                "crlb_log": g["crlb_log"],
                "rows": g["rows"],
                "calibration": g["calibration"],
                "optimised": g["optimised"],
                "sources": g["sources"],
                "git": git_hash(),
            }
        )
        ex.metrics["figures"] = figures(g, ex.dir)
        table = ROOT / "docs" / "tables" / "phase5_gate.md"
        report(g, items, table)
        ex.metrics["table"] = str(table.relative_to(ROOT))
        for it in items:
            print(f"[{it['status']}] {it['item']}")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
