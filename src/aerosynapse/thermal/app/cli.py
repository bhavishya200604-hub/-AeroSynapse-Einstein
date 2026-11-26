"""Phase 8 command-line interface: the app's computations without a browser (same backend, src/app/backend.py).

    python -m aerosynapse.thermal.app.cli cases                                        # the cases, parameters and PINN ranges
    python -m aerosynapse.thermal.app.cli validity --case cavity --set Re=12000        # the validity guard only
    python -m aerosynapse.thermal.app.cli solve    --case B_slab --set Bi=2 --opt N=80 --out slab.csv
    python -m aerosynapse.thermal.app.cli predict  --case A_steady --set Pe=12 --out pinn.json
    python -m aerosynapse.thermal.app.cli compare  --case D_graetz --set Re=30 Pr=1 Bi=5
    python -m aerosynapse.thermal.app.cli converge --case A_steady --set Pe=10 --grids 10 20 40 80 160
    python -m aerosynapse.thermal.app.cli inverse  --q-vol 1e5 --noise 0.02 --unknown alpha k h [--data my.csv]

Every command prints a JSON summary; --out writes the full result (.json) or the field (.csv). Exit code 2 when the
validity guard says 'stop' and --strict is given (for scripts that must not use extrapolated answers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from aerosynapse.thermal.app import backend as B


def _kv(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not _:
            raise SystemExit(f"expected name=value, got {it!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _summary(cmd: str, key: str | None, r: dict) -> dict:
    keep = (
        "err_inf",
        "wall_s",
        "Nu",
        "psi_min",
        "converged",
        "n_members",
        "rel_l2",
        "max_abs",
        "solver_s",
        "pinn_s",
        "speedup",
        "Nu_solver",
        "Nu_pinn",
        "Nu_rel_err",
        "rank",
        "identifiable",
        "Bi",
        "chi2_per_reading",
        "message",
    )
    s = {k: r[k] for k in keep if k in r}
    if cmd == "predict":
        s["mean_spread"] = float(np.mean(r["sd"]))
    if cmd == "converge":
        s["observed_order"] = {k: v["observed_order"] for k, v in r["schemes"].items()}
    if cmd == "inverse":
        s["estimates"] = r["rows"]
    return B.jsonable(s)


def _write(path: str, key: str | None, cmd: str, r: dict) -> None:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        if cmd == "solve":
            p.write_bytes(B.to_csv(B.field_columns(key, r, "theta")))
        elif cmd == "predict":
            p.write_bytes(B.to_csv(B.field_columns(key, r, "mean")))
        else:
            raise SystemExit("CSV output is for solve / predict; use .json for the other commands")
    else:
        p.write_text(json.dumps(B.jsonable(r), indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cases", "validity", "solve", "predict", "compare", "converge", "inverse"])
    ap.add_argument("--case", default="B_slab", choices=list(B.CASES))
    ap.add_argument("--set", nargs="*", default=[], help="parameters, e.g. Re=30 Pr=1 Bi=5")
    ap.add_argument("--opt", nargs="*", default=[], help="conditions / grid, e.g. N=80 scheme=quick members=[...]")
    ap.add_argument("--grids", nargs="*", type=int, default=None)
    ap.add_argument("--schemes", nargs="*", default=None)
    ap.add_argument("--q-vol", type=float, default=1.0e5)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sensors", type=int, default=8)
    ap.add_argument("--unknown", nargs="*", default=["alpha", "k", "h"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None, help="CSV with x [m], t [s], T [, sigma]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 2 if the validity guard says stop")
    a = ap.parse_args(argv)

    if a.command == "cases":
        rows = {
            k: {
                "title": c.title,
                "dim": c.dim,
                "transient": c.transient,
                "parameters": {p.name: {"default": p.default, "pinn_range": p.train} for p in c.params},
                "conditions": c.conditions,
                "pinn": c.pinn,
                "note": c.pinn_note,
                "available": B.pinn_available(k),
            }
            for k, c in B.CASES.items()
        }
        print(json.dumps(B.jsonable(rows), indent=1))
        return 0

    c = B.case(a.case)
    params = {**c.defaults(), **{k: float(v) for k, v in _kv(a.set).items()}}
    unknown = [k for k in params if k not in c.defaults()]
    if unknown:
        raise SystemExit(f"unknown parameter(s) {unknown} for {a.case}; parameters: {list(c.defaults())}")
    opts = _kv(a.opt)
    val = B.validity(a.case, params, opts)
    if a.command == "validity":
        print(json.dumps(val, indent=1))
        return 2 if a.strict and any(v["level"] == "stop" for v in val) else 0
    if a.command == "solve":
        r = B.solve(a.case, params, opts)
    elif a.command == "predict":
        r = B.predict(a.case, params, opts)
    elif a.command == "compare":
        r = B.compare(a.case, params, opts)
    elif a.command == "converge":
        r = B.convergence(a.case, params, a.grids, a.schemes, opts)
    else:
        r = B.inverse(a.q_vol, a.noise, a.sensors, tuple(a.unknown), seed=a.seed, data_file=a.data)
        val = []
    out = {
        "command": a.command,
        "case": None if a.command == "inverse" else a.case,
        "parameters": params,
        "validity": val,
        "result": _summary(a.command, a.case, r),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        _write(a.out, a.case, a.command, r)
        print(f"wrote {a.out}", file=sys.stderr)
    return 2 if a.strict and any(v["level"] == "stop" for v in val) else 0


if __name__ == "__main__":
    raise SystemExit(main())
