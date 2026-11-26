"""Phase 2: generate datasets from USER-DEFINABLE conditions (defaults in configs/data/defaults.yaml).

    python experiments/p2_data.py --profile smoke
    python experiments/p2_data.py --profile smoke --config configs/data/examples/hot_bottom_wall.yaml --tag hotbottom
    python experiments/p2_data.py --set cases.caseC.grid=48 --set noise.levels=[0,0.05] --set cases.caseA.enabled=false
    python experiments/p2_data.py --describe --config my.yaml        # print resolved conditions, generate nothing

Layering: defaults < --config files (in order) < profile blocks < --set overrides. Every run writes the fully resolved spec to
data/datasets/<tag>/resolved_spec.yaml (and into results/p2_data_<tag>/config.yaml) so any dataset is reproducible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from aerosynapse.thermal.data import design as D
from aerosynapse.thermal.data import generators as G
from aerosynapse.thermal.data import sensors as S
from aerosynapse.thermal.data import spec as SP
from aerosynapse.thermal.data.dataset_io import read_dataset, sha256, write_dataset
from aerosynapse.thermal.data.noise import add_noise
from aerosynapse.thermal.data.normalisation import fit_normaliser
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.style import DIV_CMAP, SEQ_CMAP, apply_style, save_fig

SPLIT_COLORS = {"train": "#0072B2", "val": "#009E73", "interp": "#E69F00", "extrap": "#D55E00"}


def param_table(des: D.CaseDesign, fixed: dict[str, float]) -> dict[str, np.ndarray]:
    n = des.u.shape[0]
    tab = {d.name: des.params[:, k] for k, d in enumerate(des.dims)}
    tab.update({k: np.full(n, v) for k, v in fixed.items()})
    return tab


def generate(name: str, des: D.CaseDesign, fixed: dict, ccfg: dict) -> tuple[dict, dict, dict, dict]:
    t0 = time.perf_counter()
    tab = param_table(des, fixed)
    if name in ("caseA", "caseB"):
        o = (G.gen_case_a if name == "caseA" else G.gen_case_b)(tab["Pe" if name == "caseA" else "Bi"], ccfg)
        out = ({"x": o["x"], "t": o["t"]}, {"theta": o["theta"], "theta_steady": o["theta_steady"]}, {})
    else:
        tab.setdefault("Ri", np.zeros(des.u.shape[0]))
        o = G.gen_case_c(tab, ccfg)
        out = (
            {"x": o["x"], "y": o["y"], "t": o["t"]},
            {k: o[k] for k in ("theta", "u", "v", "p")},
            {k: o[k] for k in ("nu_top", "nu_bottom", "ke", "steps", "max_div", "wall_s")},
        )
    return (*out, {"gen_wall_s": time.perf_counter() - t0})


def sensor_file(path: Path, name: str, ds, spec: dict, seed: int) -> dict:
    is2d = name == "caseC"
    sc, levels, model = spec["sensors"], spec["noise"]["levels"], spec["noise"]["model"]
    layouts = sc["layouts"] or (list(S.LAYOUTS_2D) if is2d else list(S.LAYOUTS_1D))
    layouts = [lay for lay in layouts if lay in (S.LAYOUTS_2D if is2d else S.LAYOUTS_1D)]
    counts = sc["counts_2d"] if is2d else sc["counts_1d"]
    idx = ds.split_idx("interp")[: sc["n_points"]]
    n_written = 0
    with h5py.File(path, "w") as f:
        f.attrs.update(
            {
                "case": name,
                "levels": json.dumps(levels),
                "noise_model": model,
                "note": "series at ALL snapshot times; noise std per model definition; seed = f(point, layout, n, level)",
            }
        )
        for pi in idx:
            for li, lay in enumerate(layouts):
                for n in counts:
                    pts = (
                        S.layout_2d(lay, n, seed, sc["margin_2d"])
                        if is2d
                        else S.layout_1d(lay, n, seed, sc["margin_1d"])
                    )
                    clean = (
                        S.sample_2d(ds.coords["x"], ds.coords["y"], ds.fields["theta"][pi], pts)
                        if is2d
                        else S.sample_1d(ds.coords["x"], ds.fields["theta"][pi], pts)
                    )
                    g = f.create_group(f"point_{int(pi)}/{lay}_n{len(pts)}")
                    g.create_dataset("xy", data=pts)
                    g.create_dataset("clean", data=clean)
                    g.create_dataset(
                        "noisy",
                        data=np.stack(
                            [
                                add_noise(clean, lv, model, 10_000 * int(pi) + 100 * li + 10 * len(pts) + k)
                                for k, lv in enumerate(levels)
                            ]
                        ),
                    )
                    n_written += 1
    return {
        "path": str(path),
        "groups": n_written,
        "points": [int(i) for i in idx],
        "layouts": layouts,
        "counts": counts,
    }


def resolution_check(ccfg: dict, des: D.CaseDesign, fixed: dict) -> dict:
    """Numerical-data accuracy: re-run two train points at 2n and compare restricted fields at all snapshots."""
    out = {}
    times = G.snapshot_times(ccfg["times"])
    n = ccfg["grid"]
    tab = param_table(des, fixed)
    tab.setdefault("Ri", np.zeros(des.u.shape[0]))
    for pi in des.idx("train")[:2]:
        pt = {k: float(v[pi]) for k, v in tab.items()}
        a = G.gen_case_c_point(pt, times, ccfg, n)
        b = G.gen_case_c_point(pt, times, ccfg, 2 * n)
        rec = {"params": pt, "n": n}
        for fld in ("theta", "u", "v"):
            fine = G.restrict_cell_average(b[fld], 2)
            rec[f"relL2_{fld}"] = float(np.linalg.norm(a[fld] - fine) / max(np.linalg.norm(fine), 1e-300))
        rec["nu_top_final"] = [float(a["nu_top"][-1]), float(b["nu_top"][-1])]
        rec["nu_top_rel_diff"] = abs(rec["nu_top_final"][0] - rec["nu_top_final"][1]) / max(
            abs(rec["nu_top_final"][1]), 1e-300
        )
        out[f"train_point_{int(pi)}"] = rec
    return out


def plots(cases: dict, tag: str, ex: Experiment) -> list[str]:
    paths = []
    fig, axs = plt.subplots(
        1, 1 + 2 * ("caseC" in cases and cases["caseC"][0].d >= 2), figsize=(12, 3.6), squeeze=False
    )
    axs = axs[0]
    row = 0
    ticks, labels = [], []
    for name in ("caseA", "caseB"):
        if name in cases:
            des = cases[name][0]
            for s in D.SPLITS:
                i = des.idx(s)
                axs[0].plot(
                    des.params[i, 0],
                    np.full(i.size, float(row)),
                    "o",
                    color=SPLIT_COLORS[s],
                    ms=5,
                    label=s if row == 0 else None,
                )
            ticks.append(row)
            labels.append(f"{name}: {des.dims[0].name}")
            row += 1
    if ticks:
        axs[0].set(yticks=ticks, yticklabels=labels, xlabel="parameter value", title="1D cases")
        axs[0].set_xscale("log" if all(cases[n][0].dims[0].is_log for n in cases if n != "caseC") else "linear")
        axs[0].legend(fontsize=7, ncol=2)
    if "caseC" in cases and cases["caseC"][0].d >= 2:
        dc = cases["caseC"][0]
        pairs = [(0, 1)] + ([(0, 2)] if dc.d >= 3 else [(1, 0)])
        for ax, (a, b) in zip(axs[1:], pairs, strict=False):
            for s in D.SPLITS:
                i = dc.idx(s)
                ax.plot(dc.params[i, a], dc.params[i, b], "o", color=SPLIT_COLORS[s], ms=5, label=s)
            ax.set(
                xlabel=dc.dims[a].name, ylabel=dc.dims[b].name, title=f"Case C: {dc.dims[a].name} vs {dc.dims[b].name}"
            )
            ax.set_xscale("log" if dc.dims[a].is_log else "linear")
            ax.set_yscale("log" if dc.dims[b].is_log else "linear")
            ax.axvline(float(dc.dims[a].to_phys(np.array(dc.u_lim))), color="k", ls=":", lw=0.8)
            ax.axhline(float(dc.dims[b].to_phys(np.array(dc.u_lim))), color="k", ls=":", lw=0.8)
    paths += save_fig(fig, f"p2_parameter_space_{tag}", ex.dir)

    nplot = sum(k in cases for k in ("caseA", "caseB"))
    if nplot:
        fig, axs = plt.subplots(1, nplot, figsize=(5.5 * nplot, 3.4), squeeze=False)
        j = 0
        for name in ("caseA", "caseB"):
            if name not in cases:
                continue
            des, ds = cases[name]
            order = np.argsort(des.params[:, 0])
            for pi in order[[0, len(order) // 2, -1]]:
                (ln,) = axs[0, j].plot(
                    ds.coords["x"], ds.fields["theta_steady"][pi], label=f"{des.dims[0].name}={des.params[pi, 0]:.3g}"
                )
                axs[0, j].plot(ds.coords["x"], ds.fields["theta"][pi, -1], "--", color=ln.get_color())
            axs[0, j].set(xlabel="$x^*$", ylabel=r"$\theta$", title=f"{name}: steady (solid), last snapshot (dashed)")
            axs[0, j].legend(fontsize=7)
            j += 1
        paths += save_fig(fig, f"p2_fields_1d_{tag}", ex.dir)

    if "caseC" in cases:
        des, ds = cases["caseC"]
        order = np.argsort(des.params[:, 0])
        pick = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
        fig, axs = plt.subplots(2, 3, figsize=(11, 6.6))
        xs, ys = ds.coords["x"], ds.coords["y"]
        for j, pi in enumerate(pick):
            th = ds.fields["theta"][pi, -1]
            im = axs[0, j].contourf(xs, ys, th, 20, cmap=SEQ_CMAP)
            axs[0, j].contour(
                xs, ys, np.cumsum(ds.fields["u"][pi, -1], axis=0) * (ys[1] - ys[0]), 12, colors="w", linewidths=0.5
            )
            tt = ", ".join(f"{d.name}={des.params[pi, k]:.3g}" for k, d in enumerate(des.dims))
            axs[0, j].set(aspect="equal", title=f"{tt} ({D.SPLITS[des.split[pi]]})", xlabel="$x^*$", ylabel="$y^*$")
            fig.colorbar(im, ax=axs[0, j], label=r"$\theta$", shrink=0.8)
            pr = ds.fields["p"][pi, -1]
            lim = max(float(np.max(np.abs(pr))), 1e-30)
            im = axs[1, j].pcolormesh(xs, ys, pr, cmap=DIV_CMAP, vmin=-lim, vmax=lim)
            axs[1, j].set(aspect="equal", title="pressure (zero-mean)", xlabel="$x^*$", ylabel="$y^*$")
            fig.colorbar(im, ax=axs[1, j], shrink=0.8)
        paths += save_fig(fig, f"p2_fields_cavity_{tag}", ex.dir)

    k1 = next((n for n in ("caseA", "caseB") if n in cases), None)
    if k1:
        fig, axs = plt.subplots(
            1, 3 if "caseC" in cases else 1, figsize=(12 if "caseC" in cases else 5, 3.4), squeeze=False
        )
        axs = axs[0]
        des, ds = cases[k1]
        pi = des.idx("interp")[0] if des.idx("interp").size else des.idx("train")[0]
        x, f = ds.coords["x"], ds.fields["theta_steady"][pi]
        axs[0].plot(x, f, "k", lw=1)
        for lay, m in (("uniform", "o"), ("clustered_right", "s")):
            xs_ = S.layout_1d(lay, 16)
            axs[0].plot(xs_, S.sample_1d(x, f, xs_), m, ms=4, mfc="none", label=lay)
        xs_ = S.layout_1d("clustered_right", 16)
        for lv, c in ((0.05, "r"), (0.10, "m")):
            axs[0].plot(
                xs_,
                add_noise(S.sample_1d(x, f, xs_), lv, "gaussian", 0),
                ".",
                color=c,
                ms=4,
                label=f"noisy {int(lv * 100)}%",
            )
        axs[0].set(xlabel="$x^*$", ylabel=r"$\theta$", title=f"1D sensors ({k1} steady) and noise")
        axs[0].legend(fontsize=6)
        if "caseC" in cases:
            for ax, lay in zip(axs[1:], ("sobol", "wall_clustered"), strict=True):
                pts = S.layout_2d(lay, 64)
                ax.plot(pts[:, 0], pts[:, 1], "o", ms=3)
                ax.set(
                    aspect="equal",
                    xlim=(0, 1),
                    ylim=(0, 1),
                    title=f"2D layout: {lay} (64)",
                    xlabel="$x^*$",
                    ylabel="$y^*$",
                )
        paths += save_fig(fig, f"p2_sensors_noise_{tag}", ex.dir)
    return [str(q) for q in paths]


def case_text(name: str, ccfg: dict, dims, fixed) -> dict:
    sw = ", ".join(f"{d.name} ∈ [{d.lo:g}, {d.hi:g}] ({d.scale})" for d in dims)
    fx = ", ".join(f"{k} = {v:g}" for k, v in fixed.items())
    c = ccfg["conditions"]
    if name == "caseA":
        phys = f"1D convection–diffusion; pulse {c['pulse']}; steady θ(0)={c['steady']['theta_left']}, θ(1)={c['steady']['theta_right']}"
    elif name == "caseB":
        phys = f"1D slab (Fourier time), θ_x(0)=0, Robin at x=1; {c['slab']}"
    else:
        vel = {
            w: v["profile"] + (f"×{v.get('amplitude', 1.0):g}" if v["profile"] != "zero" else "")
            for w, v in c["velocity"].items()
        }
        th = {
            w: (
                t["type"]
                + (f"={t.get('value', 0)}" if t["type"] != "robin" else f"(Bi={t['Bi']}, θ∞={t.get('theta_inf', 0)})")
            )
            for w, t in c["thermal"].items()
        }
        phys = f"2D cavity; wall velocity {vel}; thermal {th}; θ0={c.get('initial_theta', 0)}; scheme {ccfg['scheme']}"
    n = ccfg["nx"] if name != "caseC" else ccfg["grid"]
    return {
        "physics": phys,
        "param_ranges": sw + (f"; fixed: {fx}" if fx else ""),
        "n_snapshots": len(G.snapshot_times(ccfg["times"])),
        "grid": f"{n} pts" if name != "caseC" else f"{n}² cells",
    }


def write_card(m: dict, tag: str, path: Path) -> None:
    L = [
        f"# Dataset card — `{tag}` (auto-generated by experiments/p2_data.py; every number is from results/p2_data_{tag}/metrics.json)",
        "",
        f"git: `{m['_git']}`. Conditions: `{m['spec_file']}` (fully resolved, user-editable layers: defaults < config < profile < --set).",
        "",
        "## Cases (conditions as run)",
        "",
        "| Case | Conditions | Parameters | Ground truth | Snapshots | Grid |",
        "|---|---|---|---|---|---|",
    ]
    for c, v in m["cases"].items():
        L.append(
            f"| {c} | {v['physics']} | {v['param_ranges']} | {v['ground_truth']} | {v['n_snapshots']} | {v['grid']} |"
        )
    L += [
        "",
        "## Splits (by PARAMETER point)",
        "",
        "| Case | train | val | interp | extrap | train box | min cross-split dist | leakage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c, v in m["cases"].items():
        n = v["split_counts"]
        L.append(
            f"| {c} | {n['train']} | {n['val']} | {n['interp']} | {n['extrap']} | {v['train_box']} | "
            f"{v['leakage']['min_cross_split_dist']:.4f} (≥ {v['leakage']['threshold']:.4f}) | {'PASS' if v['leakage']['all_ok'] else 'FAIL'} |"
        )
    L += ["", "## Files (git-ignored)", "", "| File | MB | sha256 |", "|---|---|---|"]
    for v in m["files"].values():
        L.append(f"| `{v['path']}` | {v['bytes'] / 1e6:.2f} | `{v['sha256'][:16]}…` |")
    L += [
        "",
        "## Normalisation / noise / sensors",
        "",
        "Inputs mapped to [-1, 1] by FIXED design bounds; field z-scores from the TRAIN split only. "
        f"Noise model `{m['noise_model']}`, levels {m['noise_levels']}; sensor series for the first {m['sensor_points']} interpolation points.",
        "",
    ]
    if m["resolution_check"]:
        L += ["## Numerical accuracy of the Case C data", ""]
        for k, v in m["resolution_check"].items():
            L.append(
                f"- {k}: rel. L2 between {v['n']}² and {2 * v['n']}² (restricted), all snapshots — θ {v['relL2_theta']:.3e}, "
                f"u {v['relL2_u']:.3e}, v {v['relL2_v']:.3e}; Nu_top(final) {v['nu_top_final'][0]:.4f} vs {v['nu_top_final'][1]:.4f}."
            )
    L += [
        "",
        "## Limitations",
        "",
        "- Case C is numerical (discretisation error quantified above). Case A pulse = infinite-line solution restricted to [0,1].",
        "- Extrapolation set = parameter extrapolation beyond the training box, laminar only.",
        "- Case D (heated channel/obstacle) is not part of this dataset.",
    ]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures_only(spec: dict, tag: str) -> list[str]:
    """Phase 7: the figures of a generated dataset, re-drawn from its HDF5 files without writing to data/."""
    from types import SimpleNamespace

    src = ROOT / "data" / "datasets" / tag
    if (src / "resolved_spec.yaml").exists():  # the conditions the dataset was generated with
        spec = yaml.safe_load((src / "resolved_spec.yaml").read_text(encoding="utf-8"))
    cases = {}
    for name, ccfg in spec["cases"].items():
        f = src / f"{name}.h5"
        if ccfg.get("enabled", True) and f.exists():
            ds = read_dataset(f)
            cases[name] = (ds.design, ds)
    if not cases:
        raise SystemExit(f"no datasets in {src}; generate them first (without --figures-only)")
    out = plots(cases, tag, SimpleNamespace(dir=ROOT / "results" / f"p2_data_{tag}"))
    print("\n".join(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[], help="user YAML (repeatable; later wins)")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="override, e.g. cases.caseC.grid=48")
    ap.add_argument("--tag", default=None, help="output folder tag (default: profile)")
    ap.add_argument("--describe", action="store_true", help="print resolved conditions and exit")
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="re-draw the figures from the existing datasets in data/datasets/<tag> (read only; nothing regenerated)",
    )
    a = ap.parse_args()
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    if a.describe:
        print(SP.describe(spec))
        print(yaml.safe_dump(spec, sort_keys=False))
        return
    tag = a.tag or a.profile
    custom = bool(a.config or a.sets)
    apply_style()
    if a.figures_only:
        figures_only(spec, tag)
        return
    out_dir = ROOT / "data" / "datasets" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    print(SP.describe(spec), flush=True)
    with Experiment(f"p2_data_{tag}", spec) as ex:
        cases, files, summary = {}, {}, {}
        for ci, (name, ccfg) in enumerate(spec["cases"].items()):
            if not ccfg.get("enabled", True):
                continue
            dims, fixed = SP.case_dims(ccfg, name)
            des = D.design_case(
                name,
                dims,
                ccfg["sizes"],
                spec["seed"] + ci,
                spec["design"]["unit_train_limit"],
                spec["design"]["method"],
                spec["design"]["min_dist_factor"],
            )
            coords, fields, scalars, info = generate(name, des, fixed, ccfg)
            files[name] = write_dataset(
                out_dir / f"{name}.h5", des, coords, fields, scalars, {"tag": tag, "case_cfg": ccfg}
            )
            ds = read_dataset(out_dir / f"{name}.h5")
            cb = {"x": (0.0, 1.0), "t": (float(coords["t"][0]), float(coords["t"][-1]))}
            if "y" in coords:
                cb["y"] = (0.0, 1.0)
            nrm = fit_normaliser(ds.design, cb, dict(ds.fields))
            nrm.save(out_dir / f"normalisation_{name}.json")
            lk = D.check_leakage(ds.design)
            assert lk["all_ok"], f"leakage check failed for {name}: {lk}"
            min_cross = min([v for k, v in lk.items() if k.startswith("min_dist_")] or [float("inf")])
            sens = sensor_file(out_dir / f"sensors_{name}.h5", name, ds, spec, spec["seed"])
            files[f"sensors_{name}"] = {
                "path": sens["path"],
                "bytes": Path(sens["path"]).stat().st_size,
                "sha256": sha256(sens["path"]),
            }
            sp_ = ds.design.params
            summary[name] = {
                **case_text(name, ccfg, dims, fixed),
                "ground_truth": "analytic" if name != "caseC" else "numerical (MAC NS + FV energy)",
                "n_points": int(sp_.shape[0]),
                "split_counts": {s: int((ds.design.split == i).sum()) for i, s in enumerate(D.SPLITS)},
                "train_box": ", ".join(
                    f"{d.name} ≤ {float(d.to_phys(np.array(ds.design.u_lim))):.4g}" for d in ds.design.dims
                ),
                "param_min": sp_.min(axis=0).tolist(),
                "param_max": sp_.max(axis=0).tolist(),
                "extrap_distance_max": float(ds.design.extrap_distance().max()),
                "leakage": {**lk, "min_cross_split_dist": min_cross, "threshold": ds.design.min_dist},
                "field_stats_train": nrm.field_stats,
                "gen": info,
                "sensors": sens,
            }
            if name == "caseC":
                summary[name]["numerics"] = {
                    "max_div_over_all_runs": float(np.max(scalars["max_div"])),
                    "total_steps": int(np.sum(scalars["steps"])),
                    "solver_wall_s_total": float(np.sum(scalars["wall_s"])),
                    "theta_min": float(ds.fields["theta"].min()),
                    "theta_max": float(ds.fields["theta"].max()),
                }
            cases[name] = (ds.design, ds)
            ex.log(case=name, n_points=int(sp_.shape[0]), gen_wall_s=info["gen_wall_s"], bytes=files[name]["bytes"])
            print(
                f"[{name}] {sp_.shape[0]} pts, {info['gen_wall_s']:.1f}s, {files[name]['bytes'] / 1e6:.2f} MB, leakage OK",
                flush=True,
            )
        res = {}
        if "caseC" in cases:
            dims, fixed = SP.case_dims(spec["cases"]["caseC"], "caseC")
            res = resolution_check(spec["cases"]["caseC"], cases["caseC"][0], fixed)
        ex.metrics.update(
            {
                "cases": summary,
                "files": files,
                "resolution_check": res,
                "noise_levels": spec["noise"]["levels"],
                "noise_model": spec["noise"]["model"],
                "sensor_points": spec["sensors"]["n_points"],
                "_git": git_hash(),
                "spec_file": str(out_dir / "resolved_spec.yaml"),
            }
        )
        ex.metrics["figures"] = plots(cases, tag, ex)
        card = (out_dir / "DATASET_CARD.md") if custom else (ROOT / "docs" / "DATASET_CARD.md")
        write_card(ex.metrics, tag, card)
        ex.metrics["dataset_card"] = str(card)
    for k, v in res.items():
        print(k, {x: (round(y, 5) if isinstance(y, float) else y) for x, y in v.items()})
    for c, v in summary.items():
        print(c, v["split_counts"], f"min cross-split dist {v['leakage']['min_cross_split_dist']:.4f}")


if __name__ == "__main__":
    main()
