"""AirfRANS (the Kaggle graph data in the project folder) as pages of Studio's Labs window: the dataset, the trained
surrogates' results, a prediction for any held-out airfoil, and training again."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st  # the native stand-in inside Studio (nativelab.shim)

ROOT = Path(__file__).resolve().parents[3]
RUNS = {
    "quick": ROOT / "results" / "airfrans_graph",
    "full": ROOT / "results" / "airfrans_graph_full",
    "smoke": ROOT / "results" / "airfrans_graph_smoke",
}


def _best() -> Path:
    """The trained run whose PointNet has the lowest mean error on the held-out full meshes."""
    best, err = RUNS["quick"], float("inf")
    for name in ("quick", "full"):
        r = RUNS[name] / "result.json"
        if r.exists():
            try:
                e = json.loads(r.read_text())["results"]["pointnet"]["test_full_mesh"]
                m = sum(e.values()) / len(e)
            except (KeyError, ValueError, ZeroDivisionError):
                continue
            if m < err:
                best, err = RUNS[name], m
    return best


OUT = _best()


def _G():
    from aerosynapse.ml import airfrans_graph as G

    return G


def page_dataset() -> None:
    G = _G()
    st.header("AirfRANS graph data (Kaggle download in the project folder)")
    files = G.graph_files()
    st.metric("simulation files", len(files))
    st.metric("folders", len({sid.split("/")[0] for sid, _ in files}))
    st.markdown(G.__doc__.split("**What is trained**")[0].replace("``", ""))
    meta = OUT / "simulations.json"
    if not meta.exists():
        st.info("Not extracted yet: use 'Train the surrogates' (it reads every file once, about 10 minutes).")
        return
    import matplotlib.pyplot as plt
    import numpy as np

    m = json.loads(meta.read_text())
    sims = m["simulations"]
    inl = np.array([s["inlet"] for s in sims])
    n = np.array([s["n_nodes"] for s in sims])
    st.metric("simulations extracted", len(sims))
    st.metric("duplicates skipped", m.get("duplicates_skipped", 0))
    st.metric("mesh nodes (median)", int(np.median(n)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(inl[:, 0], inl[:, 1], s=8, c=n, cmap="viridis")
    ax[0].set(
        xlabel="inlet u (scaled)", ylabel="inlet v (scaled)", title="The 1000 flow conditions (colour: mesh nodes)"
    )
    ax[1].hist(n, 40, color="#2563eb")
    ax[1].set(xlabel="mesh nodes", ylabel="simulations", title="Mesh sizes")
    st.pyplot(fig)


def page_results() -> None:
    st.header("Trained surrogates: held-out results")
    r = OUT / "result.json"
    if not r.exists():
        st.info("No trained surrogates yet: use 'Train the surrogates'.")
        return
    res = json.loads(r.read_text())
    st.caption(
        f"{res['simulations']} simulations: {res['split']['train']} train, {res['split']['val']} validation, {res['split']['test']} held out; profile {res['profile']}, {res['seconds'] / 60:.0f} min"
    )
    import pandas as pd

    rows = []
    for k, v in res["results"].items():
        for part in ("test_sampled", "test_full_mesh", "test_surface"):
            rows.append(
                {
                    "model": k,
                    "where": part.replace("test_", "").replace("_", " "),
                    **{f: (f"{100 * e:.2f} %" if e is not None else "n/a at the wall") for f, e in v[part].items()},
                }
            )
    st.subheader("Relative L2 error on the held-out airfoils")
    st.dataframe(pd.DataFrame(rows))
    for k, v in res.get("models", {}).items():
        st.metric(f"{k}: parameters", v["parameters"])
        st.metric(f"{k}: training time", f"{v['train_s'] / 60:.1f} min")
    from PIL import Image

    for f in res["figures"]:
        st.subheader(f.split("_", 1)[1].rsplit(".", 1)[0].replace("_", " "))
        st.image(Image.open(OUT / f))


def page_predict() -> None:
    G = _G()
    st.header("Predict a held-out airfoil (every mesh node)")
    if not (OUT / "model_mlp.pt").exists():
        st.info("No trained surrogates yet: use 'Train the surrogates'.")
        return
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np
    import torch

    meta = json.loads((OUT / "simulations.json").read_text())
    sims = meta["simulations"]
    sp = G.split(len(sims))
    test = [sims[int(i)]["id"] for i in sp["test"]]
    sid = st.selectbox("Held-out simulation (folder/file)", test, key="af_sim")
    kind = st.radio("Model", ["pointnet", "mlp"], key="af_model")
    field = st.selectbox("Field", G.FIELDS, index=2, key="af_field")
    ck = torch.load(OUT / f"model_{kind}.pt", weights_only=True)
    m = G.PointMLP() if kind == "mlp" else G.PointNetFlow()
    m.load_state_dict(ck["state"])
    m.eval()

    class _D:
        mu, sd = ck["mu"].numpy(), ck["sd"].numpy()

    s = next(i for i, x in enumerate(sims) if x["id"] == sid)
    g = G.read_graph(ROOT / sims[s]["file"])
    p = G.predict(m, _D, g["x"], g["pos"])
    e = G.rel_l2(p, g["y"])
    cols = st.columns(4)
    for c, f, v in zip(cols, G.FIELDS, e, strict=True):
        c.metric(f"{f} error", f"{100 * v:.2f} %")
    j = G.FIELDS.index(field)
    keep = (g["pos"][:, 0] > -0.6) & (g["pos"][:, 0] < 1.8) & (np.abs(g["pos"][:, 1]) < 0.6)
    tri = mtri.Triangulation(g["pos"][keep, 0], g["pos"][keep, 1])
    fig, ax = plt.subplots(3, 1, figsize=(11, 12))
    t, q = g["y"][keep, j], p[keep, j]
    lo, hi = np.percentile(t, [1, 99])
    for a, z, ttl in ((ax[0], t, "AirfRANS (RANS)"), (ax[1], q, f"{kind} prediction")):
        im = a.tricontourf(tri, z, 50, cmap="viridis", vmin=lo, vmax=hi)
        fig.colorbar(im, ax=a)
        a.set_aspect("equal")
        a.set_title(f"{field}: {ttl}")
    r = np.percentile(np.abs(q - t), 99)
    im = ax[2].tricontourf(tri, q - t, 50, cmap="coolwarm", vmin=-r, vmax=r)
    fig.colorbar(im, ax=ax[2])
    ax[2].set_aspect("equal")
    ax[2].set_title(f"{field}: prediction - RANS")
    st.pyplot(fig)
    surf = g["x"][:, 2] <= g["x"][:, 2].min() + 1e-7
    o = np.argsort(g["pos"][surf, 0])
    fig, a = plt.subplots(figsize=(10, 4.5))
    a.plot(g["pos"][surf, 0][o], g["y"][surf, 2][o], "k.", ms=3, label="RANS")
    a.plot(g["pos"][surf, 0][o], p[surf, 2][o], "r.", ms=3, label=kind)
    a.invert_yaxis()
    a.set(xlabel="x / chord", ylabel="p (scaled)", title="Wall pressure")
    a.legend()
    st.pyplot(fig)


def page_train() -> None:
    G = _G()
    st.header("Train the surrogates on the AirfRANS graph data")
    prof = st.radio(
        "Profile",
        ["quick", "full", "smoke"],
        format_func=lambda p: {
            "quick": "quick (20 epochs, about 1.5 h)",
            "full": "full (50 epochs)",
            "smoke": "smoke (40 simulations, 3 epochs)",
        }[p],
        key="af_prof",
    )
    again = st.checkbox("Read the files again (otherwise the extracted points are reused)", value=False, key="af_again")
    st.caption(
        f"Results go to {OUT}. Every file is read with the safe loader; the 1000 simulations are split 80/10/10 by simulation."
    )
    if not st.button("Train now", key="af_go"):
        return
    import time

    t0 = time.time()
    bar = st.progress(0.0, "starting")
    lines: list[str] = []

    def log(s: str) -> None:
        lines.append(s)
        bar.progress(min(0.99, len(lines) / 70), s.strip()[:120])

    out = RUNS[prof]
    steps = ("extract", "train", "evaluate") if again or not (out / "data.npz").exists() else ("train", "evaluate")
    res = G.run(out, prof, steps, log=log)
    st.success(f"finished at {time.strftime('%H:%M:%S')} after {(time.time() - t0) / 60:.1f} min")
    st.code("\n".join(lines[-40:]))
    st.json(res["results"])


PAGES = {
    "Dataset": page_dataset,
    "Results": page_results,
    "Predict a held-out airfoil": page_predict,
    "Train the surrogates": page_train,
}
