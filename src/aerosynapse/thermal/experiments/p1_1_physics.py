"""Step 1.1 experiment: sanity metrics for nondimensionalisation, analytic solutions and MMS.

Run:  python experiments/p1_1_physics.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.physics import mms
from aerosynapse.thermal.physics import nondim as nd
from aerosynapse.thermal.utils.experiment import Experiment, cli_profile
from aerosynapse.thermal.viz.style import apply_style, save_fig

CFG = {"seed": 0, "Pe_list": [1, 10, 100, 1000], "Bi_list": [0.1, 1.0, 10.0, 100.0], "n_eig": 4}


def figures(xs: np.ndarray, ex: Experiment) -> list[str]:
    """The analytic solutions of Cases A and B (one figure)."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for Pe in CFG["Pe_list"]:
        axs[0].plot(xs[::200], an.steady_convdiff(xs[::200], Pe), label=f"Pe={Pe}")
    axs[0].set(xlabel=r"$x^*$", ylabel=r"$\theta$", title="Case A steady")
    axs[0].legend()
    xg = np.linspace(-0.2, 1.6, 600)
    for tt in [0, 0.4, 0.8, 1.2]:
        axs[1].plot(xg, an.gaussian_pulse(xg, tt, Pe=100.0), label=f"$t^*$={tt}")
    axs[1].set(xlabel=r"$x^*$", ylabel=r"$\theta$", title="Case A pulse, Pe=100")
    axs[1].legend()
    xb = np.linspace(0, 1, 200)
    for tt in [0.01, 0.1, 0.3, 1.0, 5.0]:
        axs[2].plot(xb, an.slab_transient(xb, tt, Bi=1.0, Q=1.0), label=f"$t_F$={tt}")
    axs[2].plot(xb, an.slab_steady(xb, 1.0, 1.0), "k--", label="steady")
    axs[2].set(xlabel=r"$x^*$", ylabel=r"$\theta$", title="Case B slab, Bi=1, Q=1")
    axs[2].legend()
    return [str(p) for p in save_fig(fig, "p1_1_analytic_solutions", ex.dir)]


def main() -> None:
    CFG["profile"] = cli_profile()  # identical work in both profiles (cheap)
    apply_style()
    with Experiment("p1_1_physics", CFG) as ex:
        n, groups = nd.buckingham_pi(["L", "U", "nu", "alpha", "k", "h", "rho", "cp", "dT", "q_vol"])
        ex.metrics["buckingham_n_groups"] = n
        ex.metrics["buckingham_groups"] = [{k: str(v) for k, v in g.items()} for g in groups]

        # steady Case A: exact BCs and max|residual| of the stable formula on a fine grid
        xs = np.linspace(0, 1, 200001)
        res = {}
        for Pe in CFG["Pe_list"]:
            th = an.steady_convdiff(xs, Pe)
            d1 = an.steady_convdiff_dx(xs, Pe)
            # theta_xx = Pe*theta_x exactly for this solution; check the discrete consistency
            d2_fd = np.gradient(d1, xs)
            res[str(Pe)] = {
                "bc0": float(th[0]),
                "bc1": float(th[-1]),
                "rel_resid_interior": float(np.max(np.abs(d1 - d2_fd / Pe)[1:-1]) / np.max(d1)),
            }
        ex.metrics["caseA_steady"] = res

        # Case B eigenvalues
        ex.metrics["caseB_eigenvalues"] = {
            str(Bi): an.robin_eigenvalues(Bi, CFG["n_eig"]).tolist() for Bi in CFG["Bi_list"]
        }

        # MMS symbolic residuals must be identically zero
        for name, ms in [
            ("mms_1d", mms.mms_1d_default(20.0, 1.0, 0.5)),
            ("mms_2d", mms.mms_2d_default(50.0)),
        ]:
            r = sp.simplify(mms.residual_operator(ms.theta, ms.velocity, ms.D, ms.s) - ms.source)
            ex.metrics[f"{name}_symbolic_residual"] = str(r)
        ex.metrics["mms_2d_velocity_divergence"] = str(mms.divergence(mms.mms_2d_default().velocity))

        ex.metrics["figures"] = figures(xs, ex)
    print(f"wrote {ex.dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
