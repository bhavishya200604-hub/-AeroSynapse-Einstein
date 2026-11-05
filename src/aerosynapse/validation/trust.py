r"""Trust and fallback logic (spec Section 3: *when should an engineer trust the model?*).

A :class:`TrustPolicy` decides, per query, whether a surrogate prediction may be used or the
classical solver must be called. Checks (each logged with its value and threshold):

1. **domain** — parameters inside the training ranges (hard OOD check);
2. **input novelty** — Mahalanobis distance of input features to the training set, threshold at the
   training-set ``q`` quantile (soft OOD check);
3. **uncertainty** — ensemble spread relative to the prediction norm below a threshold;
4. **physics residual** — a physics-consistency residual (e.g. mass-conservation error) below a
   threshold.
Every decision is appended to a JSON-lines log. Thresholds are set on a *calibration split*, never
on the test data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.safety import DISCLAIMER

FloatArr = NDArray[np.float64]


@dataclass
class Decision:
    use_surrogate: bool
    reasons: list[str]
    checks: dict[str, dict[str, float | bool]]


@dataclass
class TrustPolicy:
    param_lo: FloatArr
    param_hi: FloatArr
    feat_mean: FloatArr
    feat_icov: FloatArr
    maha_threshold: float
    unc_threshold: float
    residual_threshold: float
    log_path: Path | None = None
    log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def calibrate(
        cls,
        params: FloatArr,
        feats: FloatArr,
        rel_spread: FloatArr,
        residual: FloatArr,
        q: float = 0.99,
        log_path: Path | None = None,
    ) -> TrustPolicy:
        mu = feats.mean(0)
        cov = np.cov(feats.T) + 1e-9 * np.eye(feats.shape[1])
        icov = np.linalg.inv(cov)
        d = np.sqrt(np.einsum("ij,jk,ik->i", feats - mu, icov, feats - mu))
        return cls(
            params.min(0),
            params.max(0),
            mu,
            icov,
            float(np.quantile(d, q)),
            float(np.quantile(rel_spread, q)),
            float(np.quantile(residual, q)),
            log_path,
        )

    def decide(
        self, params: FloatArr, feat: FloatArr, rel_spread: float, residual: float, query_id: str = ""
    ) -> Decision:
        maha = float(np.sqrt((feat - self.feat_mean) @ self.feat_icov @ (feat - self.feat_mean)))
        checks: dict[str, dict[str, float | bool]] = {
            "domain": {"ok": bool(np.all(params >= self.param_lo) and np.all(params <= self.param_hi))},
            "input_novelty": {"value": maha, "threshold": self.maha_threshold, "ok": maha <= self.maha_threshold},
            "uncertainty": {
                "value": rel_spread,
                "threshold": self.unc_threshold,
                "ok": rel_spread <= self.unc_threshold,
            },
            "physics_residual": {
                "value": residual,
                "threshold": self.residual_threshold,
                "ok": residual <= self.residual_threshold,
            },
        }
        reasons = [k for k, v in checks.items() if not v["ok"]]
        dec = Decision(not reasons, reasons, checks)
        entry = {"query": query_id, **asdict(dec)}
        self.log.append(entry)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=float) + "\n")
        return dec


def guarded_predict(
    policy: TrustPolicy,
    surrogate: Callable[[], tuple[FloatArr, float, float]],
    solver: Callable[[], FloatArr],
    params: FloatArr,
    feat: FloatArr,
    query_id: str = "",
) -> tuple[FloatArr, str, Decision]:
    """Return (prediction, source, decision): the surrogate only if every check passes."""
    pred, spread, residual = surrogate()
    d = policy.decide(params, feat, spread, residual, query_id)
    if d.use_surrogate:
        return pred, "surrogate", d
    return solver(), "classical_solver", d


def trust_report_markdown(name: str, sections: dict[str, dict[str, Any]], tier: str) -> str:
    lines = [f"# Trust report — {name}", "", f"> {DISCLAIMER}", "", f"**Validation tier of this model:** {tier}", ""]
    for sec, vals in sections.items():
        lines += [f"## {sec}", "", "| quantity | value |", "|---|---|"]
        for k, v in vals.items():
            vv = f"{v:.4g}" if isinstance(v, float) else str(v)
            lines.append(f"| {k} | {vv} |")
        lines.append("")
    return "\n".join(lines)
