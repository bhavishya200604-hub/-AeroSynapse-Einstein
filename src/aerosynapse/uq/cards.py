"""Method cards for each UQ method (spec Section 9)."""

UQ_CARDS = {
    "gaussian_process": {
        "estimates": "epistemic uncertainty of the regression function + homoscedastic aleatoric noise",
        "assumptions": "stationary RBF covariance, Gaussian noise, smooth target",
        "calibration": "hyper-parameters by maximum marginal likelihood",
        "evaluation": "coverage curve, PIT histogram, NLL on held-out data",
        "decision_impact": "drives BO acquisition (EI/PoF) and surrogate trust regions",
    },
    "deep_ensemble": {
        "estimates": "epistemic (model) uncertainty via member disagreement",
        "assumptions": "members are diverse; spread correlates with error (checked, not assumed)",
        "calibration": "none intrinsic; optional variance scaling on a calibration split",
        "evaluation": "coverage curve in- and out-of-distribution; error–spread correlation",
        "decision_impact": "trust/fallback threshold for surrogate queries",
    },
    "monte_carlo": {
        "estimates": "parameter-uncertainty propagation through a trusted model",
        "assumptions": "input distribution known; enough samples (standard error reported)",
        "calibration": "n/a (exact in the limit); convergence via standard error",
        "evaluation": "comparison with analytic distribution where available",
        "decision_impact": "robust objectives (expectation, CVaR) and chance constraints",
    },
    "sobol": {
        "estimates": "fraction of output variance due to each input (global sensitivity)",
        "assumptions": "independent inputs; finite variance",
        "calibration": "n/a",
        "evaluation": "Ishigami analytic indices",
        "decision_impact": "which parameters need to be identified / controlled",
    },
    "kalman_filter": {
        "estimates": "state uncertainty under linear-Gaussian model and observation error",
        "assumptions": "linear dynamics, Gaussian errors with known covariances",
        "calibration": "Q and R set from the twin experiment",
        "evaluation": "2-sigma coverage of the truth",
        "decision_impact": "when the state estimate is reliable enough to act on",
    },
    "numerical_uncertainty": {
        "estimates": "discretisation error of reference solutions (stored per dataset sample, GCI)",
        "assumptions": "asymptotic range",
        "calibration": "observed order",
        "evaluation": "exact solutions where available",
        "decision_impact": "floor below which surrogate errors are not meaningful",
    },
    "model_discrepancy": {
        "estimates": "difference between model class and reality (e.g. inviscid vs viscous)",
        "assumptions": "not quantified in this platform (no experimental data); stated qualitatively",
        "calibration": "not available",
        "evaluation": "not available — listed as a limitation",
        "decision_impact": "caps the validation tier",
    },
}
