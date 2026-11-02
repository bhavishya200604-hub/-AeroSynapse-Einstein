# Known issues, limitations, not-run items

| ID | Area | Issue | Impact | Status |
|---|---|---|---|---|
| KI-1 | numerics | Burgers FV observed order 1.82 at n = 512 (pre-asymptotic; minmod clips at extrema) | order claim "≈2" is asymptotic only | documented |
| KI-2 | numerics | Upwind conv-diff observed order 0.955 at n = 512 (approaching 1 from below) | none | documented |
| KI-3 | numerics | 2D Poisson FD assembly used a Python loop | performance only | resolved: vectorised sparse assembly (n = 256 assembled and solved in about 0.6 s on the build machine) |
| KI-4 | infra | GitHub Actions CI | — | resolved: CI run 14 (https://github.com/bhavishya200604-hub/-AeroSynapse-Einstein/actions/runs/41487440211) passed on Ubuntu and Windows (Python 3.13, CPU torch) plus the strict docs build; earlier runs failed on a torch pin and torch-version-dependent typing, both fixed |
| KI-5 | data | Ghia et al. (1982) values come from a fetched third-party transcription, not checked against the original PDF | validation reference uncertainty | documented |
| KI-6 | ml | Learning-rate annealing (α=0.9) diverges for penalty-BC Poisson (rel. L2 0.97); final weights 4.15e8–2.03e9 | adaptive weighting not recommended without capping | negative result, fix proposed |
| KI-7 | ml | conv-diff PINNs with a sharp initial pulse are inaccurate at low Pe (rel. L2 0.33 at Pe=2, σ0=0.05) | see Phase 3 | studied |
| KI-8 | study | Adaptive samplers (RAR/RAD/R3) give no significant gain at equal budget for Pe ≥ 100; slightly worse at Pe = 1 | negative result for the core hypothesis | reported |
| KI-9 | ml | PINN fails at Pe = 1 (initial diffusion layer, early-time rel. L2 ≈ 0.54) for every sampler | stiffness failure | diagnosed; remedy proposed, not run |
| KI-10 | optim | Gradient descent (fixed step) fails on Rosenbrock; Adam not converged in 3000 evaluations | method limitation | reported |
| KI-11 | optim | Thermal and source demos have corner optima (bounds active); the robust and deterministic optima coincide | robust-optimisation demo uninformative | reported |
| KI-12 | inverse | PINN α estimate 24 % off (adjoint LSQ accurate) | PINN inverse route not reliable at this budget | reported |
| KI-13 | data | AirfRANS not downloaded (kaggle.com blocked by the environment network policy) | RANS-data surrogate and drag study not run | open — needs network access or a Kaggle/Colab run. 2026-11-02: retried from a new session: still blocked: curl `https://www.kaggle.com/api/v1/datasets/download/alexandrepers/airfrans` -> `curl: (56) CONNECT tunnel failed, response 403` (http_code 000); kagglehub -> `ProxyError: HTTPSConnectionPool(host='api.kaggle.com', port=443) ... Tunnel connection failed: 403 Forbidden` |
| KI-14 | fluids | Cylinder uses a staircase immersed boundary (first-order geometry) | C_D error −2.1 % at h = 0.01 | documented |
| KI-15 | fluids | Marching BL vs Blasius max error ≈ 1 %, not reduced by refining y (first-order x-marching dominates) | accuracy floor | documented |
