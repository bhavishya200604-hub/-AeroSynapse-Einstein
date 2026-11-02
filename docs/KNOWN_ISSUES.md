# Known issues, limitations, not-run items

| ID | Area | Issue | Impact | Status |
|---|---|---|---|---|
| KI-1 | numerics | Burgers FV observed order 1.82 at n = 512 (pre-asymptotic; minmod clips at extrema) | order claim "≈2" is asymptotic only | documented |
| KI-2 | numerics | Upwind conv-diff observed order 0.955 at n = 512 (approaching 1 from below) | none | documented |
| KI-3 | numerics | 2D Poisson FD assembly uses a Python loop (slow for n > 300) | performance only | open (P3) |
| KI-4 | infra | GitHub Actions CI configured but not executed in this run | CI badge not shown | open |
| KI-5 | data | Ghia et al. (1982) values come from a fetched third-party transcription, not checked against the original PDF | validation reference uncertainty | documented |
| KI-6 | ml | Learning-rate annealing (α=0.9) diverges for penalty-BC Poisson (rel. L2 0.97); final weights 4.15e8–2.03e9 | adaptive weighting not recommended without capping | negative result, fix proposed |
| KI-7 | ml | conv-diff PINNs with a sharp initial pulse are inaccurate at low Pe (rel. L2 0.33 at Pe=2, σ0=0.05) | see Phase 3 | studied |
