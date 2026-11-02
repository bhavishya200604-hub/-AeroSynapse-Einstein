# Known issues, limitations, not-run items

| ID | Area | Issue | Impact | Status |
|---|---|---|---|---|
| KI-1 | numerics | Burgers FV observed order 1.82 at n = 512 (pre-asymptotic; minmod clips at extrema) | order claim "≈2" is asymptotic only | documented |
| KI-2 | numerics | Upwind conv-diff observed order 0.955 at n = 512 (approaching 1 from below) | none | documented |
| KI-3 | numerics | 2D Poisson FD assembly uses a Python loop (slow for n > 300) | performance only | open (P3) |
| KI-4 | infra | GitHub Actions CI configured but not executed in this run | CI badge not shown | open |
| KI-5 | data | Ghia et al. (1982) values come from a fetched third-party transcription, not checked against the original PDF | validation reference uncertainty | documented |
