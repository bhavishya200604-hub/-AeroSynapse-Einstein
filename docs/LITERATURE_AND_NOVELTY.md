# Literature review, verification log and novelty assessment

Verification policy: an entry is **VERIFIED** only if its existence (title + identifier) was fetched
during this run, and the "what was verified" column says exactly what. Author lists marked "not
verified" were not confirmed (the arXiv API was reachable only partially). **UNVERIFIED** entries are
standard references recalled from memory. They are named in docstrings for orientation, are **not**
in `paper/references.bib`, and none of their numbers is used.

## 1. Verified references

| Key | Work | Identifier | What was verified (how) |
|---|---|---|---|
| raissi2017pinn1 | Physics Informed Deep Learning (Part I) | arXiv:1711.10561 | title, abstract (research search) |
| lu2021deepxde | DeepXDE (introduces RAR) | arXiv:1907.04502 | title, abstract incl. RAR (research search) |
| wu2023sampling | Comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs (RAD, RAR-D; >6000 PINN runs) | arXiv:2207.10289 | title, abstract |
| daw2022r3 | R3 sampling (Daw, Bu, Wang, Perdikaris, Karpatne) | arXiv:2207.02338 | title, abstract, authors (arXiv API) |
| krishnapriyan2021failure | Characterizing possible failure modes in PINNs | arXiv:2109.01050 | title, abstract (convection failure, curriculum, seq2seq) |
| wang2021gradient | Gradient pathologies in PINNs (learning-rate annealing) | arXiv:2001.04536 | title, abstract; Algorithm-1 formula and recommended α = 0.9 (PDF fetched) |
| wang2022causal | Respecting causality is all you need | arXiv:2203.07404 | title, abstract |
| tancik2020fourier | Fourier features | arXiv:2006.10739 | title, abstract |
| li2021fno | Fourier Neural Operator | arXiv:2010.08895 | title, abstract |
| lu2021deeponet | DeepONet (Lu, Jin, Karniadakis) | arXiv:1910.03193; DOI 10.1038/s42256-021-00302-5 | title, abstract, authors, DOI (arXiv API) |
| lakshminarayanan2017ensembles | Deep ensembles | arXiv:1612.01474 | title, abstract |
| zhang2023asm | Residual/gradient-based adaptive sampling for sharp solutions | arXiv:2302.08035 | title, abstract |
| lpinn2022 | Lagrangian PINNs | arXiv:2205.02902 | title, abstract |
| lossweight2025 | Loss weight and model complexity for PINNs in CFD (incl. high-Pe conv-diff) | arXiv:2509.21393 | title, abstract |
| ghia1982 | Ghia, Ghia, Shin, J. Comput. Phys. 48(3) 387–411 | — | Tables I–II via a third-party transcription (two GitHub gists), stored with provenance in `src/aerosynapse/data/ghia1982.json`; not checked against the original PDF |
| schafer1996 | Schäfer & Turek, DFG cylinder benchmark | Notes Numer. Fluid Mech. 52 | 2D-1 reference C_D = 5.57953523384 from the Featflow benchmark page (`data/cylinder_dfg2d1.json`) |
| takamoto2022pdebench | PDEBench | arXiv:2210.07182; DOI 10.18419/darus-2986 | data licence CC BY 4.0 (DaRUS page), code MIT (README) |
| bonnet2022airfrans | AirfRANS | OpenReview Zp8YmiQ_bDC | licence ODbL-1.0 (GitHub, OpenReview, PyG docs); `airfrans` API (readthedocs) |
| ohana2024well | The Well | arXiv:2412.00568 | existence, size, access; licence **not** verified |

Also checked (search hits, not cited in the paper): FBOAL arXiv:2212.11776; RANG arXiv:2205.01051;
GLF sampling arXiv:2510.24026.

## 2. Unverified (memory-only) references — orientation only

Roache's grid convergence index (1994); Hess & Smith panel method (1967); Kulfan CST (2008); Deb et al.
NSGA-II (2002); Jones et al. EGO / expected improvement (1998); Saltelli et al. (2010) and Jansen Sobol
estimators; Ishigami & Homma (1990); Shu & Osher SSP-RK3 (1988); van Leer limiter (1979); Leonard QUICK
(1979); Chorin projection (1968); Cole (1951) / Hopf (1950); Blasius (1908); Rockafellar & Uryasev CVaR
(2000); Rasmussen & Williams GPML (2006); Abbott & von Doenhoff (NACA sections); the Prandtl–Glauert rule;
flow-regime thresholds in `physics/regime.py`.
Where these supply a formula, the formula is verified *by our own tests*: the Ishigami indices against Monte
Carlo, the KKT multipliers against the analytic QP, the Blasius f''(0) computed by shooting, NACA thickness
checked, and the Hess–Smith result checked against the exact cylinder C_p.

## 3. What the literature already establishes (relevant to RQ Q1)

* Residual-based adaptive sampling (RAR, RAD, RAR-D, R3, ASM, FBOAL, RANG) improves PINN accuracy on many
  benchmarks, including sharp-gradient problems (wu2023sampling, daw2022r3, zhang2023asm).
* PINNs fail on convection-dominated problems because the optimisation landscape is hard to train, not
  because the network is too small (krishnapriyan2021failure). Remedies include curriculum and
  sequence-to-sequence training, Lagrangian reformulation (lpinn2022), causal weighting (wang2022causal),
  and adaptive weights (wang2021gradient).
* Loss weighting for high-Pe convection–diffusion has been studied (lossweight2025).

## 4. Novelty assessment (verdict)

The verdict is written after Phase 3; it depends on the measured outcome summarised in
`docs/phases/PHASE_3_SUMMARY.md`.

**Verdict: not novel enough for a standalone methods paper.** The methods (RAR, RAD, R3, LHS, Sobol,
Fourier features, causal weighting) are all established. Adaptive collocation for convection-dominated
PINNs has been studied before (§3). What this project can claim honestly is:

1. A **controlled, reproducible re-examination**: one transient conv-diff family with an *exact* reference,
   equal collocation budgets and iterations, paired seeds, Holm-corrected tests, compute accounting and
   negative results. This is useful as a benchmark or workshop contribution, or as a thesis chapter.
2. **Engineering integration** (verified solvers, independent validation, trust/fallback, the 8G
   protocol with trusted re-verification) as research software. This would suit a software venue such as
   JOSS once tests and documentation mature, not a methods journal.

To become a publishable research contribution, the work would need a mechanism-level explanation of when
residual-based sampling helps (e.g. NTK or residual-spectrum measurements), at least one new method that
beats the strongest verified baselines under equal compute, and evaluation on community benchmarks
(PDEBench) rather than self-generated data only.
