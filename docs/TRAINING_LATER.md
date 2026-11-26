# Training and runs left for later

Everything below is implemented and tested, but was **not run** in the build environment (network policy,
compute or missing tools). No numbers from these runs appear anywhere in the repository.

## 1. AirfRANS RANS-coefficient study (drag data; KI-13)
The build environment's proxy refused `www.kaggle.com` and `api.kaggle.com` (403), also from a fresh session.
Run it where Kaggle is reachable: your laptop, or a Kaggle/Colab notebook.

**Option A — Kaggle/Colab notebook (no local disk needed):** open `notebooks/airfrans_kaggle.ipynb`, run all
cells. It downloads AirfRANS with `kagglehub`, installs the package and runs the study.

**Option B — local (Windows 11 or Linux):**
```bash
pip install -e ".[torch]" airfrans kagglehub
```
```bash
python -c "import kagglehub; print(kagglehub.dataset_download('alexandrepers/airfrans'))"
```
```bash
python scripts/run_airfrans.py --root <folder containing manifest.json> --max-sims 100
```
Drop `--max-sims` for all simulations. Outputs: `results/airfrans_coefficients/` (record, metrics) and
`data_derived/airfrans_coefficients.csv` (derived C_D/C_L table with the ODbL-1.0 attribution; commit this,
**not** the raw dataset — GitHub limits files to 100 MB). Then write the summary from `metrics.json` only and run
`python scripts/audit_claims.py`.

What it computes (see `src/aerosynapse/ml/airfrans.py`): per-simulation C_D/C_L via the official `airfrans`
API; an MLP surrogate (log Re, AoA, NACA parameters) → (C_D, C_L) with an explicit convergence criterion
(validation-loss plateau) on the official full / scarce / Reynolds / AoA splits; and the inviscid panel-method
C_L against RANS C_L (model discrepancy).

## 2. Full-profile reruns (GPU / Colab)
```bash
python scripts/reproduce_all.py --profile full
```
or per phase via `notebooks/phase{2..6}_full_colab.ipynb`. More seeds and iterations; expected to take hours.
Quick-profile results are the only ones reported.

## 3. OpenFOAM (execution mode D)
Install OpenFOAM, then `aerosynapse.numerics.external.write_cavity_case(...)`, run `blockMesh && icoFoam`,
and import the centre line with `import_line_csv` for a cross-check against `results/phase6_fluids_aero`.

## 4. LaTeX PDF
No TeX toolchain in the build environment. `cd paper && latexmk -pdf main.tex` (sources validated by
`python scripts/check_paper.py`).
