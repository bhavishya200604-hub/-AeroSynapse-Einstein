# Hardware and storage report

Measured on 2026-11-11 with `Get-CimInstance` (Win32_Processor, Win32_ComputerSystem, Win32_VideoController,
Win32_LogicalDisk) and `Get-PhysicalDisk`. The tables are measurements; the consequences further down are derived
from them and from timed runs, and each says which.

## Machine

| Item | Measured value |
|---|---|
| CPU | AMD Ryzen 3 3250U, 2 cores, 4 logical processors, 2.6 GHz |
| RAM | 5.9 GB total; 0.7 GB free at the time of measurement (desktop app, Studio and editor running) |
| GPU | AMD Radeon integrated graphics, 2 GB shared memory; **no CUDA**, no `nvidia-smi` |
| OS | Windows 11 Home Single Language 10.0.26200 |
| PyTorch | 2.8.0.dev (CPU build); `torch.cuda.is_available()` is False; 2 threads |
| Python | 3.12.8 in `.venv` (numpy 2.1.3, scipy 1.15.1, scikit-learn 1.6.1, NeuralFoil 0.3.3, AeroSandbox 4.2.10) |
| C++ | MinGW g++ 6.3, 32-bit, flags `-O3 -march=native -static` |

## Disks

| Drive | Device | Size | Free | Holds |
|---|---|---|---|---|
| C: | WD PC SN530 NVMe SSD | 237 GB | 49.4 GB | Windows, `C:\AeroSynapse_Data` (created, empty) |
| D: | Toshiba MQ04ABF100 HDD (SATA, 1 TB, two partitions) | 467 GB | 433 GB | free |
| E: | same HDD | 365 GB | 322 GB | this repository and its `results/` |

## What this hardware can and cannot support

These are consequences of the measurements, stated so that plans stay honest.

* **CFD.** The flow solver measured 0.34 s per step on 132,600 cells (draft grid, 4 threads), so about 2.6 us per
  cell-step (timed). A standard Falcon run (331,000 cells) took 45 min (timed). By that rate a fine grid of about
  1 million cells and 3,000 steps would take about 2 h per case (derived). The solver is a 32-bit build, so one
  process cannot use more than 2 GB; it needs about 0.34 kB per cell (232 MB peak at 685,000 cells, measured in the
  mesh study of 2026-11-11), so about 5 million cells fit. Measured cost per step: 2.67 s at 685,000 cells (full
  model, 68 cells per length), 0.87 s at 343,000 (half model). The 8-grid mesh study took about 4.5 h. A half model at
  96 cells per length (about 1 million cells) would take roughly 1.5 h (derived).
* **Optimisation with CFD in the loop.** One settled draft run takes about 8 min (full model). A 14-run Bayesian
  study is about 2 h; a population method such as CMA-ES with hundreds of CFD evaluations is not feasible here.
  CFD-in-the-loop work must use sample-efficient methods (Gaussian-process Bayesian optimisation) or cheaper
  models in the loop with CFD as the judge.
* **Machine learning.** No GPU. Training is CPU-only with 2 cores and under 6 GB of RAM. Neural operators
  (FNO, DeepONet) on 2D fields of a few hundred samples are feasible in hours; anything at the scale of
  pretrained foundation models is not. Model sizes in this project are therefore limited by measured hardware,
  not by an arbitrary policy.
* **Memory.** With 0.7 GB free during interactive work, large arrays must be memory-mapped or streamed. The
  existing fighter dataset (`data_store/fighter_rows`) is already sharded.
* **Storage.** 49 GB free on the SSD is enough for datasets of a few GB and for CFD volumes (a draft volume is
  about 3 MB; a fine one about 25 MB). The HDD has 750 GB free and suits archives and backups, not hot data.

## Storage root

`C:\AeroSynapse_Data` was created with the requested sub-folders (datasets, raw_data, processed_data,
simulation_cache, CFD_results, FEM_results, model_checkpoints, pretrained_models, trained_models,
optimization_runs, experiment_logs, tensorboard, temporary, backups, reports). Nothing has been moved there yet:
the code still writes to `results/` in the repository. Pointing runs at the storage root is a roadmap item
(R-9) so that it is done with a free-space check, not silently.

## Rules adopted

* Before a disk-heavy job, check free space on the target drive; stop launching new jobs below 10 GB free on C:.
* Never delete results to make space without asking; temporary and reproducible files are listed first.
* Paid cloud compute is never started without the owner's approval.
