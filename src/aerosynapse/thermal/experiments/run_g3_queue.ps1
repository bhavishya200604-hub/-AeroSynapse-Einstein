# Step 6B.2b, G3 Amendment A1 (paper/PREREG_G1_G3.md; D133): the remaining G3 runs as a queue.
#   batch A (in parallel): registered Graetz seeds 1-2 (2 workers) + slab extension seeds 0-4 full / smoke (1 worker)
#   batch B (after A):     exploratory graetz_rad8 seeds 0-2 (3 workers)
#   then:                  the G3 analysis (registered test, extension, exploratory)
# At most 3 trainings run at a time (memory); every run writes results/<exp_id>/ and its own log on E: (HDD).
# Start detached (survives the closing of the terminal):
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-File','experiments\run_g3_queue.ps1'
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$status = "results_g3_queue.log"

function Log($m) { "$(Get-Date -Format s) $m" | Out-File -FilePath $status -Append -Encoding utf8 }

function Start-Run($argsList, $log) {
    Start-Process -FilePath $py -ArgumentList $argsList -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
}

Log "queue start"
$a1 = Start-Run @("-m", "experiments.p4_7_ablation", "--config", "results/p4_7_ablation_smoke/config.yaml",
    "--profile", "full", "--set", '"ablation_study.only=[full-exact_psi]"', "--set", '"ablation_study.seeds=[1, 2]"',
    "--tag", "g3_full_s12") "results_g3_s12.log"
$a2 = Start-Run @("-m", "experiments.p6b_g3_budget", "--parts", "slab,slab_smoke", "--reuse",
    "--set", '"execution.workers=1"') "results_g3_slab_ext.log"
Log "batch A started: graetz seeds 1-2 (pid $($a1.Id)), slab extension (pid $($a2.Id))"
$a2.WaitForExit(); Log "slab extension finished (exit $($a2.ExitCode))"
$a1.WaitForExit(); Log "graetz seeds 1-2 finished (exit $($a1.ExitCode))"

$b = Start-Run @("-m", "experiments.p4_7_ablation", "--config", "results/p4_7_ablation_smoke/config.yaml",
    "--profile", "full", "--set", '"ablation_study.only=[full-exact_psi]"', "--set", '"ablation_study.seeds=[0, 1, 2]"',
    "--set", '"sampling.adaptive.n_candidates=32768"', "--set", '"execution.workers=3"',
    "--tag", "g3amend_rad8") "results_g3_rad8.log"
Log "batch B started: graetz_rad8 seeds 0-2 (pid $($b.Id))"
$b.WaitForExit(); Log "graetz_rad8 finished (exit $($b.ExitCode))"

$c = Start-Run @("-m", "experiments.p6b_g3_budget", "--analysis-only") "results_g3_analysis.log"
$c.WaitForExit(); Log "analysis finished (exit $($c.ExitCode)); queue done"
