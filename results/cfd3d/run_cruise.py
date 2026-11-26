import json, sys
sys.path.insert(0, "src")
from aerosynapse.cad import kernel
from aerosynapse.cfd import runner
from aerosynapse.workflow.stages import reference
d = json.load(open("projects/falcon/08_export/design.json"))
L = kernel.info(d, h=0.1)["length"]
ref = reference(d, L)
num = runner.Numerics(cells_per_length=L / 0.32, steps=2500, max_cells=2.4e6, residual_drop=1e-5, save_volume=True, full_model=True)
r = runner.run(d, runner.Flight(mach=0.85, alpha=2.0, altitude=11000), ref, "results/cfd3d/falcon_cruise_full", num)
print(json.dumps({k: r.get(k) for k in ("cells", "seconds", "half_model", "coefficients")}, default=str))
