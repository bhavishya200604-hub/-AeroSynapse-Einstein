import json, sys, time
sys.path.insert(0, "src")
from aerosynapse.cad import kernel
from aerosynapse.cfd import runner
from aerosynapse.sim import flow3d
from aerosynapse.workflow.stages import reference
d = json.load(open("projects/falcon/08_export/design.json"))
L = kernel.info(d, h=0.1)["length"]
ref = reference(d, L)
out = "results/cfd3d/tiny"
num = runner.Numerics(cells_per_length=20, steps=250, max_cells=2.4e6, residual_drop=1e-5, save_volume=True, full_model=True)
runner.run(d, runner.Flight(mach=0.85, alpha=4.0, altitude=11000), ref, out, num)
import pathlib; pathlib.Path(out, "run_info.json").write_text(json.dumps({"case": "cruise", "quality": "tiny", "flight": {"mach": 0.85, "alpha": 4}, "reference": ref}))
t = time.time(); r = flow3d.postprocess(out); print("post s", time.time() - t)
print(open(out + "/summary.txt").read())
