# Quickstart (5 minutes, Windows 11 or Linux)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev,torch,ui]"
aerosynapse doctor
aerosynapse run configs/workflow_quick.yaml
streamlit run src/aerosynapse/ui/app.py
```

`aerosynapse run` executes define → simulate → train → validate → optimise → report and writes
`results/workflow_quick/` including `report.html`.

```python
from aerosynapse.physics.library import convdiff_1d
from aerosynapse.numerics.convdiff import solve_convdiff_fv
p = convdiff_1d(pe=50)
r = solve_convdiff_fv(0, 1, 256, 1.0, 1/50, p.initial_condition, 0.5, scheme="vanleer")
print(abs(r.u - p.exact(r.x, 0.5)).max())
```
