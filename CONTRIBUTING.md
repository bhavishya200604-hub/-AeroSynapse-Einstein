# Contributing

1. Install: `pip install -e ".[dev,torch]"` (Windows 11 or Linux, Python >= 3.10).
2. Every change needs tests. Numerical code needs a *verification* test (manufactured or analytical
   solution with an observed order of accuracy), not only a smoke test.
3. Run `nox -s lint typecheck tests` (or `ruff check src tests && mypy && pytest`) before pushing.
4. Never weaken a tolerance to make a test pass; fix the code or document the limitation in
   `docs/KNOWN_ISSUES.md`.
5. Every reported number must come from `results/<experiment_id>/` and be regenerable by a script.
6. New physics modules / solvers / models / optimisers register through `aerosynapse.registry`
   (see `docs/guides/adding_a_physics_module.md`); do not edit core code to add one.
7. Use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:` ...).
