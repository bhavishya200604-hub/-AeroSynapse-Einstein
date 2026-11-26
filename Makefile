# Optional convenience wrapper (the cross-platform runner is nox; see noxfile.py).
.PHONY: lint typecheck test reproduce figures docs
lint: ; ruff check src tests && ruff format --check src tests
typecheck: ; mypy
test: ; pytest -m "not slow and not gpu"
reproduce: ; python scripts/reproduce_all.py --profile quick
figures: ; aerosynapse figures
docs: ; mkdocs build --strict
