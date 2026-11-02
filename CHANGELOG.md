# Changelog
All notable changes are documented here (Keep a Changelog; semantic versioning).

## [Unreleased]
### Added
- Phase 0: package scaffold (`src/aerosynapse`), pydantic configs, experiment records (Section 13
  fields), plugin registry, exception hierarchy, device/hardware report, safety ladder, CLI `doctor`,
  CI/pre-commit/Docker/devcontainer, planning document A–N.
- Phase 1: physics definition API, BC framework (Dirichlet/Neumann/Robin/periodic/initial/interface),
  Level-1 problem library with exact solutions; FD heat 1D/2D, FV conv-diff (upwind/central/QUICK/
  van Leer), Burgers FV, wave leapfrog, Poisson FD 1D/2D, P1 FEM, adaptive dt, 1D AMR, verification
  toolkit (order, Richardson, GCI, von Neumann), solver cards.
