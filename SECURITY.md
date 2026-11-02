# Security policy

AeroSynapse-Einstein is research software (validation tier 1/5, see `docs/VV_REPORT.md`). It is not
intended for, and must not be used in, safety-critical or certified engineering workflows.

To report a vulnerability (e.g. unsafe deserialisation of checkpoints or configs), open a private
GitHub security advisory on the repository. Checkpoints are loaded with `torch.load(weights_only=True)`
and configs with `yaml.safe_load`; do not load files from untrusted sources.
