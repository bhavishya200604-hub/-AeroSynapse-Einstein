"""Layer E: scientific validation engine (independent of training loss)."""

from aerosynapse.validation.metrics import error_distribution, field_metrics

__all__ = ["error_distribution", "field_metrics"]
