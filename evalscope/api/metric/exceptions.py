"""Failures that make a metric unavailable rather than an incorrect verdict."""


class MetricUnavailableError(RuntimeError):
    """Invalid scoring inputs or execution failures must remain excluded from recall."""
