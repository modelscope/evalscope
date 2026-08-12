"""Shared Hypothesis configuration and fixtures for metric semantics tests.

All property tests under ``tests/report/semantics/`` run with a single profile so that every
property is exercised with at least ``MIN_PROPERTY_EXAMPLES`` generated inputs.
"""
import logging
import pytest
from hypothesis import HealthCheck, settings

#: Name of the profile shared by every property test in this package.
SEMANTICS_PROFILE = 'metric_semantics'

#: Minimum number of generated examples per property test.
MIN_PROPERTY_EXAMPLES = 100

settings.register_profile(
    SEMANTICS_PROFILE,
    max_examples=MIN_PROPERTY_EXAMPLES,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(SEMANTICS_PROFILE)


@pytest.fixture
def caplog_evalscope(caplog, monkeypatch):
    """Capture WARN+ from the evalscope logger, which sets ``propagate=False`` by default."""
    monkeypatch.setattr(logging.getLogger('evalscope'), 'propagate', True)
    caplog.set_level(logging.WARNING, logger='evalscope')
    return caplog
