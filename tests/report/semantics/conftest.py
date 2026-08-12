"""Shared Hypothesis configuration for metric semantics property tests.

All property tests under ``tests/report/semantics/`` run with a single profile so that every
property is exercised with at least ``MIN_PROPERTY_EXAMPLES`` generated inputs.
"""
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
