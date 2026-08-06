"""Tests for the perf field semantics.

* ``TestPerfFieldCoverage`` -- Property 21: every public perf field key has an entry and every
  key comes from the perf name constants.
* ``TestPerfDirections`` -- latency is lower_is_better, throughput is higher_is_better.
* ``TestPerfDiagnostics`` -- counts, cache and failure details carry no direction.
"""

import pytest
from typing import Dict, FrozenSet

from evalscope.api.metric.semantics import MetricDirection, MetricRole
from evalscope.metrics.semantics.perf import PERF_FIELD_SEMANTICS
from evalscope.metrics.semantics.resolver import SemanticsResolver, is_public_perf_field
from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics


def _constant_values(holder: type) -> FrozenSet[str]:
    """Return the public string constants declared on a perf constants holder."""
    return frozenset(
        value for name, value in vars(holder).items() if not name.startswith('_') and isinstance(value, str)
    )


PUBLIC_FIELD_KEYS: FrozenSet[str] = _constant_values(Metrics) | _constant_values(PercentileMetrics)

#: Fields whose value gets smaller as the system gets better.
LOWER_IS_BETTER_FIELDS = (
    Metrics.AVERAGE_LATENCY,
    Metrics.AVERAGE_TIME_TO_FIRST_TOKEN,
    Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN,
    Metrics.AVERAGE_INTER_TOKEN_LATENCY,
    Metrics.AVERAGE_FIRST_TURN_TTFT,
    Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT,
    PercentileMetrics.TTFT,
    PercentileMetrics.ITL,
    PercentileMetrics.TPOT,
    PercentileMetrics.LATENCY,
)

#: Fields whose value gets bigger as the system gets better.
HIGHER_IS_BETTER_FIELDS = (
    Metrics.REQUEST_THROUGHPUT,
    Metrics.OUTPUT_TOKEN_THROUGHPUT,
    Metrics.TOTAL_TOKEN_THROUGHPUT,
    Metrics.INPUT_TOKEN_THROUGHPUT,
    PercentileMetrics.OUTPUT_THROUGHPUT,
    PercentileMetrics.INPUT_THROUGHPUT,
    PercentileMetrics.TOTAL_THROUGHPUT,
    PercentileMetrics.DECODE_THROUGHPUT,
)

#: Fields that describe the run instead of grading it.
DIAGNOSTIC_FIELDS = (
    Metrics.TOTAL_REQUESTS,
    Metrics.SUCCEED_REQUESTS,
    Metrics.FAILED_REQUESTS,
    Metrics.STREAM_REQUESTS,
    Metrics.NON_STREAM_REQUESTS,
    Metrics.NUMBER_OF_CONCURRENCY,
    Metrics.AVERAGE_CACHED_PERCENT,
    Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST,
    Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST,
    PercentileMetrics.INPUT_TOKENS,
    PercentileMetrics.OUTPUT_TOKENS,
)


class TestPerfFieldCoverage:
    """Feature: metric-semantics-governance, Property 21: the perf semantics map covers every
    public field key, and every declared key is one of the perf name constants."""

    def test_every_public_field_is_declared(self) -> None:
        missing = sorted(PUBLIC_FIELD_KEYS - set(PERF_FIELD_SEMANTICS))

        assert missing == []

    def test_no_declared_key_is_invented(self) -> None:
        unexpected = sorted(set(PERF_FIELD_SEMANTICS) - PUBLIC_FIELD_KEYS)

        assert unexpected == []

    @pytest.mark.parametrize('field_key', sorted(PERF_FIELD_SEMANTICS))
    def test_every_entry_resolves(self, field_key: str) -> None:
        semantics = PERF_FIELD_SEMANTICS[field_key].resolve(field_key)

        assert semantics.metric_name == field_key

    def test_public_fields_are_strict_for_the_resolver(self) -> None:
        for field_key in PUBLIC_FIELD_KEYS:
            assert is_public_perf_field(field_key)

    def test_extension_field_is_not_strict(self) -> None:
        assert not is_public_perf_field('Some Vendor Extension')


class TestPerfDirections:
    """Latency and throughput must not be confused with each other."""

    @pytest.mark.parametrize('field_key', LOWER_IS_BETTER_FIELDS)
    def test_latency_is_lower_is_better(self, field_key: str) -> None:
        semantics = PERF_FIELD_SEMANTICS[field_key].resolve(field_key)

        assert semantics.direction is MetricDirection.LOWER_IS_BETTER

    @pytest.mark.parametrize('field_key', HIGHER_IS_BETTER_FIELDS)
    def test_throughput_is_higher_is_better(self, field_key: str) -> None:
        semantics = PERF_FIELD_SEMANTICS[field_key].resolve(field_key)

        assert semantics.direction is MetricDirection.HIGHER_IS_BETTER


class TestPerfDiagnostics:
    """Counts, cache details and failure details never carry a verdict."""

    @pytest.mark.parametrize('field_key', DIAGNOSTIC_FIELDS)
    def test_field_is_diagnostic(self, field_key: str) -> None:
        semantics = PERF_FIELD_SEMANTICS[field_key].resolve(field_key)

        assert semantics.role is MetricRole.DIAGNOSTIC
        assert semantics.direction is MetricDirection.NONE
        assert semantics.comparison_group is None


class TestResolvePerfField:
    """The resolver reads the perf table and degrades only for extension fields."""

    def test_public_field_resolves(self) -> None:
        resolved = SemanticsResolver().resolve_perf_field(Metrics.AVERAGE_LATENCY)

        assert not resolved.degraded
        assert resolved.semantics.direction is MetricDirection.LOWER_IS_BETTER

    def test_extension_field_degrades_without_blocking(self) -> None:
        resolved = SemanticsResolver().resolve_perf_field('Some Vendor Extension')

        assert resolved.degraded
        assert not resolved.blocks_standard_semantics
        resolved.raise_if_blocked()

    def test_undeclared_public_field_blocks(self) -> None:
        resolved = SemanticsResolver(perf_fields={}).resolve_perf_field(Metrics.AVERAGE_LATENCY)

        assert resolved.blocks_standard_semantics
        assert 'PERF_FIELD_SEMANTICS' in '\n'.join(resolved.audit_messages)
