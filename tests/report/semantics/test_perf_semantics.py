"""Tests for the perf field semantics.

* ``TestPerfFieldCoverage`` -- every public perf field key has an entry and every
  key comes from the perf name constants.
* ``TestPerfDirections`` -- latency is lower_is_better, throughput is higher_is_better.
* ``TestPerfDiagnostics`` -- counts, cache and failure details carry no direction.
"""

import pytest
from typing import Dict, FrozenSet

from evalscope.api.metric.semantics import MetricDirection, MetricRole
from evalscope.metrics.semantics import attach_perf_semantics, format_perf_value, resolve_perf_semantics
from evalscope.metrics.semantics.perf import PERF_API_ALIASES, PERF_SEMANTICS
from evalscope.metrics.semantics.resolver import SemanticsResolver
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
    """The perf semantics map covers every public field key and only uses perf name constants."""

    def test_every_public_field_is_declared(self) -> None:
        missing = sorted(PUBLIC_FIELD_KEYS - set(PERF_SEMANTICS))

        assert missing == []

    @pytest.mark.parametrize('field_key', sorted(PUBLIC_FIELD_KEYS))
    def test_every_entry_resolves(self, field_key: str) -> None:
        semantics = PERF_SEMANTICS[field_key].resolve(field_key)

        assert semantics.metric_name == field_key


class TestPerfDirections:
    """Latency and throughput must not be confused with each other."""

    @pytest.mark.parametrize('field_key', LOWER_IS_BETTER_FIELDS)
    def test_latency_is_lower_is_better(self, field_key: str) -> None:
        semantics = PERF_SEMANTICS[field_key].resolve(field_key)

        assert semantics.direction is MetricDirection.LOWER_IS_BETTER

    @pytest.mark.parametrize('field_key', HIGHER_IS_BETTER_FIELDS)
    def test_throughput_is_higher_is_better(self, field_key: str) -> None:
        semantics = PERF_SEMANTICS[field_key].resolve(field_key)

        assert semantics.direction is MetricDirection.HIGHER_IS_BETTER


class TestPerfDiagnostics:
    """Counts, cache details and failure details never carry a verdict."""

    @pytest.mark.parametrize('field_key', DIAGNOSTIC_FIELDS)
    def test_field_is_diagnostic(self, field_key: str) -> None:
        semantics = PERF_SEMANTICS[field_key].resolve(field_key)

        assert semantics.role is MetricRole.DIAGNOSTIC
        assert semantics.direction is MetricDirection.NONE


class TestResolvePerfField:
    """The resolver reads the perf table and degrades for anything it does not declare."""

    def test_public_field_resolves(self) -> None:
        resolved = SemanticsResolver().resolve_perf_field(Metrics.AVERAGE_LATENCY)

        assert not resolved.degraded
        assert resolved.semantics.direction is MetricDirection.LOWER_IS_BETTER

    def test_extension_field_degrades(self) -> None:
        resolved = SemanticsResolver().resolve_perf_field('Some Vendor Extension')

        assert resolved.degraded
        assert resolved.semantics.role is MetricRole.DIAGNOSTIC
        assert 'PERF_SEMANTICS' in '\n'.join(resolved.audit_messages)


class TestPerfKeySpaces:
    """Perf numbers resolve through stable constants and API paths.

    A response declares semantics under the identifier it exposes the value by. Getting this wrong
    is silent: the consumer looks a key up, misses, and the metric loses its direction and unit
    without any error.
    """

    def test_stable_key_spaces_share_one_registry(self) -> None:
        for field_key in (Metrics.AVERAGE_LATENCY, 'best_rps'):
            assert field_key in PERF_SEMANTICS

    def test_non_constant_key_spaces_are_declared_as_aliases(self) -> None:
        # Asserted as containment, not as a copy of the table: a new alias is a legitimate edit, while
        # an alias pointing at nothing declared is the silent failure worth catching.
        assert set(PERF_API_ALIASES) <= set(PERF_SEMANTICS)

    @pytest.mark.parametrize(
        'field_key,expected_semantic_id',
        [
            # Perf constants, used by the percentile and summary JSON.
            (Metrics.AVERAGE_LATENCY, 'perf.latency.seconds'),
            # Stable API paths, used by in-report perf and the run list.
            ('best_rps', 'perf.throughput.requests_per_second'),
        ],
    )
    def test_every_key_space_resolves(self, field_key: str, expected_semantic_id: str) -> None:
        resolved = resolve_perf_semantics([field_key])

        assert field_key in resolved, f'{field_key!r} did not resolve'
        assert resolved[field_key]['semantic_id'] == expected_semantic_id

    @pytest.mark.parametrize('field_key', ['success_rate'])
    def test_success_rate_is_already_a_percentage(self, field_key: str) -> None:
        # The perf pipeline formats it as `87.5%`, so scaling it again would render `8750%`.
        semantics = resolve_perf_semantics([field_key])[field_key]

        assert semantics['display_multiplier'] == 1.0
        assert semantics['display_unit'] == '%'
        assert semantics['value_range'] == {'min': 0.0, 'max': 100.0}

    @pytest.mark.parametrize('field_key', ['ttft', 'tpot'])
    def test_in_report_streaming_latency_converts_seconds_to_milliseconds(self, field_key: str) -> None:
        semantics = resolve_perf_semantics([field_key])[field_key]

        assert semantics['raw_unit'] == 's'
        assert semantics['display_multiplier'] == 1000.0
        assert semantics['display_unit'] == 'ms'

    def test_perf_formatter_uses_registry_precision_and_can_omit_repeated_units(self) -> None:
        assert format_perf_value(1.23456, Metrics.AVERAGE_LATENCY) == '1.235 s'
        assert format_perf_value(1.23456, Metrics.AVERAGE_LATENCY, include_unit=False) == '1.235'
        assert format_perf_value(0.7, Metrics.APPROX_SPECULATIVE_ACCEPTANCE_RATE) == '70%'

    def test_report_payload_persists_the_semantics_it_displays(self) -> None:
        payload = attach_perf_semantics({
            'summary': {
                'n_samples': 2,
                'latency': {},
                'throughput': {
                    'avg_output_tps': 12.5,
                    'avg_req_ps': 1.5,
                },
                'usage': {
                    'input_tokens': {},
                    'output_tokens': {},
                    'total_tokens': {},
                },
                'ttft': {},
            }
        })

        assert payload['metric_semantics']['ttft']['display_unit'] == 'ms'
        assert payload['metric_semantics']['throughput.avg_output_tps']['display_unit'] == 'tok/s'
