"""Unit tests for the semantics-driven metric formatting.

One formatting rule set, driven only by the display fields of ``MetricSemantics``, is shared by
the CLI, the HTML report, the reports API and the Web UI.
"""

import pytest
from typing import Optional

from evalscope.api.metric.semantics import (
    MetricDirection,
    MetricDisplayKind,
    MetricIdentity,
    MetricRole,
    MetricSemantics,
    ValueRange,
)
from evalscope.metrics.semantics.formatting import (
    DIAGNOSTIC_FALLBACK_PRECISION,
    MISSING_PLACEHOLDER,
    format_metric_label,
    format_metric_labels,
    format_metric_value,
    is_missing_value,
)

RATIO_RANGE = ValueRange(min=0.0, max=1.0)


def make_percent_semantics(
    multiplier: float = 100.0,
    precision: int = 1,
    unit: Optional[str] = '%',
    role: MetricRole = MetricRole.PRIMARY,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
    value_range: ValueRange = RATIO_RANGE,
) -> MetricSemantics:
    """Build a percent-rendered declaration with overridable display fields."""
    return MetricSemantics(
        semantic_id='quality.accuracy.ratio',
        metric_name='Accuracy',
        role=role,
        direction=direction,
        value_range=value_range,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=multiplier,
        display_unit=unit,
        display_precision=precision,
    )


def make_number_semantics(
    precision: int = 3,
    display_unit: Optional[str] = 's',
    raw_unit: Optional[str] = 's',
    role: MetricRole = MetricRole.PRIMARY,
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER,
) -> MetricSemantics:
    """Build a number-rendered declaration with overridable display fields."""
    return MetricSemantics(
        semantic_id='perf.latency.seconds',
        metric_name='Latency',
        role=role,
        direction=direction,
        raw_unit=raw_unit,
        display_kind=MetricDisplayKind.NUMBER,
        display_unit=display_unit,
        display_precision=precision,
    )


class TestRoundTiesPositive:
    """Rendering must never fall back to banker's rounding.

    Asserted through ``format_metric_value`` at precision 0 rather than against a rounding helper:
    Tie-breaking is a property of the rendered text, which is the only thing production emits.
    The percent and official-scale tie cases live with their own display kind, in
    ``TestFormatPercent``.
    """

    @pytest.mark.parametrize(
        'value, expected', [
            (0.5, '1'),
            (1.5, '2'),
            (2.5, '3'),
            (-0.5, '0'),
            (-2.5, '-2'),
            (12.5, '13'),
        ]
    )
    def test_rounds_halves_toward_positive_infinity(self, value: float, expected: str) -> None:
        semantics = make_number_semantics(
            precision=0, display_unit=None, raw_unit=None, role=MetricRole.DIAGNOSTIC, direction=MetricDirection.NONE
        )
        assert format_metric_value(value, semantics) == expected


class TestIsMissingValue:
    """Only finite numbers are renderable."""

    @pytest.mark.parametrize('value', [None, float('nan'), float('inf'), float('-inf'), 'x', True])
    def test_missing_inputs(self, value: object) -> None:
        assert is_missing_value(value) is True

    @pytest.mark.parametrize('value', [0, 0.0, -1.5, 12, 1e12])
    def test_finite_numbers_are_present(self, value: float) -> None:
        assert is_missing_value(value) is False


class TestFormatPercent:
    """Percent rendering scales by ``display_multiplier`` and glues the unit to the number."""

    def test_ratio_is_scaled_and_rounded(self) -> None:
        assert format_metric_value(0.8567, make_percent_semantics()) == '85.7%'

    def test_official_scale_is_not_rescaled(self) -> None:
        # 87.25 at precision 1 is also the tie case: it must round up, not to even.
        semantics = make_percent_semantics(multiplier=1.0, value_range=ValueRange(min=0.0, max=100.0))
        assert format_metric_value(87.25, semantics) == '87.3%'

    def test_no_space_before_percent_sign(self) -> None:
        assert ' ' not in format_metric_value(0.5, make_percent_semantics())

    def test_trailing_zeros_are_trimmed(self) -> None:
        assert format_metric_value(0.9, make_percent_semantics()) == '90%'
        assert format_metric_value(1.0, make_percent_semantics()) == '100%'
        assert format_metric_value(0.0, make_percent_semantics()) == '0%'

    def test_tie_rounds_toward_positive_infinity(self) -> None:
        assert format_metric_value(0.125, make_percent_semantics(precision=0)) == '13%'


class TestFormatNumber:
    """Number rendering applies declared scaling and separates the unit with one space."""

    def test_unit_follows_a_single_space(self) -> None:
        assert format_metric_value(1.23456, make_number_semantics()) == '1.235 s'

    def test_unit_is_omitted_when_undeclared(self) -> None:
        semantics = make_number_semantics(precision=2, display_unit=None, raw_unit=None)
        assert format_metric_value(7.5, semantics) == '7.5'

    def test_negative_value_rounds_to_nearest(self) -> None:
        semantics = make_number_semantics(precision=2, display_unit=None, raw_unit=None)
        assert format_metric_value(-3.456, semantics) == '-3.46'

    def test_value_below_precision_collapses_to_zero(self) -> None:
        assert format_metric_value(0.0001234, make_number_semantics()) == '0 s'

    def test_negative_value_below_precision_has_no_minus_zero(self) -> None:
        assert format_metric_value(-0.0001234, make_number_semantics()) == '0 s'

    def test_zero_precision_drops_the_decimal_point(self) -> None:
        semantics = make_number_semantics(
            precision=0, display_unit=None, raw_unit=None, role=MetricRole.DIAGNOSTIC, direction=MetricDirection.NONE
        )
        assert format_metric_value(12.0, semantics) == '12'


class TestMissingAndFallback:
    """Missing values and missing semantics both have one defined rendering."""

    @pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
    def test_missing_value_uses_placeholder(self, value: Optional[float]) -> None:
        assert format_metric_value(value, make_percent_semantics()) == MISSING_PLACEHOLDER

    def test_missing_semantics_uses_fallback_precision(self) -> None:
        assert format_metric_value(0.87654321, None) == '0.8765'
        assert DIAGNOSTIC_FALLBACK_PRECISION == 4

    def test_missing_semantics_adds_no_unit(self) -> None:
        assert format_metric_value(250.0, None) == '250'

    def test_missing_semantics_and_missing_value(self) -> None:
        assert format_metric_value(None, None) == MISSING_PLACEHOLDER


class TestMetricLabels:

    def test_scored_metric_uses_display_name_and_direction(self) -> None:
        assert format_metric_label('mean_acc', make_percent_semantics()) == 'Accuracy ↑'

    def test_lower_is_better_uses_down_arrow(self) -> None:
        semantics = make_percent_semantics(direction=MetricDirection.LOWER_IS_BETTER)

        assert format_metric_label('wer', semantics) == 'Accuracy ↓'

    def test_identity_dimensions_use_stable_semantic_order(self) -> None:
        identity = MetricIdentity(
            name='accuracy',
            aggregation='mean',
            dimensions={
                'level': 'overall',
                'target': 'answer',
                'strict': True
            },
        )

        assert format_metric_label(identity, make_percent_semantics()) == 'Accuracy ↑ · Answer · Overall · Yes'

    def test_overlap_variant_precedes_statistic(self) -> None:
        identity = MetricIdentity(
            name='rouge',
            aggregation='mean',
            dimensions={
                'statistic': 'recall',
                'variant': 'l'
            },
        )

        assert format_metric_label(identity, make_percent_semantics()) == 'Accuracy ↑ · L · Recall'

    def test_diagnostic_and_missing_semantics_keep_final_name(self) -> None:
        diagnostic = make_number_semantics(
            role=MetricRole.DIAGNOSTIC, direction=MetricDirection.NONE, display_unit=None, raw_unit=None
        )

        assert format_metric_label('judge_failed', diagnostic) == 'judge_failed'
        assert format_metric_label('unknown_metric', None) == 'unknown_metric'

    def test_duplicate_display_names_are_disambiguated(self) -> None:
        labels = format_metric_labels([
            ('mean_acc', make_percent_semantics()),
            ('mean_fact_acc', make_percent_semantics(role=MetricRole.AUXILIARY)),
        ])

        assert labels == {
            'mean_acc': 'Accuracy ↑ (mean_acc)',
            'mean_fact_acc': 'Accuracy ↑ (mean_fact_acc)',
        }


def test_formatting_reads_only_display_fields() -> None:
    """Role, direction, ranges and identifiers must not influence the rendered text."""
    base = make_percent_semantics()
    diagnostic = make_percent_semantics(role=MetricRole.DIAGNOSTIC, direction=MetricDirection.NONE)
    lower_is_better = make_percent_semantics(direction=MetricDirection.LOWER_IS_BETTER)

    assert format_metric_value(0.4321, base) == format_metric_value(0.4321, diagnostic)
    assert format_metric_value(0.4321, base) == format_metric_value(0.4321, lower_is_better)
