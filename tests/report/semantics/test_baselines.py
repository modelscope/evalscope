"""Unit tests for the common metric semantics baseline table.

Feature: metric-semantics-governance
"""
import pytest
from typing import List

from evalscope.api.metric.semantics import (
    METRIC_CONTRACT_VERSION,
    MetricDirection,
    MetricDisplayKind,
    MetricRole,
    MetricSemantics,
    lookup_baseline,
)
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES

#: Baselines the catalog and the legacy mapping are allowed to reference (requirement 2.1).
REQUIRED_BASELINE_IDS: List[str] = [
    'quality.accuracy.ratio',
    'quality.f1.ratio',
    'quality.precision.ratio',
    'quality.recall.ratio',
    'quality.exact_match.ratio',
    'quality.pass_at_k.ratio',
    'quality.wer.ratio',
    'quality.cer.ratio',
    'quality.score.points_100',
    'quality.judge_score.unbounded',
    'perf.latency.seconds',
    'perf.latency.milliseconds',
    'perf.throughput.tokens_per_second',
    'perf.throughput.requests_per_second',
    'diagnostic.count.items',
    'diagnostic.parse_status.ratio',
    'diagnostic.unspecified',
]

BASELINE_IDS = sorted(SEMANTIC_BASELINES)
DIAGNOSTIC_IDS = sorted(
    baseline_id for baseline_id, semantics in SEMANTIC_BASELINES.items() if semantics.role == MetricRole.DIAGNOSTIC
)


class TestBaselineTableShape:

    @pytest.mark.parametrize('baseline_id', REQUIRED_BASELINE_IDS)
    def test_required_baseline_is_declared(self, baseline_id: str) -> None:
        assert baseline_id in SEMANTIC_BASELINES

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_key_equals_semantic_id(self, baseline_id: str) -> None:
        assert SEMANTIC_BASELINES[baseline_id].semantic_id == baseline_id

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_baseline_revalidates_against_the_contract(self, baseline_id: str) -> None:
        semantics = SEMANTIC_BASELINES[baseline_id]

        # Re-running the model validators proves the declaration satisfies R1-R5 and that a
        # serialized baseline can be rebuilt from report.json without loss.
        assert MetricSemantics.model_validate(semantics.model_dump()) == semantics
        assert semantics.contract_version == METRIC_CONTRACT_VERSION

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_semantic_id_uses_the_declared_domains(self, baseline_id: str) -> None:
        assert baseline_id.split('.')[0] in {'quality', 'perf', 'diagnostic'}

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_baseline_is_resolvable_through_the_contract_layer(self, baseline_id: str) -> None:
        assert lookup_baseline(baseline_id) is SEMANTIC_BASELINES[baseline_id]


class TestDiagnosticBaselines:

    def test_diagnostic_baselines_exist(self) -> None:
        assert DIAGNOSTIC_IDS

    @pytest.mark.parametrize('baseline_id', DIAGNOSTIC_IDS)
    def test_diagnostic_baseline_carries_no_direction_or_comparison_group(self, baseline_id: str) -> None:
        semantics = SEMANTIC_BASELINES[baseline_id]

        assert semantics.direction == MetricDirection.NONE
        assert semantics.comparison_group is None

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_only_diagnostic_baselines_drop_the_direction(self, baseline_id: str) -> None:
        semantics = SEMANTIC_BASELINES[baseline_id]

        if semantics.role != MetricRole.DIAGNOSTIC:
            assert semantics.direction != MetricDirection.NONE


class TestComparisonGroupNaming:

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_comparison_group_matches_the_domain_concept_prefix(self, baseline_id: str) -> None:
        semantics = SEMANTIC_BASELINES[baseline_id]
        if semantics.comparison_group is None:
            pytest.skip(f'{baseline_id} declares no comparison group')

        domain, concept = semantics.semantic_id.split('.')[:2]

        assert semantics.comparison_group == f'{domain}.{concept}'
        assert semantics.semantic_id.startswith(f'{semantics.comparison_group}.')


class TestDisplayDeclarations:

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_percent_baselines_declare_range_and_multiplier(self, baseline_id: str) -> None:
        semantics = SEMANTIC_BASELINES[baseline_id]
        if semantics.display_kind != MetricDisplayKind.PERCENT:
            pytest.skip(f'{baseline_id} is not rendered as percent')

        assert semantics.value_range is not None
        assert semantics.display_multiplier is not None
        assert semantics.display_unit == '%'

    def test_ratio_baselines_scale_by_100_and_point_scales_by_1(self) -> None:
        assert SEMANTIC_BASELINES['quality.accuracy.ratio'].display_multiplier == 100.0
        assert SEMANTIC_BASELINES['quality.score.points_100'].display_multiplier == 1.0

    def test_error_rate_baselines_are_lower_is_better(self) -> None:
        for baseline_id in ('quality.wer.ratio', 'quality.cer.ratio'):
            assert SEMANTIC_BASELINES[baseline_id].direction == MetricDirection.LOWER_IS_BETTER

    def test_perf_baselines_declare_raw_and_display_units(self) -> None:
        for baseline_id in (
            'perf.latency.seconds',
            'perf.latency.milliseconds',
            'perf.throughput.tokens_per_second',
            'perf.throughput.requests_per_second',
        ):
            semantics = SEMANTIC_BASELINES[baseline_id]
            assert semantics.raw_unit
            assert semantics.display_unit == semantics.raw_unit
            assert semantics.display_kind == MetricDisplayKind.NUMBER

    def test_perf_directions_follow_latency_and_throughput(self) -> None:
        assert SEMANTIC_BASELINES['perf.latency.seconds'].direction == MetricDirection.LOWER_IS_BETTER
        assert SEMANTIC_BASELINES['perf.latency.milliseconds'].direction == MetricDirection.LOWER_IS_BETTER
        assert (SEMANTIC_BASELINES['perf.throughput.tokens_per_second'].direction == MetricDirection.HIGHER_IS_BETTER)
        assert (SEMANTIC_BASELINES['perf.throughput.requests_per_second'].direction == MetricDirection.HIGHER_IS_BETTER)

    def test_unbounded_judge_score_stays_a_plain_number(self) -> None:
        semantics = SEMANTIC_BASELINES['quality.judge_score.unbounded']

        assert semantics.value_range is None
        assert semantics.display_kind == MetricDisplayKind.NUMBER
        assert semantics.display_unit is None
