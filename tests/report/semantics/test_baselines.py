"""Unit tests for the common metric semantics baseline table.

Every entry is a ``MetricSemantics`` literal built at import time, so the contract rules are already
enforced before any test runs: a diagnostic baseline carrying a direction, or a percent
baseline missing its range, would make this module unimportable. Tests re-asserting those rules
cannot fail independently and were removed. What remains pins the things no validator checks --
which baselines must exist, the key/``semantic_id`` correspondence, the naming convention, and the
concrete direction / scale / unit choices.
"""
from typing import List

import pytest

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricSemantics
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES

#: Baselines the catalog and the legacy mapping are allowed to reference.
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

#: Baselines rendered as a percentage, selected up front so the display test does not skip.
PERCENT_BASELINE_IDS = [
    baseline_id for baseline_id in BASELINE_IDS
    if SEMANTIC_BASELINES[baseline_id].display_kind == MetricDisplayKind.PERCENT
]


class TestBaselineTableShape:

    @pytest.mark.parametrize('baseline_id', REQUIRED_BASELINE_IDS)
    def test_required_baseline_is_declared(self, baseline_id: str) -> None:
        assert baseline_id in SEMANTIC_BASELINES

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_key_equals_semantic_id(self, baseline_id: str) -> None:
        assert SEMANTIC_BASELINES[baseline_id].semantic_id == baseline_id

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_baseline_survives_a_serialization_round_trip(self, baseline_id: str) -> None:
        semantics = SEMANTIC_BASELINES[baseline_id]

        # A serialized baseline must be rebuildable from report.json without loss. The contract rules
        # themselves are already enforced when this module imports the table.
        assert MetricSemantics.model_validate(semantics.model_dump()) == semantics

    @pytest.mark.parametrize('baseline_id', BASELINE_IDS)
    def test_semantic_id_uses_the_declared_domains(self, baseline_id: str) -> None:
        assert baseline_id.split('.')[0] in {'quality', 'perf', 'diagnostic'}


class TestDisplayDeclarations:

    @pytest.mark.parametrize('baseline_id', PERCENT_BASELINE_IDS)
    def test_percent_baselines_render_with_a_percent_sign(self, baseline_id: str) -> None:
        semantics = SEMANTIC_BASELINES[baseline_id]

        # `value_range` / `display_multiplier` are already required by the model; only the unit is a
        # convention this table has to keep on its own.
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
