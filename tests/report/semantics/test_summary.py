"""Unit tests for the primary metric summary helper.

Feature: metric-semantics-governance
"""
import pytest
from pydantic import ValidationError
from typing import Optional

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricRole, MetricSemantics, ValueRange
from evalscope.metrics.semantics.summary import (
    MetricSummary,
    PrimaryMetricRef,
    SummaryStatus,
    summarize_primary_metrics,
)

ACCURACY = MetricSemantics(
    semantic_id='quality.accuracy.ratio',
    metric_name='Accuracy',
    role=MetricRole.PRIMARY,
    direction=MetricDirection.HIGHER_IS_BETTER,
    value_range=ValueRange(min=0.0, max=1.0),
    display_kind=MetricDisplayKind.PERCENT,
    display_multiplier=100.0,
    display_unit='%',
    display_precision=1,
    comparison_group='quality.accuracy',
)

WER = MetricSemantics(
    semantic_id='quality.wer.ratio',
    metric_name='WER',
    role=MetricRole.PRIMARY,
    direction=MetricDirection.LOWER_IS_BETTER,
    value_range=ValueRange(min=0.0, max=1.0),
    display_kind=MetricDisplayKind.PERCENT,
    display_multiplier=100.0,
    display_unit='%',
    display_precision=1,
    comparison_group='quality.wer',
)


def ref(
    dataset_name: str,
    score: Optional[float] = 0.5,
    semantics: Optional[MetricSemantics] = ACCURACY,
    metric_name: str = 'mean_acc',
) -> PrimaryMetricRef:
    return PrimaryMetricRef(
        dataset_name=dataset_name,
        metric_name=metric_name,
        score=score,
        semantics=semantics,
    )


class TestSingleMetric:

    def test_single_declared_ref_becomes_the_summary_value(self) -> None:
        summary = summarize_primary_metrics([ref('gsm8k', score=0.4123)])

        assert summary.status == SummaryStatus.SINGLE_METRIC
        assert summary.summary_score == 0.4123
        assert summary.summary_semantics == ACCURACY
        assert [item.dataset_name for item in summary.primary_metrics] == ['gsm8k']

    def test_single_declared_ref_without_score_keeps_the_semantics(self) -> None:
        summary = summarize_primary_metrics([ref('gsm8k', score=None)])

        assert summary.status == SummaryStatus.SINGLE_METRIC
        assert summary.summary_score is None
        assert summary.summary_semantics == ACCURACY

    def test_single_ref_without_semantics_is_mixed(self) -> None:
        summary = summarize_primary_metrics([ref('third_party', semantics=None, metric_name='')])

        assert summary.status == SummaryStatus.MIXED_METRICS
        assert summary.summary_score is None
        assert summary.summary_semantics is None


class TestMultipleMetrics:

    def test_shared_semantic_id_is_not_aggregated(self) -> None:
        summary = summarize_primary_metrics([ref('gsm8k', score=0.4), ref('math_500', score=0.6)])

        assert summary.status == SummaryStatus.NO_AGGREGATE
        assert summary.summary_score is None
        assert summary.summary_semantics is None
        assert len(summary.primary_metrics) == 2

    def test_different_semantic_ids_are_mixed(self) -> None:
        summary = summarize_primary_metrics([
            ref('gsm8k', score=0.4),
            ref('speech_asr', score=0.07, semantics=WER, metric_name='wer'),
        ])

        assert summary.status == SummaryStatus.MIXED_METRICS
        assert summary.summary_score is None
        assert summary.summary_semantics is None

    def test_one_missing_semantics_makes_the_collection_mixed(self) -> None:
        summary = summarize_primary_metrics([ref('gsm8k', score=0.4), ref('custom', semantics=None, metric_name='')])

        assert summary.status == SummaryStatus.MIXED_METRICS
        assert summary.summary_score is None
        assert summary.summary_semantics is None

    def test_no_equal_weight_mean_is_computed(self) -> None:
        scores = [0.2, 0.4, 0.9]
        summary = summarize_primary_metrics([
            ref(f'dataset_{index}', score=score) for index, score in enumerate(scores)
        ])

        assert summary.summary_score is None
        assert [item.score for item in summary.primary_metrics] == scores


class TestEdgeCases:

    def test_empty_collection_has_no_summary_value(self) -> None:
        summary = summarize_primary_metrics([])

        assert summary.status == SummaryStatus.MIXED_METRICS
        assert summary.summary_score is None
        assert summary.summary_semantics is None
        assert summary.primary_metrics == []

    def test_input_order_is_preserved(self) -> None:
        names = ['c', 'a', 'b']
        summary = summarize_primary_metrics([ref(name) for name in names])

        assert [item.dataset_name for item in summary.primary_metrics] == names

    def test_models_are_frozen_and_forbid_extra_fields(self) -> None:
        summary = summarize_primary_metrics([ref('gsm8k')])

        with pytest.raises(ValidationError):
            summary.summary_score = 1.0

        with pytest.raises(ValidationError):
            PrimaryMetricRef(dataset_name='gsm8k', metric_name='mean_acc', unknown='x')

        with pytest.raises(ValidationError):
            MetricSummary(status=SummaryStatus.SINGLE_METRIC, unknown='x')

    def test_status_values_match_the_api_contract(self) -> None:
        assert [status.value for status in SummaryStatus] == ['single_metric', 'no_aggregate', 'mixed_metrics']
