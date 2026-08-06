"""Unit tests for final report metric name composition.

Feature: metric-semantics-governance
"""
from typing import Dict, List

from evalscope.api.metric import AggScore
from evalscope.metrics.semantics.naming import compose_final_metric_name


class TestComposeFinalMetricName:

    def test_prefixes_aggregation_name_when_enabled(self) -> None:
        agg_score = AggScore(score=0.5, metric_name='Accuracy', aggregation_name='mean', num=2)

        assert compose_final_metric_name(agg_score, add_aggregation_name=True) == 'mean_Accuracy'

    def test_defaults_to_adding_aggregation_name(self) -> None:
        agg_score = AggScore(score=0.5, metric_name='Accuracy', aggregation_name='mean', num=2)

        assert compose_final_metric_name(agg_score) == 'mean_Accuracy'

    def test_empty_aggregation_name_falls_back_to_metric_name(self) -> None:
        agg_score = AggScore(score=0.5, metric_name='Accuracy', aggregation_name='', num=2)

        assert compose_final_metric_name(agg_score, add_aggregation_name=True) == 'Accuracy'

    def test_default_aggregation_name_falls_back_to_metric_name(self) -> None:
        # AggScore.aggregation_name defaults to '' and is never None.
        agg_score = AggScore(score=0.5, metric_name='Accuracy', num=2)

        assert compose_final_metric_name(agg_score, add_aggregation_name=True) == 'Accuracy'

    def test_disabled_flag_keeps_plain_metric_name(self) -> None:
        agg_score = AggScore(score=0.5, metric_name='Accuracy', aggregation_name='mean', num=2)

        assert compose_final_metric_name(agg_score, add_aggregation_name=False) == 'Accuracy'


class TestGeneratorParity:
    """The helper must reproduce the spelling rule of ReportGenerator.generate_report()."""

    @staticmethod
    def _generated_metric_names(score_dict: Dict[str, List[AggScore]], add_aggregation_name: bool) -> List[str]:
        from evalscope.report.generator import ReportGenerator

        class _StubAdapter:
            name = 'stub_benchmark'
            pretty_name = 'Stub Benchmark'
            description = 'stub'
            category_map: Dict[str, List[str]] = {}

        report = ReportGenerator.generate_report(
            score_dict=score_dict,
            model_name='stub_model',
            data_adapter=_StubAdapter(),
            add_aggregation_name=add_aggregation_name,
        )
        return [metric.name for metric in report.metrics]

    def test_parity_with_report_generator(self) -> None:
        agg_scores = [
            AggScore(score=0.5, metric_name='Accuracy', aggregation_name='mean', num=2),
            AggScore(score=0.25, metric_name='Pass@1', aggregation_name='', num=2),
        ]
        score_dict = {'subset_a': agg_scores}

        for add_aggregation_name in (True, False):
            expected = [
                compose_final_metric_name(agg_score, add_aggregation_name=add_aggregation_name)
                for agg_score in agg_scores
            ]

            assert self._generated_metric_names(score_dict, add_aggregation_name) == expected
