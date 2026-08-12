import pytest
from typing import Any, Dict, List

import evalscope  # noqa: F401  # imported for benchmark registration side effects
from evalscope import TaskConfig
from evalscope.api.metric import MetricIdentity, MetricKind, SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.report import Report
from evalscope.report.generator import ReportGenerator


def _sample_score(value: Dict[str, Any], sample_id: int = 0, category: str = '') -> SampleScore:
    return SampleScore(
        score=Score(value=value),
        sample_id=sample_id,
        sample_metadata={'category': category} if category else {},
    )


def _generate_report(benchmark_name: str, sample_scores: List[SampleScore]) -> Report:
    adapter = get_benchmark(benchmark_name)
    scores = adapter.aggregate_scores(sample_scores)
    return ReportGenerator.generate_report({'default': scores}, 'model', adapter)


@pytest.mark.parametrize(
    ('benchmark_name', 'sample_scores', 'primary_name', 'expected_semantics'),
    [
        (
            'k2_verifier',
            [_sample_score({
                'finish_reason_tool_call': 1,
                'successful_tool_call': 1,
                'should_call_tool': 1,
            })],
            'trigger_similarity',
            'quality.similarity.ratio',
        ),
        (
            'minimax_verifier',
            [_sample_score({
                'inference_error': 0,
                'error_only_reasoning': 0,
                'tool_calls_run': 1,
                'tool_calls_finish_tool_calls': 1,
                'tool_calls_schema_valid': 1,
                'expected_tool_call_labeled': 1,
                'tool_calls_match': 1,
                'language_following_checked': 1,
                'language_following_valid': 1,
                'repeat_ngram_checked': 1,
                'repeat_ngram_valid': 1,
                'scenario_check_checked': 1,
                'scenario_check_valid': 1,
            })],
            'tool_calls_match_rate',
            'quality.accuracy.ratio',
        ),
        (
            'mcp_atlas',
            [_sample_score({
                'coverage_score': 1.0,
                'pass': 1.0,
            })],
            'pass_rate',
            'quality.pass_at_k.ratio',
        ),
    ],
)
def test_vendor_benchmarks_generate_reports_from_actual_aggregators(
    benchmark_name: str,
    sample_scores: List[SampleScore],
    primary_name: str,
    expected_semantics: str,
) -> None:
    report = _generate_report(benchmark_name, sample_scores)

    assert report.primary_metric_identity is not None
    assert report.primary_metric_identity.name == primary_name
    assert report.primary_metric is not None
    assert report.primary_metric.semantics.semantic_id == expected_semantics
    assert report.primary_metric.semantics.kind is MetricKind.QUALITY


def test_minimax_auxiliary_rates_have_declared_directions() -> None:
    report = _generate_report(
        'minimax_verifier',
        [_sample_score({
            'inference_error': 0,
            'error_only_reasoning': 1,
            'tool_calls_run': 1,
            'tool_calls_finish_tool_calls': 1,
            'tool_calls_schema_valid': 1,
            'expected_tool_call_labeled': 1,
            'tool_calls_match': 1,
            'language_following_checked': 1,
            'language_following_valid': 1,
            'repeat_ngram_checked': 1,
            'repeat_ngram_valid': 1,
            'scenario_check_checked': 1,
            'scenario_check_valid': 1,
        })],
    )
    semantics = {metric.identity.name: metric.semantics for metric in report.metrics}

    assert semantics['error_only_reasoning_rate'].semantic_id == 'quality.error_rate.ratio'
    for name in ('language_following_success_rate', 'repeat_ngram_pass_rate', 'scenario_check_pass_rate'):
        assert semantics[name].semantic_id == 'quality.accuracy.ratio'
        assert semantics[name].kind is MetricKind.QUALITY


@pytest.mark.parametrize(
    ('benchmark_name', 'scorer_name', 'metric_result', 'primary_name'),
    [
        ('genai_bench', 'VQAScore', 0.73, 'vqa_model_score'),
        ('hpdv2', 'HPSv2.1Score', 0.73, 'hps_v2_1_score'),
        ('tifa160', 'PickScore', 0.73, 'pick_score'),
        ('evalmuse', 'FGA_BLIP2Score', {
            'overall_score': 0.73, 'object-detail': 0.61
        }, 'fga_blip2_score'),
        ('general_t2i', 'PickScore', 0.73, 'pick_score'),
    ],
)
def test_t2i_benchmarks_emit_structured_primary_and_breakdowns(
    benchmark_name: str,
    scorer_name: str,
    metric_result: Any,
    primary_name: str,
) -> None:
    adapter = get_benchmark(benchmark_name)
    sample_scores = []
    for sample_id, category in enumerate(('basic', 'advanced')):
        score = Score()
        adapter._record_metric_result(score, scorer_name, metric_result)
        sample_scores.append(SampleScore(score=score, sample_id=sample_id, sample_metadata={'category': category}))

    aggregates = adapter.aggregate_scores(sample_scores)
    identities = {aggregate.identity for aggregate in aggregates}
    report = ReportGenerator.generate_report({'default': aggregates}, 'model', adapter)

    primary_identity = MetricIdentity(name=primary_name, aggregation='mean', dimensions={'scope': 'overall'})
    assert primary_identity in identities
    assert MetricIdentity(name=primary_name, aggregation='mean', dimensions={'category': 'basic'}) in identities
    assert MetricIdentity(name=primary_name, aggregation='mean', dimensions={'category': 'advanced'}) in identities
    assert report.primary_metric_identity == primary_identity
    assert report.primary_metric is not None
    assert report.primary_metric.semantics.semantic_id == 'quality.model_score.unbounded'

    if isinstance(metric_result, dict):
        assert MetricIdentity(
            name=primary_name,
            aggregation='mean',
            dimensions={'component': 'object_detail'},
        ) in identities


def test_general_t2i_does_not_force_an_unknown_metric_to_primary() -> None:
    config = TaskConfig(
        datasets=['general_t2i'],
        dataset_args={'general_t2i': {
            'metric_list': ['VendorScore']
        }},
    )
    adapter = get_benchmark('general_t2i', config)
    score = Score()
    adapter._record_metric_result(score, 'VendorScore', 3.5)

    aggregates = adapter.aggregate_scores([SampleScore(score=score, sample_id=0)])
    report = ReportGenerator.generate_report({'default': aggregates}, 'model', adapter)

    assert adapter.primary_metric is None
    assert report.primary_metric_identity is None
    assert report.metrics[0].semantics.kind is MetricKind.DIAGNOSTIC
