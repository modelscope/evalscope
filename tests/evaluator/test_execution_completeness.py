from collections import defaultdict

from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.evaluator.evaluator import DefaultEvaluator
from evalscope.evaluator.perf_collector import PerfCollector


def _sample_score(sample_id: int) -> SampleScore:
    return SampleScore(sample_id=sample_id, score=Score(value={'acc': 1.0}, main_score_name='acc'))


def test_execution_summary_tracks_partial_failure_and_cached_scores() -> None:
    evaluator = object.__new__(DefaultEvaluator)
    evaluator._sample_scores_by_subset = {'test': [_sample_score(0), _sample_score(1)]}
    evaluator._execution_errors_by_subset = defaultdict(int, {'test': 1})

    summary = evaluator._build_execution_summary({'test': [object(), object(), object()]})

    assert summary.model_dump() == {
        'requested': 3,
        'succeeded': 2,
        'errored': 1,
        'incomplete': True,
        'subsets': {
            'test': {
                'requested': 3,
                'succeeded': 2,
                'errored': 1,
            }
        },
    }


def test_execution_summary_marks_all_success_complete() -> None:
    evaluator = object.__new__(DefaultEvaluator)
    evaluator._sample_scores_by_subset = {'test': [_sample_score(0)]}
    evaluator._execution_errors_by_subset = defaultdict(int)

    summary = evaluator._build_execution_summary({'test': [object()]})

    assert not summary.incomplete
    assert summary.errored == 0


def test_perf_coverage_counts_missing_request_metrics() -> None:
    evaluator = object.__new__(DefaultEvaluator)
    evaluator.perf_collector = PerfCollector()
    evaluator._perf_request_count = 0
    evaluator._perf_metric_count = 0
    task_state = type(
        'TaskStateProbe',
        (), {
            'messages': [
                ChatMessageAssistant(content='first', perf_metrics=PerformanceMetrics(latency=1.0)),
                ChatMessageAssistant(content='second'),
            ],
            'output': ModelOutput.from_content(model='test', content='second'),
            'sample_id': 1,
        },
    )()

    evaluator._record_perf(task_state)

    assert evaluator._perf_request_count == 2
    assert evaluator._perf_metric_count == 1
    assert evaluator.perf_collector.get_perf_dict()['summary']['n_samples'] == 1
