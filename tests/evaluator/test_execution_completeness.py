from types import SimpleNamespace

from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.evaluator.evaluator import DefaultEvaluator, _PoolContext, _WorkItem
from evalscope.evaluator.execution_tracker import ExecutionTracker
from evalscope.evaluator.perf_collector import PerfCollector


def _sample_score(sample_id: int) -> SampleScore:
    return SampleScore(sample_id=sample_id, score=Score(value={'acc': 1.0}, main_score_name='acc'))


def test_execution_summary_tracks_partial_failure_and_cached_scores() -> None:
    tracker = ExecutionTracker()
    tracker.record_error('test')

    summary = tracker.summarize({'test': [object(), object(), object()]}, {'test': [_sample_score(0), _sample_score(1)]})

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
    tracker = ExecutionTracker()

    summary = tracker.summarize({'test': [object()]}, {'test': [_sample_score(0)]})

    assert not summary.incomplete
    assert summary.errored == 0


def test_pool_records_all_ignored_failures() -> None:
    evaluator = object.__new__(DefaultEvaluator)
    evaluator.benchmark_name = 'test'
    evaluator.task_config = SimpleNamespace(ignore_errors=True, eval_batch_size=1)
    evaluator._execution_tracker = ExecutionTracker()

    def fail_work_item(*args, **kwargs):
        raise RuntimeError('inference failed')

    evaluator._process_work_item = fail_work_item
    context = _PoolContext(
        work_items=[_WorkItem(subset='test', sample=object())],
        cached_scores_by_subset={},
        review_pending_by_subset={},
        model_prediction_dir='',
        total_cached=0,
    )

    assert evaluator._run_pool(context) == {}
    summary = evaluator._execution_tracker.summarize({'test': [object()]}, {})
    assert summary.model_dump(exclude={'subsets'}) == {
        'requested': 1,
        'succeeded': 0,
        'errored': 1,
        'incomplete': True,
    }


def test_batch_review_error_tracks_cached_and_successful_samples() -> None:
    evaluator = object.__new__(DefaultEvaluator)
    evaluator.task_config = SimpleNamespace(ignore_errors=True)
    evaluator._execution_tracker = ExecutionTracker()
    evaluator._sample_scores_by_subset = {}
    evaluator.benchmark = SimpleNamespace(
        use_batch_scoring=True,
        aggregate_scores=lambda sample_scores: [],
    )
    task_state = SimpleNamespace(sample_id=2)

    def review_subset(subset, task_states, review_fn, on_error):
        on_error(task_states[0], RuntimeError('batch review failed'))
        return [_sample_score(1)]

    evaluator.batch_reviewer = SimpleNamespace(review_subset=review_subset)
    context = _PoolContext(
        work_items=[],
        cached_scores_by_subset={'test': [_sample_score(0)]},
        review_pending_by_subset={'test': [task_state]},
        model_prediction_dir='',
        total_cached=1,
    )

    evaluator._aggregate_scores({'test': [object(), object(), object()]}, context, {'test': []})
    summary = evaluator._execution_tracker.summarize(
        {'test': [object(), object(), object()]}, evaluator._sample_scores_by_subset
    )
    assert summary.model_dump(exclude={'subsets'}) == {
        'requested': 3,
        'succeeded': 2,
        'errored': 1,
        'incomplete': True,
    }


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
