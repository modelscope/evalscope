"""Pass 2 of batch review returns only refined scores: when every preliminary review failed it
returns an empty list (the subset is then skipped), not a list of ``None`` for aggregation."""
from types import SimpleNamespace

from evalscope.evaluator.batch_reviewer import BatchReviewer


def _reviewer(batch_calculate_metrics=None):
    benchmark = SimpleNamespace(batch_calculate_metrics=batch_calculate_metrics)
    return BatchReviewer(benchmark, cache_manager=None, task_config=SimpleNamespace(eval_batch_size=2))


def test_all_reviews_failed_returns_no_scores():
    reviewer = _reviewer()
    assert reviewer._review_task_states(['s0', 's1'], [None, None], on_result=lambda *_: None, on_error=None) == []


def test_failed_reviews_are_left_out():
    reviewer = _reviewer(lambda task_states, sample_scores: list(sample_scores))
    scores = reviewer._review_task_states(['s0', 's1'], [None, 'score1'], on_result=lambda *_: None, on_error=None)
    assert scores == ['score1']
