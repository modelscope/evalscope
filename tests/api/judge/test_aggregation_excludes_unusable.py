"""An unusable judge review carries an empty ``Score.value``; every adapter that keeps its own
``aggregate_scores`` must exclude such a sample from the metric rather than count it as 0.

These guard the four aggregators fixed alongside the JSON-judge migration: the generic
aggregators are covered separately in ``tests/metrics/aggregators/test_missing_metric_keys.py``.
"""
from typing import Dict, Optional

import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import DataCollection, ScoreStatus

JUDGE_ARGS = {'model_id': 'judge', 'api_url': 'http://judge/v1', 'api_key': 'k'}


def scored(value: Dict[str, float], sample_id: int, metadata: Optional[dict] = None) -> SampleScore:
    return SampleScore(
        score=Score(value=value, status=ScoreStatus.SUCCESS),
        sample_id=sample_id,
        sample_metadata=metadata or {},
    )


def excluded(sample_id: int, metadata: Optional[dict] = None) -> SampleScore:
    """What ``JudgeExecutor.build_score`` produces for an unusable review: empty value, bad status."""
    return SampleScore(
        score=Score(value={}, status=ScoreStatus.PARSE_ERROR),
        sample_id=sample_id,
        sample_metadata=metadata or {},
    )


def _judge_benchmark(name: str):
    cfg = TaskConfig(model='m', datasets=[name], judge={'strategy': 'llm', 'models': JUDGE_ARGS})
    return get_benchmark(name, config=cfg)


def _by_metric(agg_scores) -> Dict[str, object]:
    return {agg.metric_name: agg for agg in agg_scores}


def test_researchrubrics_excludes_unusable_sample_instead_of_crashing() -> None:
    adapter = _judge_benchmark('researchrubrics')
    meta = {'domain': 'science', 'conceptual_breadth': None, 'logical_nesting': None, 'exploration': None}
    result = adapter.aggregate_scores([
        scored({'compliance_score': 1.0}, sample_id=1, metadata=meta),
        excluded(sample_id=2, metadata=meta),
    ])
    by_metric = _by_metric(result)
    # Previously raised KeyError('compliance_score'); now the excluded sample simply drops out.
    assert by_metric['compliance_score'].score == 1.0
    assert by_metric['compliance_score'].num == 1


def test_researchrubrics_all_unusable_yields_no_rows() -> None:
    adapter = _judge_benchmark('researchrubrics')
    assert adapter.aggregate_scores([excluded(sample_id=1), excluded(sample_id=2)]) == []


def test_mcp_atlas_excludes_unusable_sample() -> None:
    adapter = _judge_benchmark('mcp_atlas')
    result = _by_metric(
        adapter.aggregate_scores([
            scored({'coverage_score': 1.0, 'pass': 1.0}, sample_id=1),
            excluded(sample_id=2),
        ])
    )
    # Averaged over the one usable sample, not diluted to 0.5 by the excluded one.
    assert result['coverage_score'].score == 1.0
    assert result['coverage_score'].num == 1
    assert result['pass_rate'].score == 1.0
    assert result['pass_rate'].num == 1


def test_drivelology_averages_each_metric_over_its_own_usable_samples() -> None:
    adapter = _judge_benchmark('drivel_writing')
    result = _by_metric(
        adapter.aggregate_scores([
            # The rule-based bert_score is always present; the judge_score can be missing.
            scored({'judge_score': 1.0, 'bert_score': 0.8}, sample_id=1),
            scored({'bert_score': 0.6}, sample_id=2),
        ])
    )
    assert result['judge_score'].score == 1.0
    assert result['judge_score'].num == 1
    assert result['bert_score'].score == pytest.approx(0.7)
    assert result['bert_score'].num == 2


COLLECTION_INFO = {
    'task_type': 'qa',
    'categories': ('c', ),
    'dataset_name': 'd',
    'subset_name': 's',
    'tags': ['t'],
    'weight': 1.0,
}


def test_data_collection_drops_unusable_sample_from_dataframe() -> None:
    adapter = get_benchmark('data_collection', config=TaskConfig(model='m'))
    scores = [
        scored({'acc': 1.0}, sample_id=1, metadata={DataCollection.INFO: COLLECTION_INFO}),
        excluded(sample_id=2, metadata={DataCollection.INFO: COLLECTION_INFO}),
    ]
    df = adapter._build_sample_dataframe(scores)
    assert len(df) == 1
    rows = adapter._group_and_compute(df, ['dataset_name'])
    # The excluded sample no longer drags the weighted mean toward 0.
    assert rows[0]['weighted_avg.'] == 1.0
    assert rows[0]['count'] == 1


def test_data_collection_all_unusable_yields_empty_levels() -> None:
    """Excluding every sample leaves a column-less dataframe: grouping it raised KeyError, which
    aborted the run after all inference had been paid for."""
    adapter = get_benchmark('data_collection', config=TaskConfig(model='m'))
    scores = [
        excluded(sample_id=1, metadata={DataCollection.INFO: COLLECTION_INFO}),
        excluded(sample_id=2, metadata={DataCollection.INFO: COLLECTION_INFO}),
    ]
    report = adapter.aggregate_scores(scores)
    assert report['subset_level'] == []
    assert report['dataset_level'] == []
    assert report['task_level'] == []
    assert report['tag_level'] == []
    assert report['category_level'] == []
    assert report['df'].empty
