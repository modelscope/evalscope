"""macro_avg. above the dataset level averages the collection's members, which are
(dataset, subset) pairs: subsets of different datasets that share a name stay separate."""
import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import DataCollection

# (dataset, subset, samples, correct)
PLAN = [('gpqa_diamond', 'default', 10, 2), ('humaneval', 'default', 30, 24), ('arc', 'ARC-Easy', 10, 10)]


def _scores():
    scores = []
    for dataset, subset, n, correct in PLAN:
        for i in range(n):
            info = {
                'task_type': 'reasoning',
                'categories': ['reasoning_index'],
                'dataset_name': dataset,
                'subset_name': subset,
                'tags': ['en'],
                'weight': 1.0,
            }
            scores.append(
                SampleScore(
                    score=Score(value={'acc': float(i < correct)}, main_score_name='acc'),
                    sample_id=len(scores),
                    sample_metadata={DataCollection.INFO: info},
                )
            )
    return scores


@pytest.mark.parametrize('level', ['task_level', 'tag_level', 'category_level'])
def test_macro_avg_keeps_same_named_subsets_of_different_datasets_apart(level):
    adapter = get_benchmark('data_collection', config=TaskConfig(model='m'))
    report = adapter.aggregate_scores(_scores())

    assert len(report['subset_level']) == 3
    # Members score 0.2, 0.8 and 1.0.
    assert report[level][0]['macro_avg.'] == pytest.approx(0.6667)
