# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for ARC-AGI-2 test inputs and task-level scoring."""
import json

import pytest

from evalscope.api.dataset.utils import as_sample_list
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

TWO_TEST_TASK = {
    'fewshots': [{'input': [[1]], 'output': [[2]]}],
    'question': [{'input': [[3]], 'output': [[4]]}, {'input': [[5]], 'output': [[6]]}],
}
ONE_TEST_TASK = {
    'fewshots': [{'input': [[7]], 'output': [[8]]}],
    'question': [{'input': [[9]], 'output': [[0]]}],
}


def _adapter(**kwargs):
    return get_benchmark('arc_agi_2', TaskConfig(datasets=['arc_agi_2'], **kwargs))


def _sample_scores(adapter, answers):
    samples = as_sample_list(adapter.record_to_sample(TWO_TEST_TASK)) + as_sample_list(
        adapter.record_to_sample(ONE_TEST_TASK)
    )
    scores = []
    for sample_id, (sample, answer) in enumerate(zip(samples, answers)):
        state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content('mock', answer), completed=True)
        score = adapter.calculate_metrics(state)
        score.sample_id = sample_id
        scores.append(score)
    return scores


def test_every_test_input_is_a_sample() -> None:
    """A task with two test inputs is asked twice, once per input, each with its own gold grid."""
    samples = as_sample_list(_adapter().record_to_sample(TWO_TEST_TASK))
    assert len(samples) == 2
    assert 'Test Input: [[3]]' in samples[0].input
    assert 'Test Input: [[5]]' in samples[1].input
    assert [json.loads(sample.target) for sample in samples] == [[[4]], [[6]]]


def test_task_scores_fraction_of_test_inputs_solved() -> None:
    """ARC Prize scoring: (1/2 + 1/1) / 2 tasks = 0.75, not 2/3 over pairs or 1.0 over first inputs."""
    adapter = _adapter()
    scores = _sample_scores(adapter, ['[[4]]', '[[0]]', '[[0]]'])
    aggregates = {(agg.aggregation, agg.dimensions.get('k')): agg for agg in adapter.aggregate_scores(scores)}
    assert aggregates[('mean', None)].score == pytest.approx(0.75)
    assert aggregates[('mean', None)].num == 2
    assert aggregates[('pass_hat_k', 1)].score == pytest.approx(0.75)


def test_evaluation_version_bumped_for_scoring_change() -> None:
    assert _adapter()._benchmark_meta.evaluation_version == 'v1.1'
