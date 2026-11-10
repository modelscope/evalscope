"""Preparation failures use the evaluator's normal error and cache policy."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evalscope.api.agent import AgentLoopResult, AgentTrace, prepare_environment
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter, DefaultDataAdapter
from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.config import TaskConfig
from evalscope.evaluator.evaluator import DefaultEvaluator
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from evalscope.utils.io_utils import OutputsStructure, jsonl_to_list


@pytest.mark.parametrize('ignore_errors', [False, True])
@pytest.mark.parametrize('batch_scoring', [False, True])
def test_preparation_error_policy_and_resume(
    ignore_errors: bool, batch_scoring: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = TaskConfig(model='mock', eval_type='mock_llm', use_cache=str(tmp_path),
                      eval_batch_size=1, collect_perf=False)
    assert task.ignore_errors is False
    if ignore_errors:
        task.ignore_errors = True
    adapter = AgentLoopAdapter(
        benchmark_meta=BenchmarkMeta(name='prepare-test', dataset_id='fake', metric_list=['accuracy']),
        task_config=task,
    )
    adapter.use_batch_scoring = batch_scoring
    adapter._resolve_strategy = lambda sample, ac: SimpleNamespace(
        name='fake', extract_final_answer=lambda result: result.final_output.completion)
    adapter._resolve_tools = lambda sample, ac: ({}, [])
    adapter.calculate_metrics = MagicMock(side_effect=lambda task_state: SampleScore(
        sample_id=task_state.sample_id, score=Score(value={'accuracy': 1.0})))
    adapter.batch_match_score = MagicMock(return_value=None)
    adapter.aggregate_subset = MagicMock(return_value=[])
    error = ConnectionError('sandbox unavailable')
    environments = {
        index: SimpleNamespace(name='fake', prepare=AsyncMock(side_effect=error if index == 0 else None),
                               close=AsyncMock())
        for index in range(2)
    }
    adapter.build_environment = lambda sample: environments[sample.id]
    model = SimpleNamespace(name='mock', generate=MagicMock(
        return_value=ModelOutput.from_content(model='mock', content='answer')))

    def run_loop(**kwargs: Any) -> AgentLoopResult:
        try:
            return AgentLoopResult(messages=[], final_output=model.generate(), trace=AgentTrace())
        finally:
            AsyncioLoopRunner.run(kwargs['environment'].close())

    monkeypatch.setattr('evalscope.api.benchmark.adapters.agent_adapter.run_agent_loop', run_loop)
    samples = [Sample(id=index, input=[ChatMessageUser(content='task')], target='answer') for index in range(2)]
    dataset = MemoryDataset(samples, name='test')
    evaluator = DefaultEvaluator(adapter, model, OutputsStructure(str(tmp_path), is_make=True), task)
    manager = evaluator.cache_manager
    try:
        context = evaluator._collect_work_items({'test': dataset})
        if not ignore_errors:
            with pytest.raises(ConnectionError) as caught:
                evaluator._run_pool(context)
            assert caught.value is error
            model.generate.assert_not_called()
            adapter.calculate_metrics.assert_not_called()
            assert not Path(manager.get_prediction_cache_path('test')).exists()
            assert not Path(manager.get_review_cache_path('test')).exists()
        else:
            results = evaluator._run_pool(context)
            evaluator._aggregate_scores({'test': dataset}, context, results)
            manager.close()
            assert [row['index'] for row in jsonl_to_list(manager.get_prediction_cache_path('test'))] == [1]
            assert [row['index'] for row in jsonl_to_list(manager.get_review_cache_path('test'))] == [1]
            assert [call.kwargs['task_state'].sample_id for call in adapter.calculate_metrics.call_args_list] == [1]
            summary = evaluator._execution_tracker.summarize({'test': dataset}, evaluator._sample_scores_by_subset)
            assert (summary.requested, summary.succeeded, summary.errored, summary.incomplete) == (2, 1, 1, True)

        environments[0].close.assert_awaited_once()
        environments[0].prepare.side_effect = None
        manager.close()
        evaluator = DefaultEvaluator(adapter, model, OutputsStructure(str(tmp_path), is_make=True), task)
        manager = evaluator.cache_manager
        resumed = evaluator._collect_work_items({'test': dataset})
        assert [item.sample.id for item in resumed.work_items] == ([0] if ignore_errors else [0, 1])
        assert resumed.total_cached == int(ignore_errors)
        results = evaluator._run_pool(resumed)
        evaluator._aggregate_scores({'test': dataset}, resumed, results)
        manager.close()
        assert sorted(row['index'] for row in jsonl_to_list(manager.get_prediction_cache_path('test'))) == [0, 1]
        assert sorted(row['index'] for row in jsonl_to_list(manager.get_review_cache_path('test'))) == [0, 1]
        assert model.generate.call_count == 2
        summary = evaluator._execution_tracker.summarize({'test': dataset}, evaluator._sample_scores_by_subset)
        assert (summary.succeeded, summary.errored, summary.incomplete) == (2, 0, False)
    finally:
        manager.close()


@pytest.mark.parametrize('ignore_errors', [False, True])
def test_grading_preparation_failure_reuses_prediction(tmp_path: Path, ignore_errors: bool) -> None:
    task = TaskConfig(model='mock', eval_type='mock_llm', use_cache=str(tmp_path),
                      ignore_errors=ignore_errors, eval_batch_size=1, collect_perf=False)
    adapter = DefaultDataAdapter(
        benchmark_meta=BenchmarkMeta(name='grading-test', dataset_id='fake', metric_list=['accuracy']),
        task_config=task,
    )
    env = SimpleNamespace(prepare=AsyncMock(side_effect=ConnectionError('grading sandbox unavailable')),
                          close=AsyncMock())

    def grade(**kwargs: Any) -> Score:
        prepare_environment(env, sample_id=kwargs['task_state'].sample_id)
        AsyncioLoopRunner.run(env.close())
        return Score(value={'accuracy': 1.0})

    adapter.match_score = grade
    model = SimpleNamespace(name='mock', generate=MagicMock(
        return_value=ModelOutput.from_content(model='mock', content='answer')))
    dataset = MemoryDataset([Sample(id=0, input=[ChatMessageUser(content='task')], target='answer')], name='test')
    outputs = OutputsStructure(str(tmp_path), is_make=True)
    evaluator = DefaultEvaluator(adapter, model, outputs, task)
    manager = evaluator.cache_manager
    try:
        context = evaluator._collect_work_items({'test': dataset})
        if ignore_errors:
            assert not evaluator._run_pool(context)
        else:
            with pytest.raises(ConnectionError, match='grading sandbox unavailable'):
                evaluator._run_pool(context)
        manager.close()
        assert len(jsonl_to_list(manager.get_prediction_cache_path('test'))) == 1
        assert not Path(manager.get_review_cache_path('test')).exists()
        env.prepare.side_effect = None
        evaluator = DefaultEvaluator(adapter, model, outputs, task)
        manager = evaluator.cache_manager
        resumed = evaluator._collect_work_items({'test': dataset})
        assert len(resumed.work_items) == 1 and not resumed.work_items[0].needs_predict
        evaluator._run_pool(resumed)
        manager.close()
        assert len(jsonl_to_list(manager.get_review_cache_path('test'))) == 1
        assert model.generate.call_count == 1
    finally:
        manager.close()
