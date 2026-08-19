"""Smoke tests for migrated Native judge adapters."""
from typing import Any, List

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import JudgeScoreType, ScoreStatus
from evalscope.metrics.judge.llm_judge import DEFAULT_PROMPT_TEMPLATE, LLMJudge


class ScriptedJudge:
    score_type = JudgeScoreType.PATTERN
    score_mapping = {'A': 1.0, 'B': 0.0}
    prompt_template = DEFAULT_PROMPT_TEMPLATE
    system_prompt = None
    build_prompt = LLMJudge.build_prompt

    def __init__(self, replies: List[str]) -> None:
        self.replies = replies
        self.judge_id = self.model_id = 'scripted'
        self.calls: List[Any] = []

    def generate(self, messages):
        self.calls.append(messages)
        return ModelOutput.from_content('scripted', self.replies[min(len(self.calls) - 1, len(self.replies) - 1)])


def make_state(prediction: str, target: str) -> TaskState:
    sample = Sample(id=0, input='Who wrote Hamlet?', target=target, metadata={})
    return TaskState(model='m', sample=sample, output=ModelOutput.from_content('m', prediction), completed=True)


def test_simple_qa_uses_the_contract_and_excludes_bad_json():
    config = TaskConfig(model='m', datasets=['simple_qa'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark('simple_qa', config)
    adapter.llm_judge = ScriptedJudge(['not JSON'])

    score = adapter.calculate_metrics(make_state('Shakespeare', 'Shakespeare')).score

    assert score.status is ScoreStatus.DEGRADED
    assert score.value['is_not_attempted'] == 1.0
    assert score.metadata['judge_attempts'][0]['status'] == 'parse_error'


def test_arena_hard_swap_is_driven_by_the_executor():
    config = TaskConfig(model='m', datasets=['arena_hard'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark('arena_hard', config)
    adapter.llm_judge = ScriptedJudge(['{"verdict": "A>B"}', '{"verdict": "B>A"}'])

    state = make_state('candidate', 'baseline')
    score = adapter.calculate_metrics(state).score

    assert score.status is ScoreStatus.SUCCESS
    assert len(score.metadata['judge_attempts']) == 2
    assert score.metadata['battle_result']['games'] == [{'score': 'A>B'}, {'score': 'B>A'}]
    assert score.judge_summary.status is ScoreStatus.SUCCESS


def test_position_swap_off_keeps_one_official_pairwise_game():
    config = TaskConfig(
        model='m',
        datasets=['arena_hard'],
        judge={
            'strategy': 'llm',
            'models': [{'model_id': 'j'}],
            'position_swap': 'off',
        },
    )
    adapter = get_benchmark('arena_hard', config)
    adapter.llm_judge = ScriptedJudge(['{"verdict": "A>B"}'])

    score = adapter.calculate_metrics(make_state('candidate', 'baseline')).score

    assert score.status is ScoreStatus.SUCCESS
    assert len(score.metadata['judge_attempts']) == 1
    assert score.metadata['battle_result']['games'] == [{'score': 'A>B'}]


def test_position_swap_on_is_ignored_for_non_pairwise_contracts():
    config = TaskConfig(
        model='m',
        datasets=['simple_qa'],
        judge={
            'strategy': 'llm',
            'models': [{'model_id': 'j'}],
            'position_swap': 'on',
        },
    )
    adapter = get_benchmark('simple_qa', config)
    adapter.llm_judge = ScriptedJudge(['{"verdict": "A"}'])

    score = adapter.calculate_metrics(make_state('Shakespeare', 'Shakespeare')).score

    assert score.status is ScoreStatus.SUCCESS
    assert len(score.metadata['judge_attempts']) == 1
