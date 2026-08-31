"""Smoke tests for migrated Native judge adapters."""

import json
from typing import Any, List

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.prbench.prbench_adapter import PRBenchAdapter
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


class TransportFailingJudge(ScriptedJudge):
    def __init__(self) -> None:
        super().__init__([])

    def generate(self, messages):
        self.calls.append(messages)
        raise RuntimeError('judge transport unavailable')


def make_state(prediction: str, target: str) -> TaskState:
    sample = Sample(id=0, input='Who wrote Hamlet?', target=target, metadata={})
    return TaskState(model='m', sample=sample, output=ModelOutput.from_content('m', prediction), completed=True)


def make_one_million_state(adapter) -> TaskState:
    sample = adapter.record_to_sample(
        {
            'id': 'sample-id',
            'case_id': 1,
            'language': 'global',
            'system_prompt': '',
            'question': 'Write a professional answer.',
            'tags': {
                'topics': ['Law'],
                'time_sensitivity': {'time_sensitivity': 'Time-agnostic', 'year_month': 'NA', 'day': 'NA'},
            },
            'rubrics': [
                {
                    'rubric_number': 1,
                    'rubric_detail': 'Includes the requested analysis.',
                    'rubric_weight': 10,
                    'rubric_tag': 'Analytical Reasoning',
                }
            ],
        }
    )
    sample.id = 0
    sample.group_id = 0
    return TaskState(
        model='m', sample=sample, output=ModelOutput.from_content('m', 'Professional answer.'), completed=True
    )


def test_one_million_bench_valid_verdict_scores_the_sample() -> None:
    config = TaskConfig(
        model='m', datasets=['one_million_bench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]}
    )
    adapter = get_benchmark('one_million_bench', config)
    adapter.llm_judge = ScriptedJudge(
        [
            json.dumps(
                {'results': [{'rubric_id': 1, 'status': '是', 'justification': 'The requested analysis is present.'}]},
                ensure_ascii=False,
            )
        ]
    )

    score = adapter.calculate_metrics(make_one_million_state(adapter)).score

    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'expert_score': 1.0, 'pass_rate': 1.0}


@pytest.mark.parametrize('judge', [ScriptedJudge(['not JSON']), TransportFailingJudge()])
def test_one_million_bench_judge_failure_excludes_the_sample(judge) -> None:
    config = TaskConfig(
        model='m', datasets=['one_million_bench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]}
    )
    adapter = get_benchmark('one_million_bench', config)
    adapter.llm_judge = judge

    score = adapter.calculate_metrics(make_one_million_state(adapter)).score

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}


def make_prbench_adapter() -> PRBenchAdapter:
    config = TaskConfig(
        model='m',
        datasets=['prbench'],
        dataset_args={'prbench': {'subset_list': ['finance']}},
        judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
    )
    return get_benchmark('prbench', config)


def make_prbench_state(adapter: PRBenchAdapter) -> TaskState:
    sample = adapter.record_to_sample({
        'task': 'task-1',
        'turns': 1,
        'field': 'Finance',
        'topic': 'Accounting',
        'expert': 'Expert',
        'rubric': [
            {
                'id': 'positive',
                'title': 'Includes the required answer.',
                'annotations': {
                    'weight_class': 'critically important',
                    'critically_important_weight': 8,
                    'criteria_category': 'Financial Accuracy',
                },
            },
            {
                'id': 'negative',
                'title': 'Contains a material error.',
                'annotations': {
                    'weight_class': 'detrimental',
                    'detrimental_weight': -4,
                    'criteria_category': 'Financial Accuracy',
                },
            },
        ],
        'prompt_0': 'Analyze the transaction.',
        'reference_texts_0': [],
        'economic_pathway': 'Value Creation',
        'decision_type': 'Modeling & Measurement',
    })
    sample.id = 0
    return TaskState(model='m', sample=sample, output=ModelOutput.from_content('m', 'Answer'), completed=True)


def test_prbench_valid_verdicts_use_official_weighting() -> None:
    adapter = make_prbench_adapter()
    adapter.llm_judge = ScriptedJudge([
        '{"explanation": "present", "criteria_met": true}',
        '{"explanation": "present", "criteria_met": true}',
    ])

    score = adapter.calculate_metrics(make_prbench_state(adapter)).score

    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'clipped_score': 0.5, 'normalized_score': pytest.approx(2 / 3)}


@pytest.mark.parametrize('reply', ['not JSON', '[ERROR] judge transport unavailable'])
def test_prbench_invalid_judge_reply_excludes_the_sample(reply: str) -> None:
    adapter = make_prbench_adapter()
    adapter.llm_judge = ScriptedJudge([reply])

    score = adapter.calculate_metrics(make_prbench_state(adapter)).score

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.metadata['judge_attempts'][0]['status'] == 'parse_error'


def test_prbench_transport_failure_excludes_the_sample() -> None:
    adapter = make_prbench_adapter()
    adapter.llm_judge = TransportFailingJudge()

    score = adapter.calculate_metrics(make_prbench_state(adapter)).score

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.metadata['judge_attempts'][0]['status'] == 'transport_error'


@pytest.mark.parametrize('benchmark_name', ['simple_qa', 'chinese_simpleqa', 'simple_vqa'])
def test_three_way_judge_parse_failure_excludes_the_sample(benchmark_name: str) -> None:
    config = TaskConfig(model='m', datasets=[benchmark_name], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark(benchmark_name, config)
    adapter.llm_judge = ScriptedJudge(['not JSON'])

    score = adapter.calculate_metrics(make_state('Shakespeare', 'Shakespeare')).score

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.metadata['judge_attempts'][0]['status'] == 'parse_error'


@pytest.mark.parametrize('benchmark_name', ['simple_qa', 'chinese_simpleqa', 'simple_vqa'])
def test_three_way_judge_transport_failure_excludes_the_sample(benchmark_name: str) -> None:
    config = TaskConfig(model='m', datasets=[benchmark_name], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark(benchmark_name, config)
    adapter.llm_judge = TransportFailingJudge()

    score = adapter.calculate_metrics(make_state('Shakespeare', 'Shakespeare')).score

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.metadata['judge_attempts'][0]['status'] == 'transport_error'


@pytest.mark.parametrize(
    'benchmark_name',
    ['baby_vision', 'imo_answerbench', 'math_verse', 'minerva_math', 'world_vqa', 'zerobench'],
)
def test_generic_pattern_contract_supports_simple_judge_benchmarks(benchmark_name: str) -> None:
    config = TaskConfig(model='m', datasets=[benchmark_name], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark(benchmark_name, config)
    adapter.llm_judge = ScriptedJudge(['{"verdict": "A"}'])

    score = adapter.calculate_metrics(make_state('Shakespeare', 'Shakespeare')).score

    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'acc': 1.0}


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


def test_arena_hard_preserves_each_placement_game_after_candidate_summary():
    config = TaskConfig(model='m', datasets=['arena_hard'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark('arena_hard', config)
    # Original: candidate B wins. Swapped: candidate A loses. The candidate summary is a tie,
    # but the official battle stream must retain the two non-tie games.
    adapter.llm_judge = ScriptedJudge(['{"verdict": "B>A"}', '{"verdict": "B>A"}'])

    score = adapter.calculate_metrics(make_state('candidate', 'baseline')).score

    assert score.value['score'] == 0.5
    assert score.metadata['battle_result']['games'] == [{'score': 'B>A'}, {'score': 'B>A'}]


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


def test_position_swap_on_overrides_alpaca_eval_official_single_pass():
    config = TaskConfig(
        model='m',
        datasets=['alpaca_eval'],
        judge={
            'strategy': 'llm',
            'models': [{'model_id': 'j'}],
            'position_swap': 'on',
        },
    )
    adapter = get_benchmark('alpaca_eval', config)
    adapter.llm_judge = ScriptedJudge(['{"verdict": "M"}', '{"verdict": "m"}'])

    score = adapter.calculate_metrics(make_state('candidate', 'baseline')).score

    assert score.status is ScoreStatus.SUCCESS
    assert score.value['win_rate'] == 0.75
    assert len(score.metadata['judge_attempts']) == 2
    assert score.metadata['non_official_position_swap'] is True
