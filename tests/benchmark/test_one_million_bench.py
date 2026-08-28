import json
from typing import Any, List

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem
from evalscope.api.metric import AggScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.one_million_bench.one_million_bench_adapter import OneMillionBenchAdapter
from evalscope.config import TaskConfig
from evalscope.constants import JudgeScoreType, ScoreStatus
from evalscope.metrics.judge.llm_judge import DEFAULT_PROMPT_TEMPLATE, LLMJudge
from evalscope.report import gen_table
from evalscope.report.generator import ReportGenerator


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

    def generate(self, messages: Any) -> ModelOutput:
        self.calls.append(messages)
        return ModelOutput.from_content('scripted', self.replies[min(len(self.calls) - 1, len(self.replies) - 1)])


def make_adapter() -> OneMillionBenchAdapter:
    config = TaskConfig(
        model='mock-model',
        datasets=['one_million_bench'],
        judge={'strategy': 'llm', 'models': [{'model_id': 'judge-model'}]},
    )
    adapter = get_benchmark('one_million_bench', config)
    assert isinstance(adapter, OneMillionBenchAdapter)
    return adapter


def make_record() -> dict:
    return {
        'id': 'sample-id',
        'case_id': 7,
        'language': 'global',
        'system_prompt': '',
        'question': 'Prepare a professional report.',
        'tags': {
            'topics': ['Economics and Finance', 'Investment'],
            'time_sensitivity': {'time_sensitivity': 'Time-agnostic', 'year_month': 'NA', 'day': 'NA'},
        },
        'rubrics': [
            {
                'rubric_number': 1,
                'rubric_detail': 'Includes the required evidence.',
                'rubric_weight': 5,
                'rubric_tag': 'Factual Information',
            },
            {
                'rubric_number': 2,
                'rubric_detail': 'Provides a complete analysis.',
                'rubric_weight': 3,
                'rubric_tag': 'Analytical Reasoning',
            },
            {
                'rubric_number': 3,
                'rubric_detail': 'Contains unsupported claims.',
                'rubric_weight': -2,
                'rubric_tag': 'Factual Information',
            },
        ],
    }


def make_state(adapter: OneMillionBenchAdapter, prediction: str = 'A detailed report.') -> TaskState:
    sample = adapter.record_to_sample(make_record())
    sample.id = 0
    sample.group_id = 0
    return TaskState(
        model='mock-model', sample=sample, output=ModelOutput.from_content('mock-model', prediction), completed=True
    )


def test_registration_and_sample_conversion() -> None:
    adapter = make_adapter()
    record = make_record()
    record['system_prompt'] = 'Act as a financial analyst.'

    sample = adapter.record_to_sample(record)

    assert adapter.dataset_id == 'evalscope/OneMillion-Bench'
    assert adapter.scoring_policy.value == 'judge_only'
    assert sample.subset_key == 'global_economics_and_finance'
    assert isinstance(sample.input[0], ChatMessageSystem)
    assert sample.input[0].text == 'Act as a financial analyst.'
    assert sample.input[1].text == record['question']
    assert json.loads(sample.target) == record['rubrics']
    assert sample.metadata['case_id'] == 7


def test_official_weighted_score_and_negative_penalty() -> None:
    adapter = make_adapter()
    judge = ScriptedJudge(
        [
            json.dumps(
                {
                    'results': [
                        {'rubric_id': 1, 'status': '是', 'justification': 'Evidence is present.'},
                        {'rubric_id': 2, 'status': '否', 'justification': 'Analysis is incomplete.'},
                        {'rubric_id': 3, 'status': '是', 'justification': 'An unsupported claim appears.'},
                    ]
                },
                ensure_ascii=False,
            )
        ]
    )
    adapter.llm_judge = judge

    score = adapter.calculate_metrics(make_state(adapter)).score

    assert score.status is ScoreStatus.SUCCESS
    assert score.value['expert_score'] == pytest.approx(3 / 8)
    assert score.value['pass_rate'] == 0.0
    assert score.main_score_name == 'expert_score'
    assert score.metadata['raw_score'] == 3
    assert score.metadata['max_score'] == 8
    assert 'rubricWeight: -2分' in judge.calls[0][0].text


def test_expert_score_is_clipped_and_pass_threshold_is_inclusive() -> None:
    adapter = make_adapter()
    positive = [
        {'rubric_id': 1, 'status': '是', 'justification': 'hit'},
        {'rubric_id': 2, 'status': '是', 'justification': 'hit'},
        {'rubric_id': 3, 'status': '否', 'justification': 'not hit'},
    ]
    adapter.llm_judge = ScriptedJudge([json.dumps({'results': positive}, ensure_ascii=False)])

    passing_score = adapter.calculate_metrics(make_state(adapter)).score

    assert passing_score.value == {'expert_score': 1.0, 'pass_rate': 1.0}


def test_missing_rubric_verdict_excludes_sample() -> None:
    adapter = make_adapter()
    adapter.llm_judge = ScriptedJudge(
        [
            json.dumps(
                {
                    'results': [
                        {'rubric_id': 1, 'status': '是', 'justification': 'hit'},
                        {'rubric_id': 2, 'status': '否', 'justification': 'miss'},
                    ]
                },
                ensure_ascii=False,
            )
        ]
    )

    score = adapter.calculate_metrics(make_state(adapter)).score

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.metadata['judge_attempts'][0]['status'] == 'parse_error'


@pytest.mark.parametrize(
    ('language', 'domain', 'expected_subset'),
    [
        ('global', 'Economics & Finance', 'global_economics_and_finance'),
        ('cn', '医疗健康', 'cn_healthcare_and_medicine'),
        ('cn', '自然科学', 'cn_natural_sciences'),
    ],
)
def test_observed_domain_aliases_are_normalized(language: str, domain: str, expected_subset: str) -> None:
    adapter = make_adapter()
    record = make_record()
    record['language'] = language
    record['tags']['topics'][0] = domain

    assert adapter.record_to_sample(record).subset_key == expected_subset


def test_report_uses_official_metric_labels_and_sample_weighted_means() -> None:
    adapter = make_adapter()
    score_dict = {
        'global_law': [
            AggScore(score=0.5, metric_name='expert_score', aggregation='mean', num=2),
            AggScore(score=0.5, metric_name='pass_rate', aggregation='mean', num=2),
        ],
        'cn_law': [
            AggScore(score=1.0, metric_name='expert_score', aggregation='mean', num=1),
            AggScore(score=1.0, metric_name='pass_rate', aggregation='mean', num=1),
        ],
    }

    report = ReportGenerator.generate_report(score_dict, 'mock-model', adapter)
    metrics = {metric.identity.name: metric for metric in report.metrics}
    table = gen_table(report_list=[report], add_overall_metric=True)

    assert adapter.aggregation == 'mean'
    assert report.primary_metric_identity == metrics['expert_score'].identity
    assert metrics['expert_score'].score == pytest.approx(2 / 3, abs=1e-4)
    assert metrics['pass_rate'].score == pytest.approx(2 / 3, abs=1e-4)
    assert 'Expert Score ↑' in table
    assert 'Pass Rate ↑' in table
    assert table.count('66.7%') == 2
    assert table.count('OVERALL') == 2
