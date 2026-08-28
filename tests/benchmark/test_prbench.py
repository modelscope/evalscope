import json
from collections import deque
from typing import Any, Dict, List

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.prbench.prbench_adapter import PRBenchAdapter
from evalscope.config import TaskConfig
from evalscope.constants import JudgeStrategy


class FakeJudge:

    def __init__(self, responses: List[str]) -> None:
        self.responses = deque(responses)
        self.model_id = self.judge_id = 'fake-judge'
        self.calls: List[Any] = []

    def generate(self, messages: Any) -> ModelOutput:
        self.calls.append(messages)
        return ModelOutput.from_content(self.model_id, self.responses.popleft())


def _rubric(title: str, weight_class: str, weight: int, category: str) -> Dict[str, Any]:
    return {
        'id': title.lower().replace(' ', '-'),
        'title': title,
        'annotations': {
            'weight_class': weight_class,
            f'{weight_class.replace(" ", "_")}_weight': weight,
            'criteria_category': category,
        },
    }


def _record() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        'task': 'task-1',
        'turns': 2,
        'field': 'Finance',
        'topic': 'Risk Management',
        'expert': 'Expert',
        'rubric': [
            _rubric('Includes the calculation', 'critically important', 8, 'Financial Accuracy'),
            _rubric('Contains a material error', 'detrimental', -4, 'Financial Accuracy'),
        ],
        'prompt_0': 'First question',
        'response_0': 'First answer',
        'reference_texts_0': ['Source A'],
        'prompt_1': 'Follow-up question',
        'reference_texts_1': [],
        'economic_pathway': 'Risk & Resilience',
        'decision_type': 'Modeling & Measurement',
    }
    for turn in range(2, 10):
        record[f'prompt_{turn}'] = None
        record[f'response_{turn - 1}'] = record.get(f'response_{turn - 1}')
        record[f'reference_texts_{turn}'] = []
    return record


def _adapter() -> PRBenchAdapter:
    config = TaskConfig(
        model='mock-model',
        datasets=['prbench'],
        dataset_args={'prbench': {'subset_list': ['finance']}},
        judge={
            'strategy': JudgeStrategy.LLM,
            'models': {'model_id': 'fake-judge', 'api_url': 'http://localhost:1/v1', 'api_key': 'fake-key'},
        },
    )
    adapter = get_benchmark('prbench', config)
    assert isinstance(adapter, PRBenchAdapter)
    return adapter


def _state(adapter: PRBenchAdapter, prediction: str = '<think>hidden</think>Final answer') -> TaskState:
    sample = adapter.record_to_sample(_record())
    sample.id = 0
    sample.group_id = 0
    return TaskState(
        model='mock-model',
        sample=sample,
        output=ModelOutput.from_content('mock-model', prediction),
        completed=True,
    )


def test_prbench_registration_and_conversation_conversion() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(_record())

    assert adapter.dataset_id == 'ScaleAI/PRBench'
    assert adapter.split_as_subset is True
    assert [message.role for message in sample.input] == ['user', 'assistant', 'user']
    assert sample.input[0].text == 'Reference Text 0:\nSource A\n\nFirst question'
    assert sample.input[-1].text == 'Follow-up question'
    assert sample.metadata['rubrics'] == [
        {
            'id': 'includes-the-calculation',
            'title': 'Includes the calculation',
            'weight': 8.0,
            'category': 'Financial Accuracy',
        },
        {
            'id': 'contains-a-material-error',
            'title': 'Contains a material error',
            'weight': -4.0,
            'category': 'Financial Accuracy',
        },
    ]


def test_prbench_mock_judge_uses_official_weighted_scores_and_prompt() -> None:
    adapter = _adapter()
    judge = FakeJudge([
        json.dumps({'explanation': 'present', 'criteria_met': True}),
        json.dumps({'explanation': 'present', 'criteria_met': True}),
    ])
    adapter.llm_judge = judge

    score = adapter.calculate_metrics(_state(adapter)).score

    assert score.value['clipped_score'] == pytest.approx(0.5)
    assert score.value['normalized_score'] == pytest.approx(2 / 3)
    assert score.main_score_name == 'clipped_score'
    assert score.extracted_prediction == 'Final answer'
    assert score.metadata['weighted_points'] == 4.0
    assert len(judge.calls) == 2
    judge_prompt = judge.calls[0][0].text
    assert 'user: Reference Text 0:\nSource A\n\nFirst question' in judge_prompt
    assert 'assistant: First answer' in judge_prompt
    assert 'user: Follow-up question' in judge_prompt
    assert 'assistant: Final answer' in judge_prompt


def test_prbench_clips_only_the_aggregate_score() -> None:
    adapter = _adapter()
    sample_scores = [
        SampleScore(sample_id=0, score=Score(value={'clipped_score': -0.4, 'normalized_score': 0.2})),
        SampleScore(sample_id=1, score=Score(value={'clipped_score': 0.2, 'normalized_score': 0.6})),
    ]

    aggregated = {score.metric_name: score for score in adapter.aggregate_scores(sample_scores)}

    assert aggregated['clipped_score'].score == 0.0
    assert aggregated['clipped_score'].aggregation == 'clipped_mean'
    assert aggregated['normalized_score'].score == pytest.approx(0.4)
    assert aggregated['normalized_score'].aggregation == 'mean'


def test_prbench_report_uses_official_metric_identities_without_overall() -> None:
    adapter = _adapter()
    subset_scores = {
        subset: adapter.aggregate_scores([
            SampleScore(
                sample_id=index,
                score=Score(value={'clipped_score': 0.25 + index / 10, 'normalized_score': 0.5 + index / 10}),
            )
        ])
        for index, subset in enumerate(['finance', 'legal', 'finance_hard', 'legal_hard'])
    }

    report = adapter._on_generate_report(subset_scores, model_name='mock-model')
    table = report.to_dataframe(add_overall_metric=adapter.add_overall_metric)

    assert report.primary_metric_identity.name == 'clipped_score'
    assert report.primary_metric_identity.aggregation == 'clipped_mean'
    assert set(table['Metric']) == {'clipped_score:clipped_mean', 'normalized_score:mean'}
    assert set(table['Subset']) == set(subset_scores)
    assert 'OVERALL' not in table['Subset'].tolist()
