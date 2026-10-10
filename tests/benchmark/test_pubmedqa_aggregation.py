"""PubMedQA must count unrecognized answers as false negatives in class recall."""

from typing import Dict, List, Tuple

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.pumed_qa.pubmed_qa_adapter import PubMedQAAdapter
from evalscope.config import TaskConfig


@pytest.fixture
def adapter() -> PubMedQAAdapter:
    return get_benchmark('pubmedqa', TaskConfig(datasets=['pubmedqa']))


def aggregate(adapter: PubMedQAAdapter, answers: List[Tuple[str, str]]) -> Dict[str, float]:
    """Exercise record conversion and the normal scoring path with canned completions."""
    scores = []
    for index, (label, answer) in enumerate(answers):
        sample = adapter.record_to_sample(
            {
                'context': 'Study abstract.',
                'question': 'Does the study support the hypothesis?',
                'answer': label.lower(),
                'reasoning': '',
            }
        )
        sample.id = index
        state = TaskState(
            model='scripted', sample=sample, output=ModelOutput.from_content('scripted', answer), completed=True
        )
        scores.append(adapter.calculate_metrics(state))
    return {score.metric_name: score.score for score in adapter.aggregate_scores(scores)}


@pytest.mark.parametrize('unrecognized', ['', 'Insufficient evidence.', 'I am unable to answer.'])
def test_unrecognized_answers_remain_in_recall_denominator(adapter: PubMedQAAdapter, unrecognized: str) -> None:
    answers = [(label, answer) for label in ['YES', 'NO', 'MAYBE'] for answer in [label, unrecognized]]
    metrics = aggregate(adapter, answers)
    assert metrics['accuracy'] == pytest.approx(0.5)
    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(0.5)
    assert metrics['f1'] == pytest.approx(2 / 3)


def test_abstentions_do_not_reverse_f1_ranking(adapter: PubMedQAAdapter) -> None:
    labels = ['YES', 'NO', 'MAYBE']
    a = aggregate(
        adapter, [(label, label if i < 8 else 'Insufficient evidence.') for label in labels for i in range(10)]
    )
    b = aggregate(
        adapter, [(label, label if i < 9 else labels[(j + 1) % 3]) for j, label in enumerate(labels) for i in range(10)]
    )
    assert a['f1'] == pytest.approx(8 / 9)
    assert b['f1'] == pytest.approx(0.9)
    assert a['f1'] < b['f1']


def test_unrecognized_answers_count_against_their_own_class(adapter: PubMedQAAdapter) -> None:
    answers = [('YES', 'YES'), ('YES', ''), ('YES', ''), ('NO', 'NO'), ('MAYBE', 'MAYBE')]
    metrics = aggregate(adapter, answers)
    assert metrics['recall'] == pytest.approx((1 / 3 + 1 + 1) / 3)
    assert metrics['f1'] == pytest.approx((0.5 + 1 + 1) / 3)


@pytest.mark.parametrize('mode', ['correct', 'wrong', 'abstain', 'empty'])
def test_existing_boundary_cases(adapter: PubMedQAAdapter, mode: str) -> None:
    labels = ['YES', 'NO', 'MAYBE']
    predictions = {'correct': labels, 'wrong': ['NO', 'MAYBE', 'YES'], 'abstain': ['', '', ''], 'empty': []}
    metrics = aggregate(adapter, list(zip(labels, predictions[mode])))
    expected = 1.0 if mode == 'correct' else 0.0
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        assert metrics[metric] == pytest.approx(expected)
