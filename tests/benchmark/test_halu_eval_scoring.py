"""Regression tests for HaluEval verdict matching against the official evaluator.

Official reference: https://github.com/RUCAIBox/HaluEval ``evaluation/evaluate.py``. For every
subset it reads the verdict as

    if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans): incorrect
    elif "Yes" in ans: ans = "Yes"
    elif "No" in ans: ans = "No"
    correct only when ans == ground_truth

so a reply that says "Yes" is never credited for a "No" label and a reply containing both verdicts is
counted as incorrect. The adapter instead checks ``reference in prediction.upper()``, so for a "NO"
label any word containing the letters "no" ("not", "note", "know", "now", ...) makes a "Yes" answer
count as correct, and a reply that says both "Yes" and "No" is credited for either label.
"""

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _state(prediction: str, label: str) -> TaskState:
    sample = Sample(id=0, input='#Your Judgement#:', target=label.upper(), metadata={'answer': label.lower()})
    return TaskState(model='m', sample=sample, output=ModelOutput.from_content('m', prediction), completed=True)


def _score(prediction: str, label: str) -> float:
    adapter = get_benchmark('halueval', TaskConfig(model='m', datasets=['halueval']))
    return adapter.calculate_metrics(_state(prediction, label)).score.main_value


@pytest.mark.parametrize(
    'prediction',
    [
        'Yes. The answer is not supported by the knowledge.',
        'Yes, the response contains information that is now outdated.',
        'Yes. Note that the summary invents a date.',
    ],
)
def test_yes_verdict_is_not_credited_for_a_no_label(prediction: str) -> None:
    assert _score(prediction, 'No') == 0


@pytest.mark.parametrize('label', ['Yes', 'No'])
def test_reply_with_both_verdicts_is_incorrect(label: str) -> None:
    assert _score('Yes and No: parts of it are hallucinated.', label) == 0


@pytest.mark.parametrize(('prediction', 'label'), [('Yes', 'Yes'), ('No', 'No'), ('Yes', 'No'), ('No', 'Yes')])
def test_plain_verdicts_match_the_official_evaluator(prediction: str, label: str) -> None:
    assert _score(prediction, label) == float(prediction == label)


@pytest.mark.parametrize(
    ('prediction', 'label'),
    [
        ('Yes. There is no mention of this date in the document.', 'Yes'),
        ('No, nothing in the answer contradicts the knowledge.', 'No'),
        ('YES', 'Yes'),
        ('NO.', 'No'),
    ],
)
def test_single_verdict_in_explanatory_reply_is_credited(prediction: str, label: str) -> None:
    assert _score(prediction, label) == 1


def test_reply_without_a_verdict_is_incorrect() -> None:
    assert _score('I do not know.', 'No') == 0


def _aggregate(*replies: tuple) -> dict:
    adapter = get_benchmark('halueval', TaskConfig(model='m', datasets=['halueval']))
    scores = [adapter.calculate_metrics(_state(prediction, label)) for prediction, label in replies]
    return {item.metric_name: item.score for item in adapter.aggregate_scores(scores)}


def test_reply_without_a_verdict_is_not_counted_as_yes_for_a_no_label() -> None:
    metrics = _aggregate(('Yes', 'Yes'), ('I do not know.', 'No'))

    assert metrics['accuracy'] == 0.5
    assert metrics['precision'] == 1.0
    assert metrics['yes_ratio'] == 0.5


def test_reply_without_a_verdict_is_a_miss_for_a_yes_label() -> None:
    metrics = _aggregate(('I do not know.', 'Yes'))

    assert metrics['recall'] == 0.0
    assert metrics['yes_ratio'] == 0.0
