"""DocMath preserves the sign of the final numeric answer in CoT output."""

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeContext
from evalscope.api.judge.types import Placement
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus


@pytest.mark.parametrize(
    ('prediction', 'reference', 'extracted', 'expected'),
    [
        ('Therefore, the answer is −262.', -262.0, '-262', 1),
        ('Total debt was $570 million. Therefore, the answer is −262.', -262.0, '-262', 1),
        ('Total debt was $570 million. Therefore, the answer is −262.', 262.0, '-262', 0),
        ('Therefore, the answer is −0.5.', -0.5, '-0.5', 1),
        (r'Therefore, the answer is \boxed{−262}.', -262.0, '-262', 1),
        ('Therefore, the answer is -262.', -262.0, '-262', 1),
    ],
)
def test_numeric_answer_preserves_its_sign(
    prediction: str, reference: float, extracted: str, expected: int
) -> None:
    adapter = get_benchmark('docmath', TaskConfig(datasets=['docmath'], judge={'strategy': 'rule'}))
    sample = adapter.record_to_sample(
        {'question': 'What is the change in debt?', 'paragraphs': ['Debt: 570, then 308.'], 'ground_truth': reference}
    )
    state = TaskState(
        model='offline', sample=sample, output=ModelOutput.from_content('offline', prediction), completed=True
    )

    score = adapter.calculate_metrics(state).score

    assert score.status is ScoreStatus.SUCCESS
    assert score.extracted_prediction == extracted
    assert score.main_value == expected


def test_judge_receives_the_signed_final_answer() -> None:
    adapter = get_benchmark('docmath', TaskConfig(datasets=['docmath']))
    sample = adapter.record_to_sample(
        {'question': 'What is the change in debt?', 'paragraphs': ['Debt: 570, then 308.'], 'ground_truth': -262.0}
    )
    prediction = 'Total debt was $570 million. Therefore, the answer is −262.'
    state = TaskState(
        model='offline', sample=sample, output=ModelOutput.from_content('offline', prediction), completed=True
    )
    context = JudgeContext(
        task_state=state,
        original_prediction=prediction,
        filtered_prediction=adapter.extract_answer(prediction, state),
        reference=sample.target,
    )
    definition = adapter.judge_definition(context)

    request = definition.request(definition.cases[0], Placement.ORIGINAL, [], context)

    assert 'Answer 1: -262.0\nAnswer 2: -262\n' in request.messages[-1].text
