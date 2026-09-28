"""Exercise chart and instrument rules through the real scoring adapters."""

from typing import Any

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus


def score_answer(name: str, prediction: str, target: str = '', metadata: dict[str, Any] | None = None) -> Any:
    adapter = get_benchmark(name, config=TaskConfig(datasets=[name], judge={'strategy': 'rule'}))
    state = TaskState(
        model='offline',
        sample=Sample(input='Read the chart or instrument.', target=target, metadata=metadata or {}),
        output=ModelOutput.from_content('offline', prediction),
        completed=True,
    )
    return adapter.calculate_metrics(state).score


@pytest.mark.parametrize(
    ('prediction', 'target', 'expected'),
    [
        ('0', '0', 1),
        ('0.0', '0', 1),
        ('-0', '0', 1),
        ('0%', '0', 1),
        ('0', '-0.0', 1),
        ('0.001', '0', 0),
        ('', '0', 0),
        ('105', '100', 1),
        ('105.01', '100', 0),
        ('95', '100', 1),
        ('94.99', '100', 0),
        ('-105', '-100', 1),
        ('18%', '0.18', 1),
        ('10', '10%', 0),
        ('blue', 'Blue', 1),
        ('red', 'Blue', 0),
    ],
)
def test_chartqa_zero_and_existing_relaxed_rules(prediction: str, target: str, expected: int) -> None:
    score = score_answer('chartqa', f'ANSWER: {prediction}', target)
    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'relaxed_acc': expected}


@pytest.mark.parametrize(
    ('prediction', 'number_correct', 'unit_correct'),
    [
        ('Answer: 0.75 m', 1, 1),
        ('Answer: 3/4 m', 1, 1),
        (r'\boxed{\frac{3}{4} m}', 1, 1),
        (r'\boxed{\frac{3}{4}} m', 1, 1),
        (r'Answer: \boxed{\frac{3}{4}} m', 1, 1),
        (r'\fbox{\frac{3}{4}} m', 1, 1),
        (r'\boxed{\boxed{0.75}} m', 1, 1),
        (r'Answer: $\frac{3}{4}$ m', 1, 1),
        (r'Answer: \frac{3}{4}\,\mathrm{m}', 1, 1),
        ('Answer: 0.74 m', 1, 1),
        ('Answer: 0.76 m', 1, 1),
        ('Answer: 0.739 m', 0, 1),
        ('Answer: 0.761 m', 0, 1),
        ('Answer: 0.75 s', 1, 0),
        (r'\boxed{\frac{3}{4}} s', 1, 0),
        (r'\boxed{\frac{3}{4}}', 1, 0),
        ('Earlier reading: 3 m.\n' + r'\boxed{\frac{3}{4}} s', 1, 0),
        (r'\boxed{3 m}, final: \boxed{\frac{3}{4}} m', 1, 1),
        (r'\boxed{0.75 m} \boxed{unfinished', 0, 0),
        (r'\boxed{\frac{3}{4}', 0, 0),
        ('', 0, 0),
        ('No answer', 0, 0),
        (r'\boxed{\frac{3}{0}} m', 0, 1),
    ],
)
def test_measurebench_math_and_units_are_independent(prediction: str, number_correct: int, unit_correct: int) -> None:
    score = score_answer(
        'measure_bench',
        prediction,
        metadata={'evaluator': 'interval_matching', 'evaluator_kwargs': '{"interval": [0.74, 0.76], "units": ["m"]}'},
    )
    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'acc': number_correct * unit_correct, 'number_acc': number_correct, 'unit_acc': unit_correct}


@pytest.mark.parametrize(
    ('prediction', 'metadata', 'expected'),
    [
        ('Answer: 0', {'interval': [0, 0], 'units': []}, {'acc': 1, 'number_acc': 1}),
        ('', {'interval': [0, 0], 'units': []}, {'acc': 0, 'number_acc': 0}),
        (r'\boxed{\frac{3}{4}}', {'interval': [0.75, 0.75], 'units': []}, {'acc': 1, 'number_acc': 1}),
        ('Answer: 0.75 m^2', {'interval': [0.75, 0.75], 'units': ['m^2']}, {'acc': 1, 'number_acc': 1, 'unit_acc': 1}),
        ('Answer: 4.2 mmHg', {'interval': [4.2, 4.2], 'units': ['mmHg']}, {'acc': 1, 'number_acc': 1, 'unit_acc': 1}),
        ('Answer: 11:15:59', {'interval': ['11:15:58', '11:16:00'], 'units': []}, {'acc': 1, 'number_acc': 1}),
        (r'\boxed{11:15:59}', {'interval': ['11:15:58', '11:16:00'], 'units': []}, {'acc': 1, 'number_acc': 1}),
        ('Answer: 11:16:01', {'interval': ['11:15:58', '11:16:00'], 'units': []}, {'acc': 0, 'number_acc': 0}),
        ('No answer', {'interval': ['11:15:58', '11:16:00'], 'units': []}, {'acc': 0, 'number_acc': 0}),
    ],
)
def test_measurebench_zero_time_and_unit_variants(
    prediction: str, metadata: dict[str, Any], expected: dict[str, float]
) -> None:
    import json

    score = score_answer(
        'measure_bench',
        prediction,
        metadata={
            'evaluator': 'interval_matching',
            'evaluator_kwargs': json.dumps(metadata),
        },
    )
    assert score.value == expected


@pytest.mark.parametrize('units', [[], ['A'], [['A'], ['A']]])
def test_measurebench_multiple_intervals(units: list[Any]) -> None:
    import json

    score = score_answer(
        'measure_bench',
        r'\boxed{\frac{19}{2}} A',
        metadata={
            'evaluator': 'multi_interval_matching',
            'evaluator_kwargs': json.dumps({'intervals': [[1, 2], [9.5, 9.7]], 'units': units}),
        },
    )
    expected = {'acc': 1, 'number_acc': 1}
    if units:
        expected['unit_acc'] = 1
    assert score.value == expected


def test_measurebench_multiple_time_intervals() -> None:
    score = score_answer(
        'measure_bench',
        'Answer: 23:15:59',
        metadata={
            'evaluator': 'multi_interval_matching',
            'evaluator_kwargs': '{"intervals": [["11:15:58", "11:16:00"], ["23:15:58", "23:16:00"]], "units": []}',
        },
    )
    assert score.value == {'acc': 1, 'number_acc': 1}


@pytest.mark.parametrize('operation', ['parse_digits', 'extract_boxed_answer_text'])
def test_measurebench_execution_failure_is_not_scored_zero(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.metrics.math import parser
    from evalscope.metrics.math.contracts import MathEvaluationError

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise MathEvaluationError('Mathematical evaluation failed')

    monkeypatch.setattr(parser, operation, fail)
    with pytest.raises(MathEvaluationError, match='evaluation failed'):
        score_answer(
            'measure_bench',
            r'\boxed{\frac{3}{4}} m',
            metadata={
                'evaluator': 'interval_matching',
                'evaluator_kwargs': '{"interval": [0.74, 0.76], "units": ["m"]}',
            },
        )


@pytest.mark.parametrize(
    ('prediction', 'unit', 'lower', 'upper', 'number_correct', 'unit_correct'),
    [
        (r'\boxed{\frac{48}{5} A}', 'A', 9.5, 9.7, 1, 1),
        (r'\boxed{\frac{48}{5}} A', 'A', 9.5, 9.7, 1, 1),
        (r'\boxed{\frac{48}{5} V}', 'A', 9.5, 9.7, 1, 0),
        (r'\boxed{\frac{48}{5}\,\mathrm{A}}', 'A', 9.5, 9.7, 1, 1),
        (r'\boxed{\boxed{9.6 V}} A', 'A', 9.5, 9.7, 1, 1),
        ('Answer: 0.75 V', 'm', 0.74, 0.76, 1, 0),
        (r'\boxed{0.75 m^2}', 'm^2', 0.74, 0.76, 1, 1),
        (r'\boxed{0.75 kg/m^3}', 'kg/m^3', 0.74, 0.76, 1, 1),
        (r'\boxed{0.75\,\mathrm{kg/m^3}}', 'kg/m^3', 0.74, 0.76, 1, 1),
        (r'\boxed{3e-2 A}', 'A', 0.029, 0.031, 1, 1),
        ('Answer: 3e-2 A', 'A', 0.029, 0.031, 1, 1),
        ('Answer: -0.75 A', 'A', -0.76, -0.74, 1, 1),
        (r'Answer: \frac{1}{3A} m', 'm', 0.33, 0.34, 0, 1),
    ],
)
def test_measurebench_unit_labels_do_not_become_math_variables(
    prediction: str, unit: str, lower: float, upper: float, number_correct: int, unit_correct: int
) -> None:
    import json

    score = score_answer(
        'measure_bench',
        prediction,
        metadata={
            'evaluator': 'interval_matching',
            'evaluator_kwargs': json.dumps({'interval': [lower, upper], 'units': [unit]}),
        },
    )
    assert score.value == {'acc': number_correct * unit_correct, 'number_acc': number_correct, 'unit_acc': unit_correct}
