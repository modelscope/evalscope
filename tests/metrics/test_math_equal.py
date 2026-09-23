"""math_equal compares a number against reference / 100 and reference * 100 only when a percent sign is written.

Building those alternatives for every numeric pair scored any answer off by exactly 100x as correct,
e.g. 1800 for a gsm8k reference of 18.
"""
import pytest

from evalscope.metrics.math.parser import math_equal
from evalscope.metrics.nlp.metrics import Accuracy


@pytest.mark.parametrize(
    ('prediction', 'reference'),
    [('1800', '18'), ('500', '5'), ('0.05', '5'), ('42', '4200'), ('0.18', '18')],
)
def test_numbers_off_by_100x_are_not_equal(prediction: str, reference: str) -> None:
    assert not math_equal(prediction, reference)


@pytest.mark.parametrize(
    ('prediction', 'reference'),
    [
        ('18', '18'),
        ('18.0', '18'),
        ('0.1', '10\\%'),
        ('10', '10\\%'),
        ('10%', '0.1'),
        ('10%', '10'),
        ('18%', '0.18'),
    ],
)
def test_equal_numbers_and_percent_forms_are_equal(prediction: str, reference: str) -> None:
    assert math_equal(prediction, reference)


def test_numeric_accuracy_rejects_answers_off_by_100x() -> None:
    assert Accuracy(numeric=True).apply(
        ['1800', '42', '18', '10'], ['18', '4200', '18', '10\\%']
    ) == [0.0, 0.0, 1.0, 1.0]
